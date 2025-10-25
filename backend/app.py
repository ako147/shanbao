import os, sqlite3
from typing import List, Tuple
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

from google import genai
from google.genai import types
from google.genai import errors

# ================== 路徑與環境 ==================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")   # 你的前端資料夾
KB_DIR    = os.path.join(BASE_DIR, "kb_raw")                      # 固定知識庫
DB_PATH   = os.path.join(BASE_DIR, "embeddings.db")               # SQLite 向量庫

_embed_model = None
_tokenizer = None

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 未設定，請在 backend/.env 或系統環境變數中設定")

# === 模型設定（集中）===
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-004")
GEN_MODEL     = os.getenv("GEN_MODEL", "gemini-2.0-flash")        # 可改 .env 切換
GENAI_VERSION = os.getenv("GENAI_API_VERSION", "v1alpha")              # 你的 google-genai 版本對應

# ================== 規則與分塊 ==================
GEM_RULES = """
回答一律用繁體中文與條列結構。
若檢索內容不足以支持答案，請明確說明「查無相關資料」，並提出可行查找方向。
若問題與檢索內容不相關，請告訴民眾「請訊問與張市長相關問題」，提列舉幾個與檢索內容有關的問題供民眾參考。
盡量解釋專有名詞，讓非專業者也能理解。
如有引用檢索片段，請在句尾標示 [1]、[2]… 對應檢索清單。
"""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 120

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    toks = text.split()
    out = []
    i = 0
    step = max(1, size - overlap)
    while i < len(toks):
        out.append(" ".join(toks[i:i+size]))
        i += step
    return out or [text]

# ================== SQLite 工具 ==================
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id     TEXT PRIMARY KEY,
        doc_id TEXT,
        text   TEXT,
        embed  BLOB
    )
    """)
    cur.execute("""CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)""")
    con.commit()
    con.close()

def add_rows(rows: List[Tuple[str, str, str, np.ndarray]]):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    for rid, did, text, vec in rows:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        cur.execute(
            "INSERT OR REPLACE INTO docs(id, doc_id, text, embed) VALUES(?,?,?,?)",
            (rid, did, text, blob),
        )
    con.commit()
    con.close()

def all_embeddings():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, doc_id, text, embed FROM docs")
    rows = cur.fetchall()
    con.close()
    if not rows:
        # 將在啟動時以 probe 自動偵測真正維度
        return [], [], [], np.zeros((0, 1), dtype=np.float32)
    ids, docids, texts, vecs = [], [], [], []
    for rid, did, txt, blob in rows:
        ids.append(rid); docids.append(did); texts.append(txt)
        vecs.append(np.frombuffer(blob, dtype=np.float32))
    return ids, docids, texts, np.vstack(vecs)

def get_meta(key: str) -> str | None:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT v FROM meta WHERE k=?", (key,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else None

def set_meta(key: str, val: str):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (key, val))
    con.commit(); con.close()

def cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int):
    if mat.shape[0] == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return idx.tolist()

# ================== Gemini Client & 包裝 ==================
# 一個 client 用預設版本（多半用於 embeddings）
gen = genai.Client(api_key=API_KEY)
# 另一個 client 可指定 API 版本（用於生成；避免你不同環境的版本不一致）
gen_chat = genai.Client(api_key=API_KEY, http_options={'api_version': GENAI_VERSION})

def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """依據 EMBED_BACKEND 決定用 HuggingFace 或 Google GenAI 產生 embedding"""
    backend = os.getenv("EMBED_BACKEND", "google").lower()

    if backend == "huggingface":
        global _embed_model, _tokenizer
        if _embed_model is None:
            print(f"🔹 使用 HuggingFace 模型：{os.getenv('EMBED_MODEL')}")
            _tokenizer = AutoTokenizer.from_pretrained(os.getenv("EMBED_MODEL"))
            _embed_model = AutoModel.from_pretrained(os.getenv("EMBED_MODEL"))
        _embed_model.eval()
        with torch.no_grad():
            inputs = _tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = _embed_model(**inputs)
            # 取最後隱藏層平均值
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return [emb.numpy().astype(np.float32) for emb in embeddings]

    # 預設使用 Google GenAI
    res = gen.models.embed_content(model=os.getenv("EMBED_MODEL", "text-embedding-004"),
                                   contents=texts)
    return [np.array(e.values, dtype=np.float32) for e in res.embeddings]


GEN_MODEL_CANDIDATES = [
    lambda: os.getenv("GEN_MODEL", GEN_MODEL),  # 先用 .env 指定值
    lambda: "gemini-2.0-flash-002",
    lambda: "gemini-1.5-flash",
    lambda: "gemini-1.5-flash-8b",
    lambda: "gemini-1.5-pro",
]

def stream_generate(query: str, sys_inst: str, temperature: float):
    """嘗試多個型號，第一個能用的就串流輸出"""
    last_err = None
    tried = set()
    for get_name in GEN_MODEL_CANDIDATES:
        name = get_name()
        if not name or name in tried:
            continue
        tried.add(name)
        try:
            stream = gen_chat.models.generate_content_stream(
                model=name,
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    temperature=temperature,
                ),
                contents=query,  # 直接給字串
            )
            for ev in stream:
                cand = (ev.candidates or [None])[0]
                if cand and cand.content and cand.content.parts:
                    for p in cand.content.parts:
                        if getattr(p, "text", None):
                            yield p.text
            return
        except errors.ClientError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("No generation model worked.")

# ================== FastAPI ==================
class ChatReq(BaseModel):
    query: str
    top_k: int = 5
    temperature: float = 0.3
    # gen_model: str | None = None  # 若要讓前端臨時指定，可加這欄並調整上面候選清單

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 啟動：建 DB + 檢查/重建索引 ----
    init_db()

    # 以最短文本探測當前 EMBED_MODEL 的維度
    probe_dim = len(embed_texts(["_dim_probe_"])[0])
    db_dim = get_meta("embed_dim")
    if db_dim is None:
        set_meta("embed_dim", str(probe_dim))
    elif int(db_dim) != probe_dim:
        print(f"⚠️ 偵測到嵌入維度改變：DB={db_dim}, MODEL={probe_dim} → 重新建立 embeddings.db")
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        init_db()
        set_meta("embed_dim", str(probe_dim))

    # 若尚未建立索引，從 kb_raw 載入 txt/md 檔案嵌入
    ids, _, _, _ = all_embeddings()
    if not ids:
        os.makedirs(KB_DIR, exist_ok=True)
        files = [f for f in os.listdir(KB_DIR) if f.lower().endswith((".txt", ".md"))]
        if files:
            for fname in files:
                path = os.path.join(KB_DIR, fname)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                chunks = chunk_text(text)
                vecs = embed_texts(chunks)
                rows = [(f"{fname}-{i}", fname, chunks[i], vecs[i]) for i in range(len(chunks))]
                add_rows(rows)
                print(f"已建立知識庫：{fname}（{len(chunks)} 分塊）")
            print("✅ 固定知識庫載入完成。")
        else:
            print("⚠️ kb_raw 內沒有 .txt/.md 檔案，知識庫為空。")
    else:
        print("已存在 embedding 知識庫，略過重建。")

    yield  # ---- 服務開始 ----

    # ---- 關閉：可在此釋放資源 ----
    # print("服務關閉")

app = FastAPI(title="Gemini RAG (Fixed KB, NumPy/SQLite)", lifespan=lifespan)

# CORS（同網域即可；若跨網域再限縮 allow_origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載整個前端資料夾到 /assets
if os.path.isdir(FRONT_DIR):
    app.mount("/assets", StaticFiles(directory=FRONT_DIR), name="assets")

# 首頁：回傳 people.html
@app.get("/")
def serve_people():
    return FileResponse(os.path.join(FRONT_DIR, "people.html"))

# 聊天（串流）
@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    # 1) 取矩陣
    _, _, texts, mat = all_embeddings()

    # 2) 查詢向量
    q_vec = embed_texts([req.query])[0]

    # 3) Top-k 片段
    idxs = cosine_topk(q_vec, mat, req.top_k)
    contexts = [texts[i] for i in idxs]
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) if contexts else "（無檢索片段）"

    # 4) system instruction
    sys_inst = (
        GEM_RULES
        + "\n\n以下是檢索到的片段，必要時可引用：\n"
        + "=== 檢索片段 ===\n"
        + context_block
    )

    # 5) 串流回傳
    def stream_gen():
        yield from stream_generate(req.query, sys_inst, req.temperature)
    return StreamingResponse(stream_gen(), media_type="text/plain")

# 清庫（重建用）
@app.post("/reset")
async def reset():
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    except Exception:
        pass
    init_db()
    return {"ok": True}
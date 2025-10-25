新增知識庫方式:
將文件上傳至kb.raw資料夾，刪除舊的embedding.db
重新啟動後就會建立新的embedding.db

啟動方式:
在powershell
pip install -r requirements.txt 第一次執行才要
進入backend資料夾
# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1     # 進入 venv
uvicorn app:app --reload --port 8000
出現「Application startup complete.」後開啟瀏覽器「http://localhost:8000/」

離開虛擬環境:
deactivate
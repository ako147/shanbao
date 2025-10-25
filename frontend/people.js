let currentMode = 'text';
// === 新增：後端 API 位址 ===
// const API_BASE = ""; // 同網域反向代理時留空字串
const API_BASE = ""; // 若本機直連 FastAPI，改用這行


// DOM
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const transferBtn = document.getElementById('transferBtn');

// 小表情池（讓對話更可愛）
const cuteEmojis = ['✨','🌟','💡','💬','🎈','🍀','👍','🙌','😊','🧡','📌'];

// 轉義使用者輸入，避免插入 HTML
function escapeHTML(str){
  return str.replace(/[&<>"']/g, m => (
    { "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]
  ));
}

// 發送訊息
// function sendMessage() {
//   const raw = messageInput.value;
//   const message = raw.trim();
//   if (!message) return;

//   addMessage(escapeHTML(message), 'user', {asHTML:false});
//   messageInput.value = '';

//   setTimeout(() => {
//     showTyping();
//     setTimeout(() => {
//       removeTyping();
//       const responses = [
//         '謝謝您的提問！以下是我對此政策的重點說明：',
//         '這是很重要的公共議題，我們的方案聚焦在三個方向：',
//         '我整理了市民最關心的重點，提供您參考：',
//         '好的～我用簡單圖解帶你快速理解：'
//       ];
//       const randomResponse = responses[Math.floor(Math.random() * responses.length)];
//       const emoji = cuteEmojis[Math.floor(Math.random()*cuteEmojis.length)];
//       addMessage(`${randomResponse} ${emoji}`, 'ai', {asHTML:true});

//       generateMultimediaResponse(message);
//     }, 1200);
//   }, 300);
// }

// === 取代原本的 sendMessage：改為呼叫 RAG 串流 ===
async function sendMessage() {
  const raw = messageInput.value;
  const message = raw.trim();
  if (!message) return;

  // 1) 顯示使用者訊息
  addMessage(escapeHTML(message), 'user', {asHTML:false});
  messageInput.value = '';

  // 2) 顯示「正在思考中」
  showTyping();

  try {
    // 3) 發送到後端，啟動串流
    const resp = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: message, top_k: 5, temperature: 0.3 })
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`);
    }

    // 4) 先移除「正在思考中」，插入一個空白的 AI 訊息框，邊收邊填
    removeTyping();
    const aiDiv = document.createElement('div');
    aiDiv.className = 'message ai';
    aiDiv.innerHTML = `
      <div class="message-avatar ai-avatar">政</div>
      <div class="message-content ai-message"><span class="stream"></span></div>
    `;
    chatMessages.appendChild(aiDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    const streamSpan = aiDiv.querySelector('.stream');

    // 5) 串流讀取：逐塊追加到同一則訊息
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      streamSpan.textContent += chunk;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 6) 串流結束（可選：在最後補上引用標註處理等）
  } catch (err) {
    // 出錯就把「正在思考中」拿掉並顯示錯誤訊息
    removeTyping();
    addMessage(`系統忙線或連線失敗，請稍後再試。\n(${err.message})`, 'ai', {asHTML:false});
  }
}


function addMessage(content, sender, opts={asHTML:false}) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${sender}`;
  
  const avatar = document.createElement('div');
  avatar.className = `message-avatar ${sender}-avatar`;
  avatar.textContent = sender === 'ai' ? '政' : '我';

  const contentDiv = document.createElement('div');
  contentDiv.className = `message-content ${sender}-message`;
  if (opts.asHTML) contentDiv.innerHTML = content;
  else contentDiv.textContent = content;

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(contentDiv);
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
  const typingDiv = document.createElement('div');
  typingDiv.className = 'message typing-message';
  typingDiv.innerHTML = `
    <div class="message-avatar ai-avatar">政</div>
    <div class="message-content ai-message loading">
      <span>正在思考中</span>
      <div class="loading-dots">
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
      </div>
    </div>
  `;
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
function removeTyping() {
  const typingMessage = document.querySelector('.typing-message');
  if (typingMessage) typingMessage.remove();
}

// 多媒體回應（可愛卡片；不插入任何照片）
function generateMultimediaResponse(message) {
  const shouldGenerateImage = /數據|統計|圖|成效|KPI/.test(message) || Math.random() < 0.25;

  if (shouldGenerateImage) {
    setTimeout(() => {
      const imageMessage = document.createElement('div');
      imageMessage.className = 'message';
      imageMessage.innerHTML = `
        <div class="message-avatar ai-avatar">政</div>
        <div class="message-content ai-message">
          這裡用一張可愛小圖幫你看重點 📊
          <div class="media-content">
            <div style="width: 230px; height: 140px; background: #fff; border: 2px dashed #cfe8ff; border-radius: 12px; display:flex; align-items:center; justify-content:center; color:#6c7a89; font-size:12px;">
              <div style="text-align:center; line-height:1.4;">
                <div style="font-size:20px; margin-bottom:6px;">📈 政策成效圖</div>
                <div>指標 A ↑　指標 B →　指標 C ↑</div>
                <div style="font-size:10px; opacity:.7; margin-top:6px;">（示意圖，實際數據以公告為準）</div>
              </div>
            </div>
          </div>
        </div>
      `;
      chatMessages.appendChild(imageMessage);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 800);
  }
}

// 綁定事件
document.addEventListener('DOMContentLoaded', () => {
  sendBtn.addEventListener('click', sendMessage);
  messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); sendMessage();
    }
  });

  //（已移除模式按鈕的綁定）


  // 標籤一鍵發問
  document.querySelectorAll('.tag').forEach(tag => {
    tag.addEventListener('click', () => {
      const topic = tag.textContent.trim();
      messageInput.value = `請用重點圖解方式，說明「${topic}」的近期進度與成效。`;
      sendMessage();
    });
  });

  // 自動高度
  messageInput.addEventListener('input', function(){
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 110) + 'px';
  });
});

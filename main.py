import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_agent = None
_agent_error: str | None = None

CHAT_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SHL Assessment Recommender</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5;
           display: flex; flex-direction: column; height: 100vh; }
    header { background: #c8102e; color: white; padding: 14px 24px;
             font-size: 1.1rem; font-weight: 600; }
    #chat-window { flex: 1; overflow-y: auto; padding: 20px 24px;
                   display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 72%; padding: 10px 14px; border-radius: 12px;
           line-height: 1.5; font-size: .93rem; white-space: pre-wrap; }
    .msg.user  { background: #c8102e; color: white; align-self: flex-end; }
    .msg.bot   { background: white; color: #222; align-self: flex-start;
                 box-shadow: 0 1px 3px rgba(0,0,0,.1); }
    .msg.error { background: #ffe0e0; color: #b00; }
    table.recs { width: 100%; border-collapse: collapse; margin-top: 8px; font-size:.85rem; }
    table.recs th { background:#f5f5f5; padding:6px 10px; text-align:left;
                    border-bottom:2px solid #ddd; }
    table.recs td { padding:6px 10px; border-bottom:1px solid #eee; }
    table.recs a  { color:#c8102e; text-decoration:none; font-weight:500; }
    table.recs a:hover { text-decoration:underline; }
    .badge { background:#e8f0fe; color:#1a73e8; border-radius:4px;
             padding:1px 6px; font-size:.78rem; font-weight:600; }
    #input-bar { display:flex; gap:10px; padding:14px 24px;
                 background:white; border-top:1px solid #ddd; }
    #user-input { flex:1; padding:10px 14px; border:1px solid #ccc;
                  border-radius:8px; font-size:.93rem; outline:none;
                  resize:none; max-height:120px; }
    #user-input:focus { border-color:#c8102e; }
    #send-btn { background:#c8102e; color:white; border:none; padding:10px 20px;
                border-radius:8px; cursor:pointer; font-size:.93rem; font-weight:600; }
    #send-btn:disabled { background:#ccc; cursor:default; }
    .typing { color:#888; font-style:italic; font-size:.85rem; }
  </style>
</head>
<body>
  <header>SHL Assessment Recommender</header>
  <div id="chat-window"></div>
  <div id="input-bar">
    <textarea id="user-input" rows="1"
              placeholder="Describe the role you're hiring for..."></textarea>
    <button id="send-btn">Send</button>
  </div>

<script>
  const win      = document.getElementById('chat-window');
  const input    = document.getElementById('user-input');
  const sendBtn  = document.getElementById('send-btn');
  let history    = [];

  function scrollDown() { win.scrollTop = win.scrollHeight; }

  function addMsg(role, text, recs) {
    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
    div.textContent = text;

    if (recs && recs.length > 0) {
      const tbl = document.createElement('table');
      tbl.className = 'recs';
      tbl.innerHTML = '<thead><tr><th>#</th><th>Assessment</th><th>Type</th></tr></thead>';
      const tb = document.createElement('tbody');
      recs.forEach((r, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${i+1}</td>
          <td><a href="${r.url}" target="_blank">${r.name}</a></td>
          <td><span class="badge">${r.test_type}</span></td>`;
        tb.appendChild(tr);
      });
      tbl.appendChild(tb);
      div.appendChild(document.createElement('br'));
      div.appendChild(tbl);
    }
    win.appendChild(div);
    scrollDown();
  }

  function showTyping() {
    const d = document.createElement('div');
    d.className = 'msg bot typing';
    d.id = 'typing';
    d.textContent = 'Thinking...';
    win.appendChild(d);
    scrollDown();
  }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    sendBtn.disabled = true;
    addMsg('user', text);
    history.push({ role: 'user', content: text });
    showTyping();

    try {
      const res  = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history }),
      });
      document.getElementById('typing')?.remove();

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const d = document.createElement('div');
        d.className = 'msg bot error';
        d.textContent = 'Error: ' + (err.detail || res.statusText);
        win.appendChild(d);
        scrollDown();
        sendBtn.disabled = false;
        return;
      }

      const data = await res.json();
      addMsg('assistant', data.reply, data.recommendations);
      history.push({ role: 'assistant', content: data.reply });

      if (data.end_of_conversation) {
        const d = document.createElement('div');
        d.className = 'msg bot';
        d.style.background = '#e6f4ea';
        d.textContent = 'Conversation complete. Refresh to start again.';
        win.appendChild(d); scrollDown();
        return;
      }
    } catch (e) {
      document.getElementById('typing')?.remove();
      const d = document.createElement('div');
      d.className = 'msg bot error';
      d.textContent = 'Network error: ' + e.message;
      win.appendChild(d); scrollDown();
    }
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = input.scrollHeight + 'px';
  });
</script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _agent_error
    try:
        from agent import SHLAgent
        _agent = SHLAgent()
        logger.info("Agent ready.")
    except Exception as exc:
        _agent_error = str(exc)
        logger.error("Agent failed to start: %s", exc)
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TIMEOUT_SECONDS = 28


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages list must not be empty")
        return v


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


@app.get("/", response_class=HTMLResponse)
def root():
    return CHAT_UI_HTML


@app.get("/health")
def health():
    if _agent_error:
        raise HTTPException(status_code=503, detail=f"Agent init failed: {_agent_error}")
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready yet")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    messages = [m.model_dump() for m in request.messages]

    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _agent.respond, messages),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent timed out (>28 s)")
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result
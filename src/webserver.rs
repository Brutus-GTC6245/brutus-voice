use std::convert::Infallible;
use std::time::Duration;

use axum::{
    Router,
    extract::State,
    response::{Html, Sse, sse::Event},
    routing::get,
};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tracing::info;

use crate::state::{AppState, ChatMsg};

// ── Embedded UI ───────────────────────────────────────────────────────────────

const HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brutus Voice</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
  header{background:#16213e;padding:16px 24px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #0f3460;box-shadow:0 2px 8px rgba(0,0,0,.3)}
  .avatar{width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#e94560,#0f3460);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#fff;flex-shrink:0}
  header h1{font-size:18px;font-weight:600}
  header p{font-size:12px;color:#888}
  .status{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12px;color:#888}
  .dot{width:8px;height:8px;border-radius:50%;background:#4caf50;box-shadow:0 0 6px #4caf50;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  #chat{flex:1;overflow-y:auto;padding:20px 16px;display:flex;flex-direction:column;gap:12px;scroll-behavior:smooth}
  .row{display:flex;align-items:flex-end;gap:8px}
  .row.user{flex-direction:row-reverse}
  .bav{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;margin-bottom:2px}
  .row.user .bav{background:#0f3460;color:#90caf9}
  .row.assistant .bav{background:#e94560;color:#fff}
  .bub{max-width:72%;padding:10px 14px;border-radius:18px;font-size:14px;line-height:1.5}
  .row.user .bub{background:#0f3460;color:#e3f2fd;border-bottom-right-radius:4px}
  .row.assistant .bub{background:#1e1e3a;color:#f0f0f0;border-bottom-left-radius:4px;border:1px solid #2a2a4a}
  .ts{font-size:10px;color:#666;margin-top:4px;text-align:right}
  .row.user .ts{color:#90caf9aa}
  .empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#444;gap:12px;font-size:14px;text-align:center}
  .empty .icon{font-size:48px}
  footer{padding:10px 24px;background:#16213e;border-top:1px solid #0f3460;font-size:11px;color:#555;text-align:center}
  .listening-bar{height:3px;background:linear-gradient(90deg,#e94560,#0f3460,#e94560);background-size:200%;animation:slide 1.5s linear infinite;display:none}
  @keyframes slide{0%{background-position:0%}100%{background-position:200%}}
</style>
</head>
<body>
<header>
  <div class="avatar">B</div>
  <div><h1>Brutus Voice</h1><p id="sub">Say "Hey Jarvis" to wake</p></div>
  <div class="status"><span id="status-text">waiting</span><div class="dot" id="dot"></div></div>
</header>
<div class="listening-bar" id="lbar"></div>
<div id="chat">
  <div class="empty" id="empty">
    <div class="icon">🎙️</div>
    <div>Say <strong>"Hey Jarvis"</strong> to start a conversation</div>
  </div>
</div>
<footer>brutus-voice · port 8080 · <span id="count">0</span> messages</footer>
<script>
const chat=document.getElementById('chat'),empty=document.getElementById('empty'),
      count=document.getElementById('count'),dot=document.getElementById('dot'),
      lbar=document.getElementById('lbar'),sub=document.getElementById('sub'),
      stxt=document.getElementById('status-text');
let n=0;
function addBubble(msg){
  if(empty.parentNode)empty.remove();
  n++;count.textContent=n;
  const row=document.createElement('div');row.className='row '+msg.role;
  const av=document.createElement('div');av.className='bav';
  av.textContent=msg.role==='user'?'B':'🤖';
  const bub=document.createElement('div');bub.className='bub';
  const txt=document.createElement('div');txt.textContent=msg.content;
  const ts=document.createElement('div');ts.className='ts';ts.textContent=msg.timestamp;
  bub.appendChild(txt);bub.appendChild(ts);
  row.appendChild(av);row.appendChild(bub);
  chat.appendChild(row);chat.scrollTop=chat.scrollHeight;
  dot.style.background='#e94560';dot.style.boxShadow='0 0 10px #e94560';
  setTimeout(()=>{dot.style.background='#4caf50';dot.style.boxShadow='0 0 6px #4caf50'},600);
}
fetch('/history').then(r=>r.json()).then(msgs=>msgs.forEach(addBubble));
const es=new EventSource('/events');
es.addEventListener('message',e=>addBubble(JSON.parse(e.data)));
es.addEventListener('status',e=>{
  const s=JSON.parse(e.data);
  stxt.textContent=s.state;
  sub.textContent=s.state==='listening'?'Listening…':s.state==='wake'?'Wake word detected!':'Say "Hey Jarvis" to wake';
  lbar.style.display=s.state==='listening'?'block':'none';
});
es.onerror=()=>{dot.style.background='#f44336';dot.style.boxShadow='0 0 6px #f44336'};
</script>
</body>
</html>"#;

// ── Route handlers ────────────────────────────────────────────────────────────

async fn root() -> Html<&'static str> {
    Html(HTML)
}

async fn history(State(state): State<AppState>) -> axum::Json<Vec<ChatMsg>> {
    axum::Json(state.history.lock().unwrap().clone())
}

async fn events(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| {
        res.ok().map(|msg| {
            let data = serde_json::to_string(&msg).unwrap_or_default();
            Ok(Event::default().data(data))
        })
    });
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

// ── Server ────────────────────────────────────────────────────────────────────

/// Bind and serve the web UI on `0.0.0.0:{port}`.
/// Spawns the server as a background Tokio task and returns immediately.
pub async fn serve(port: u16, state: AppState) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/",        get(root))
        .route("/history", get(history))
        .route("/events",  get(events))
        .with_state(state);

    let addr     = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding web server to {addr}"))?;

    info!("web UI → http://localhost:{port}");

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    Ok(())
}

// bring Context into scope for the with_context call above
use anyhow::Context;

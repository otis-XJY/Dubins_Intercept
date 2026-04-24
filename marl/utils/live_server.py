import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse


class LiveFrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._status: Dict[str, Any] = {}
        self._updated_at = 0.0

    def update(self, *, jpeg: Optional[bytes] = None, status: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if jpeg is not None:
                self._jpeg = jpeg
                self._updated_at = time.time()
            if status is not None:
                self._status = dict(status)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def get_updated_at(self) -> float:
        with self._lock:
            return float(self._updated_at)


def _index_html(stream_path: str = "/stream.mjpg") -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Dubins Live</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: #0b0f19;
      color: #e6e6e6;
    }}
    .wrap {{
      --side-width: 340px;
      --col-gap: 1px;
      --divider-w: 10px;
      display: grid;
      grid-template-columns: 1fr var(--divider-w) var(--side-width);
      gap: var(--col-gap);
      padding: 16px;
      height: 100vh;
      box-sizing: border-box;
    }}
    .divider {{
      width: var(--divider-w);
      border-radius: 999px;
      cursor: col-resize;
      user-select: none;
      background: rgba(59, 130, 246, 0.35);
      border: 1px solid rgba(59, 130, 246, 0.65);
      box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }}
    .divider:active {{
      background: rgba(59, 130, 246, 0.55);
    }}
    .card {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    .viewer {{
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 8px;
    }}
    img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #0b0f19;
    }}
    .side {{
      height: 100%;
      display: flex;
      flex-direction: column;
      padding: 12px 12px 12px 12px;
      box-sizing: border-box;
    }}
    .kv {{
      display: grid;
      grid-template-columns: minmax(190px, 320px) 1fr;
      column-gap: 14px;
      row-gap: 8px;
      font-size: 13px;
      line-height: 1.4;
      padding: 10px 0 0 0;
      flex: 1 1 auto;
      overflow: auto;
    }}
    .k {{
      color: rgba(230,230,230,0.75);
      font-size: 12px;
      line-height: 1.25;
      white-space: normal;
      word-break: break-word;
      overflow-wrap: anywhere;
      padding-right: 6px;
      align-self: start;
    }}
    .v {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New";
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font-size: 11px;
      line-height: 1.25;
      align-self: start;
    }}
    .hint {{
      font-size: 12px;
      color: rgba(230,230,230,0.65);
      padding: 10px 0 12px 0;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      font-size: 12px;
      margin-right: 6px;
    }}
    /* sliders removed */
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card viewer">
      <img id="live" src="{stream_path}" alt="live stream"/>
    </div>
    <div id="divider" class="divider" title="按住拖拽调整左右栏宽度"></div>
    <div class="card side">
      <div class="hint">
        <span class="pill">/stream.mjpg</span>
        <span class="pill">/status.json</span>
      </div>
      <div id="kv" class="kv"></div>
      <div class="hint">状态每 0.5s 轮询一次。若无画面，请确认训练已启动且 render_mode=rgb_array。</div>
    </div>
  </div>
  <script>
    function isPlainObject(x) {{
      return x !== null && typeof x === 'object' && !Array.isArray(x);
    }}

    function summarizeArray(arr) {{
      const n = arr.length;
      if (n <= 60) return JSON.stringify(arr);
      // Special-case boolean arrays (e.g. Capflag_full) to show compact info.
      const allBool = arr.every(v => typeof v === 'boolean');
      if (allBool) {{
        const trueIdx = [];
        for (let i = 0; i < n; i++) {{
          if (arr[i] === true) trueIdx.push(i);
          if (trueIdx.length >= 30) break;
        }}
        const trueCount = arr.reduce((a, v) => a + (v === true ? 1 : 0), 0);
        return `BoolArray(len=${{n}}, true=${{trueCount}}, trueIdx(sample)=${{JSON.stringify(trueIdx)}})`;
      }}
      const head = arr.slice(0, 20);
      return `Array(len=${{n}}, head=${{JSON.stringify(head)}} ...)`;
    }}

    function prettyValue(v) {{
      if (v === null || v === undefined) return String(v);
      if (Array.isArray(v)) return summarizeArray(v);
      if (isPlainObject(v)) return JSON.stringify(v, null, 2);
      return String(v);
    }}

    function flatten(obj, prefix = '', out = {{}}) {{
      if (!isPlainObject(obj)) {{
        out[prefix || 'value'] = obj;
        return out;
      }}
      const keys = Object.keys(obj).sort();
      for (const k of keys) {{
        const v = obj[k];
        const kp = prefix ? `${{prefix}}.${{k}}` : k;
        if (isPlainObject(v)) {{
          flatten(v, kp, out);
        }} else {{
          out[kp] = v;
        }}
      }}
      return out;
    }}

    function setupResizableDivider() {{
      const wrap = document.querySelector('.wrap');
      const divider = document.getElementById('divider');

      // Drag to resize side width (same effect as moving the slider)
      if (divider && wrap) {{
        let dragging = false;
        let startX = 0;
        let startW = 0;
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

        divider.addEventListener('mousedown', (e) => {{
          dragging = true;
          startX = e.clientX;
          // current CSS var, fallback to computed width
          const cssW = getComputedStyle(wrap).getPropertyValue('--side-width').trim();
          startW = Number(cssW.replace('px','')) || 340;
          document.body.style.cursor = 'col-resize';
          document.body.style.userSelect = 'none';
          e.preventDefault();
        }});

        window.addEventListener('mousemove', (e) => {{
          if (!dragging) return;
          const dx = e.clientX - startX;
          // Move right -> increase side width
          let w = startW + dx;
          const minW = 260;
          const maxW = Math.min(720, Math.floor(wrap.getBoundingClientRect().width * 0.65));
          w = clamp(w, minW, maxW);
          wrap.style.setProperty('--side-width', `${{Math.round(w)}}px`);
        }});

        window.addEventListener('mouseup', () => {{
          if (!dragging) return;
          dragging = false;
          document.body.style.cursor = '';
          document.body.style.userSelect = '';
        }});
      }}
    }}

    async function poll() {{
      try {{
        const r = await fetch('/status.json', {{cache: 'no-store'}});
        if (!r.ok) return;
        const s = await r.json();
        const flat = flatten(s);
        const keys = Object.keys(flat).sort();
        const kv = document.getElementById('kv');
        kv.innerHTML = '';
        for (const k of keys) {{
          const dk = document.createElement('div');
          dk.className = 'k';
          dk.textContent = k;
          const dv = document.createElement('div');
          dv.className = 'v';
          dv.textContent = prettyValue(flat[k]);
          kv.appendChild(dk);
          kv.appendChild(dv);
        }}
      }} catch (e) {{}}
    }}
    setupResizableDivider();
    poll();
    setInterval(poll, 500);
  </script>
</body>
</html>
"""


class LiveHTTPServer:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8765, store: Optional[LiveFrameStore] = None):
        self.host = str(host)
        self.port = int(port)
        self.store = store or LiveFrameStore()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> int:
        store = self.store
        index = _index_html()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format_, *args):
                return

            def do_GET(self):
                path = urlparse(self.path).path

                if path == "/":
                    body = index.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/status.json":
                    payload = store.get_status()
                    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/stream.mjpg":
                    boundary = "frame"
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                    self.end_headers()
                    last_ts = 0.0
                    try:
                        while True:
                            ts = store.get_updated_at()
                            if ts <= 0.0 or ts == last_ts:
                                time.sleep(0.05)
                                continue
                            last_ts = ts
                            jpeg = store.get_jpeg()
                            if jpeg is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                    except Exception:
                        return

                self.send_response(404)
                self.end_headers()

        httpd = None
        final_port = int(self.port)
        bind_errors = []
        for p in range(int(self.port), int(self.port) + 20):
            try:
                httpd = ThreadingHTTPServer((self.host, int(p)), Handler)
                final_port = int(p)
                break
            except OSError as e:
                bind_errors.append(f"{p}: {e}")
        if httpd is None:
            raise RuntimeError("live server 启动失败，端口绑定失败：" + " | ".join(bind_errors))

        self._httpd = httpd
        self.port = int(final_port)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread = t
        t.start()
        return self.port

    def urls(self) -> Dict[str, str]:
        bind_url = f"http://{self.host}:{self.port}/"
        if self.host == "0.0.0.0":
            open_url = f"http://127.0.0.1:{self.port}/"
            try:
                local_ip = socket.gethostbyname(socket.gethostname())
                remote_hint = f"http://{local_ip}:{self.port}/"
            except Exception:
                remote_hint = f"http://<服务器IP>:{self.port}/"
            return {"bind_url": bind_url, "open_url": open_url, "remote_hint": remote_hint}
        return {"bind_url": bind_url, "open_url": bind_url, "remote_hint": bind_url}

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            finally:
                self._httpd = None


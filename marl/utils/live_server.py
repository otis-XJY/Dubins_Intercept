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
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 16px;
      padding: 16px;
      height: 100vh;
      box-sizing: border-box;
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
      padding: 12px 12px 4px 12px;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 8px 10px;
      font-size: 13px;
      line-height: 1.4;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255,255,255,0.10);
    }}
    .k {{ color: rgba(230,230,230,0.75); }}
    .v {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New";
      white-space: pre-wrap;
      word-break: break-all;
      font-size: 11px;
      line-height: 1.35;
      max-height: 40vh;
      overflow: auto;
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
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card viewer">
      <img id="live" src="{stream_path}" alt="live stream"/>
    </div>
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
    async function poll() {{
      try {{
        const r = await fetch('/status.json', {{cache: 'no-store'}});
        if (!r.ok) return;
        const s = await r.json();
        const keys = Object.keys(s).sort();
        const kv = document.getElementById('kv');
        kv.innerHTML = '';
        for (const k of keys) {{
          const dk = document.createElement('div');
          dk.className = 'k';
          dk.textContent = k;
          const dv = document.createElement('div');
          dv.className = 'v';
          dv.textContent = String(s[k]);
          kv.appendChild(dk);
          kv.appendChild(dv);
        }}
      }} catch (e) {{}}
    }}
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


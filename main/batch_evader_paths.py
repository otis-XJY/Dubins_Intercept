import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

import matplotlib

# 服务器无图形界面时自动切换到 Agg，避免 Tk 报错
if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Draw.Draw_map import Draw_map
from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainIso, WH_main_obtainMapRef  # noqa: E402


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少 PyYAML 依赖。请先安装：pip install PyYAML（或按 requirements.txt 安装）。"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是 dict，实际为 {type(data)}")
    return data


def _time_tag(*, with_seconds: bool = True) -> str:
    fmt = "%m%d_%H%M%S" if with_seconds else "%m%d_%H%M"
    return datetime.now().strftime(fmt)


def _ensure_unique_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    for k in range(1, 10000):
        cand = f"{base_dir}_{k}"
        if not os.path.exists(cand):
            return cand
    raise RuntimeError(f"无法为目录分配唯一名称：{base_dir}")


def _fast_save(obj: Any, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.jbl")
    joblib.dump(obj, path, compress=3)
    return path


def _load_mid_points(points_file: str, expected_paths: int) -> List[np.ndarray]:
    with open(points_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mid_points_raw = data.get("mid_points", data) if isinstance(data, dict) else data
    if not isinstance(mid_points_raw, list):
        raise ValueError("点文件格式错误：mid_points 必须是列表")
    if len(mid_points_raw) != expected_paths:
        raise ValueError(f"点文件路径数量不匹配：期望 {expected_paths}，实际 {len(mid_points_raw)}")
    parsed = []
    for idx, path_pts in enumerate(mid_points_raw):
        if not path_pts:
            parsed.append(np.empty((0, 2), dtype=float))
            continue
        arr = np.asarray(path_pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"第 {idx + 1} 条路径点格式错误，必须是 [x, y] 列表")
        parsed.append(arr)
    return parsed


def _run_web_picker(
    *,
    background_png: str,
    points_file: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    ax_bbox: Tuple[float, float, float, float],
    S: np.ndarray,
    E: np.ndarray,
    host: str,
    port: int,
) -> None:
    state = {"saved": False}

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>路径网页选点</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    canvas {{ border: 1px solid #888; cursor: crosshair; max-width: 100%; height: auto; }}
    button {{ margin-right: 8px; margin-top: 8px; }}
    #status {{ margin-top: 12px; color: #333; }}
  </style>
</head>
<body>
  <h3>网页选点（路径中间点）</h3>
  <canvas id="canvas"></canvas>
  <div>
    <button id="prevBtn">上一条路径</button>
    <button id="nextBtn">下一条路径</button>
    <button id="undoBtn">撤销当前点</button>
    <button id="clearBtn">清空当前路径点</button>
    <button id="saveBtn">保存并结束</button>
  </div>
  <div id="status"></div>
  <script>
    const meta = {json.dumps({'xlim': list(xlim), 'ylim': list(ylim), 'ax_bbox': list(ax_bbox), 'S': S.tolist(), 'E': E.tolist()})};
    const numPaths = meta.S.length;
    const points = Array.from({{ length: numPaths }}, () => []);
    let current = 0;
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const img = new Image();
    img.src = '/background.png';

    function getAxesPixelRect() {{
      const [bx, by, bw, bh] = meta.ax_bbox;
      const left = bx * canvas.width;
      const right = (bx + bw) * canvas.width;
      const top = (1 - (by + bh)) * canvas.height;
      const bottom = (1 - by) * canvas.height;
      return {{ left, right, top, bottom }};
    }}

    function dataToPixel(x, y) {{
      const [xmin, xmax] = meta.xlim;
      const [ymin, ymax] = meta.ylim;
      const rect = getAxesPixelRect();
      const px = rect.left + (x - xmin) / (xmax - xmin) * (rect.right - rect.left);
      const py = rect.top + (ymax - y) / (ymax - ymin) * (rect.bottom - rect.top);
      return [px, py];
    }}

    function pixelToData(px, py) {{
      const [xmin, xmax] = meta.xlim;
      const [ymin, ymax] = meta.ylim;
      const rect = getAxesPixelRect();
      const x = xmin + ((px - rect.left) / (rect.right - rect.left)) * (xmax - xmin);
      const y = ymax - ((py - rect.top) / (rect.bottom - rect.top)) * (ymax - ymin);
      return [x, y];
    }}

    function insideAxes(px, py) {{
      const rect = getAxesPixelRect();
      return px >= rect.left && px <= rect.right && py >= rect.top && py <= rect.bottom;
    }}

    function drawPoint(x, y, color, r = 5) {{
      const [px, py] = dataToPixel(x, y);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      for (let i = 0; i < numPaths; i++) {{
        const s = meta.S[i];
        const e = meta.E[i];
        drawPoint(s[0], s[1], i === current ? '#00aa00' : '#66aa66', 6);
        drawPoint(e[0], e[1], i === current ? '#aa00aa' : '#aa66aa', 6);
      }}

      const arr = points[current];
      if (arr.length > 0) {{
        ctx.beginPath();
        const [sx, sy] = dataToPixel(meta.S[current][0], meta.S[current][1]);
        ctx.moveTo(sx, sy);
        for (const [x, y] of arr) {{
          const [px, py] = dataToPixel(x, y);
          ctx.lineTo(px, py);
        }}
        const [ex, ey] = dataToPixel(meta.E[current][0], meta.E[current][1]);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = '#0066ff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }}

      for (const [x, y] of arr) {{
        drawPoint(x, y, '#ff0000', 4);
      }}

      statusEl.textContent = `当前路径: ${{current + 1}}/${{numPaths}}，当前中间点数量: ${{arr.length}}`;
    }}

    img.onload = () => {{
      canvas.width = img.width;
      canvas.height = img.height;
      draw();
    }};

    canvas.addEventListener('click', (evt) => {{
      const rect = canvas.getBoundingClientRect();
      const px = (evt.clientX - rect.left) * (canvas.width / rect.width);
      const py = (evt.clientY - rect.top) * (canvas.height / rect.height);
      if (!insideAxes(px, py)) return;
      const [x, y] = pixelToData(px, py);
      points[current].push([x, y]);
      draw();
    }});

    document.getElementById('prevBtn').onclick = () => {{ current = Math.max(0, current - 1); draw(); }};
    document.getElementById('nextBtn').onclick = () => {{ current = Math.min(numPaths - 1, current + 1); draw(); }};
    document.getElementById('undoBtn').onclick = () => {{ points[current].pop(); draw(); }};
    document.getElementById('clearBtn').onclick = () => {{ points[current] = []; draw(); }};
    document.getElementById('saveBtn').onclick = async () => {{
      const resp = await fetch('/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ mid_points: points }})
      }});
      statusEl.textContent = resp.ok ? '已保存，服务器将自动结束。' : '保存失败，请查看终端日志。';
    }};
  </script>
</body>
</html>
"""

    class PickerHandler(BaseHTTPRequestHandler):
        def log_message(self, format_, *args):
            return

        def do_GET(self):
            if self.path == "/":
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/background.png":
                with open(background_png, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/save":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            with open(points_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            state["saved"] = True
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    httpd = None
    final_port = port
    bind_errors = []
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), PickerHandler)
            final_port = p
            break
        except OSError as e:
            bind_errors.append(f"{p}: {e}")
    if httpd is None:
        raise RuntimeError("网页选点服务启动失败，端口绑定失败：" + " | ".join(bind_errors))

    bind_url = f"http://{host}:{final_port}/"
    if final_port != port:
        print(f"[EPath] 端口 {port} 不可用，已自动切换到 {final_port}。")

    if host == "0.0.0.0":
        open_url = f"http://127.0.0.1:{final_port}/"
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            remote_hint = f"http://{local_ip}:{final_port}/"
        except Exception:
            remote_hint = f"http://<服务器IP>:{final_port}/"
        print(f"[EPath] 网页服务监听地址: {bind_url}")
        print(f"[EPath] 本机访问地址: {open_url}")
        print(f"[EPath] 远程访问地址(示例): {remote_hint}")
    else:
        open_url = bind_url
        print(f"[EPath] 网页选点已启动: {bind_url}")

    print('[EPath] 完成选点后请点击“保存并结束”。')
    try:
        webbrowser.open(open_url)
    except Exception:
        pass

    httpd.serve_forever()
    if not state["saved"]:
        raise RuntimeError("网页选点未完成，未检测到保存操作。")


def _interpolate_path(path_pts: np.ndarray, *, stepsize: float, theta0: float) -> np.ndarray:
    smooth_parts = []
    for j in range(path_pts.shape[0] - 1):
        p1 = path_pts[j, :]
        p2 = path_pts[j + 1, :]
        dist = float(np.linalg.norm(p2 - p1))
        n = int(max(2, np.ceil(dist / stepsize)))
        x = np.linspace(p1[0], p2[0], n)
        y = np.linspace(p1[1], p2[1], n)
        seg = np.column_stack((x, y))
        smooth_parts.append(seg if j == 0 else seg[1:])
    smooth = np.vstack(smooth_parts) if len(smooth_parts) > 0 else path_pts[:1, :]
    delta = np.diff(smooth, axis=0)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    theta_all = np.concatenate(([theta0], theta))
    return np.column_stack((smooth, theta_all))


def _list_time_maps(map_root: str, time_maps_cfg: Optional[List[str]]) -> List[str]:
    if time_maps_cfg is not None:
        return [str(x) for x in time_maps_cfg]
    if not os.path.isdir(map_root):
        return []
    out = []
    for name in sorted(os.listdir(map_root)):
        base = os.path.join(map_root, name)
        if os.path.isdir(base) and os.path.exists(os.path.join(base, "Map.jbl")):
            out.append(name)
    return out


def _worker_process_profile(args: Tuple) -> Dict[str, Any]:
    """单个 worker：处理一个 (TimeMap, profile) 对。在子进程中执行（仅 replay 模式）。"""
    (tm, save_dir, points_file, map_data, evader_arr, value_id, stepsize, profile_idx, save_isomap) = args
    try:
        from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainIso, WH_main_obtainMapRef

        S = evader_arr[:, 0:2]
        ValuePos = np.asarray(map_data["ValuePos"], dtype=float)
        value_id_arr = np.asarray(value_id, dtype=int)
        ValuePosOb = ValuePos[value_id_arr - 1, :]
        E = ValuePosOb[:, 0:2]
        v_E = float(map_data["v_E"])

        mid_points_by_path = _load_mid_points(points_file, expected_paths=S.shape[0])

        PathE2Val_true: Dict[int, np.ndarray] = {}
        for i in range(S.shape[0]):
            mid = mid_points_by_path[i]
            if mid.size > 0:
                pts = np.vstack([S[i, :], mid, E[i, :]])
            else:
                pts = np.vstack([S[i, :], E[i, :]])
            path_xyz = _interpolate_path(pts, stepsize=stepsize, theta0=float(evader_arr[i, 2]))
            PathE2Val_true[i] = path_xyz

        _fast_save(PathE2Val_true, save_dir, "PathE2Val_true")

        Map_copy = dict(map_data)
        Map_copy["Evader"] = evader_arr.copy()

        Map_copy, final_pathE2Val, vthetaAllE2Val = WH_main_obtainMapRef(Map_copy, evader_arr, ValuePos, 1)
        pathFinalE2ValIn, IsoMapE2ValIn_i_tt, _ = WH_main_obtainIso(Map_copy, v_E, final_pathE2Val, vthetaAllE2Val, 0)
        if save_isomap:
            _fast_save(IsoMapE2ValIn_i_tt, save_dir, "IsoMapE2ValIn_i_tt")
        _fast_save(pathFinalE2ValIn, save_dir, "pathFinalE2ValIn")

        return {"tm": tm, "profile": profile_idx, "save_dir": save_dir, "error": None}
    except Exception as e:
        return {"tm": tm, "profile": profile_idx, "save_dir": save_dir, "error": repr(e)}


def _randomize_evader_and_value_id(
    Evader: np.ndarray,
    value_id: np.ndarray,
    rng: np.random.Generator,
    vid_range: Optional[Tuple[int, int]],
    x_range_epos: Optional[Tuple[float, float]],
    y_range_epos: Optional[Tuple[float, float]],
    profile_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """对 Evader 位置和 value_id 进行随机化（返回新数组，不修改原数组）。"""
    evader_new = Evader.copy()
    vid_new = value_id.copy()

    if vid_range is not None:
        vid_min, vid_max = vid_range
        vid_new = rng.integers(vid_min, vid_max + 1, size=evader_new.shape[0])
        print(f"[EPath] profile={profile_idx} 随机化 value_id_list: {vid_new.tolist()}", flush=True)

    if x_range_epos is not None and y_range_epos is not None:
        x_min, x_max = x_range_epos
        y_min, y_max = y_range_epos
        evader_new[:, 0] = rng.uniform(x_min, x_max, size=evader_new.shape[0])
        evader_new[:, 1] = rng.uniform(y_min, y_max, size=evader_new.shape[0])
        print(f"[EPath] profile={profile_idx} 随机化 evader 位置:", flush=True)
        for i in range(evader_new.shape[0]):
            print(f"  evader[{i}]: x={evader_new[i, 0]:.1f}, y={evader_new[i, 1]:.1f}, theta={evader_new[i, 2]:.4f}", flush=True)

    return evader_new, vid_new


def main():
    parser = argparse.ArgumentParser(description="批量敌方无人机路径构建：生成 map/<TimeMap>/<TimeEpath>/*")
    parser.add_argument("--config", required=True, help="evader_paths.yaml 路径")
    parser.add_argument("--num-workers", type=int, default=None, help="并行 worker 数（覆盖 YAML 配置，仅 replay 模式）")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    map_root = str(cfg.get("map_root", "./map"))
    time_maps = _list_time_maps(map_root, cfg.get("time_maps"))
    if len(time_maps) == 0:
        raise RuntimeError("未发现任何 TimeMap（请先执行环境初始化/构建阶段）。")

    m_profiles = int(cfg.get("m_profiles", 1))
    with_seconds = bool(cfg.get("time_with_seconds", True))

    mode = str(cfg.get("mode", "interactive")).strip().lower()
    if mode not in ("interactive", "replay"):
        raise ValueError("evader_paths.mode 必须为 interactive 或 replay")

    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 1))
    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 1, len(time_maps) * m_profiles)

    # 启动信息（立即输出）
    if mode == "interactive":
        print(f"[EPath] mode={mode} time_maps={len(time_maps)} m_profiles={m_profiles} (串行执行，忽略 num_workers)", flush=True)
    else:
        print(f"[EPath] mode={mode} time_maps={len(time_maps)} m_profiles={m_profiles} num_workers={num_workers}", flush=True)

    web_cfg = cfg.get("web_picker", {})
    if not isinstance(web_cfg, dict):
        web_cfg = {}
    host = str(web_cfg.get("host", "127.0.0.1"))
    port = int(web_cfg.get("port", 8765))

    evader_cfg = cfg.get("evader", {})
    if not isinstance(evader_cfg, dict):
        evader_cfg = {}

    evader_arr = evader_cfg.get("init")
    if evader_arr is None:
        raise ValueError("evader.init 必须提供（Nx3：x,y,theta）")
    Evader = np.asarray(evader_arr, dtype=float)
    if Evader.ndim != 2 or Evader.shape[1] < 3:
        raise ValueError("evader.init 需为 Nx3（x,y,theta）")
    Evader = Evader[:, :3]

    value_id_list = cfg.get("value_id_list")
    if not (isinstance(value_id_list, list) and len(value_id_list) == Evader.shape[0]):
        raise ValueError("value_id_list 必须是列表，长度需等于 evader 数量（1-based id）")
    value_id = np.asarray([int(x) for x in value_id_list], dtype=int)

    # ---------- 随机化配置（仅读取配置，实际随机化在每个 profile 循环中执行） ----------
    rand_cfg = cfg.get("randomize", {})
    if not isinstance(rand_cfg, dict):
        rand_cfg = {}
    rand_enabled = bool(rand_cfg.get("enabled", False))
    rand_seed = rand_cfg.get("seed", None)
    rng = np.random.default_rng(rand_seed)

    # 预解析随机化范围配置（避免重复解析）
    vid_range = None
    if rand_enabled:
        rand_vid_cfg = rand_cfg.get("value_ids", {})
        if isinstance(rand_vid_cfg, dict) and bool(rand_vid_cfg.get("enabled", False)):
            vid_range_raw = rand_vid_cfg.get("range", [1, 3])
            if not (isinstance(vid_range_raw, list) and len(vid_range_raw) == 2):
                raise ValueError("randomize.value_ids.range 必须是 [min, max] 格式")
            vid_min, vid_max = int(vid_range_raw[0]), int(vid_range_raw[1])
            if vid_min < 1 or vid_max < vid_min:
                raise ValueError(f"randomize.value_ids.range 无效：{vid_range_raw}")
            vid_range = (vid_min, vid_max)

    x_range_epos = None
    y_range_epos = None
    if rand_enabled:
        rand_epos_cfg = rand_cfg.get("evader_pos", {})
        if isinstance(rand_epos_cfg, dict) and bool(rand_epos_cfg.get("enabled", False)):
            x_range_raw = rand_epos_cfg.get("x_range", [0, 2000])
            y_range_raw = rand_epos_cfg.get("y_range", [0, 2000])
            if not (isinstance(x_range_raw, list) and len(x_range_raw) == 2 and
                    isinstance(y_range_raw, list) and len(y_range_raw) == 2):
                raise ValueError("randomize.evader_pos.x_range/y_range 必须是 [min, max] 格式")
            x_min, x_max = float(x_range_raw[0]), float(x_range_raw[1])
            y_min, y_max = float(y_range_raw[0]), float(y_range_raw[1])
            if x_max < x_min or y_max < y_min:
                raise ValueError(f"randomize.evader_pos 范围无效：x={x_range_raw}, y={y_range_raw}")
            x_range_epos = (x_min, x_max)
            y_range_epos = (y_min, y_max)

    stepsize_override = cfg.get("Stepsize")
    points_file_template = str(cfg.get("points_file_template", "PathMidPts.json"))
    save_isomap = bool(cfg.get("save_isomap", False))

    # 设置日志
    os.makedirs(map_root, exist_ok=True)
    log_path = os.path.join(map_root, "_batch_epath.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("batch_evader_paths")

    ok = []
    failed = []

    if mode == "interactive":
        # interactive 模式：始终串行（需要人机交互）
        print(f"[EPath] interactive 模式，串行执行", flush=True)
        for tm in time_maps:
            tm_dir = os.path.join(map_root, tm)
            map_path = os.path.join(tm_dir, "Map.jbl")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"缺少 {map_path}")
            Map = joblib.load(map_path)
            if not isinstance(Map, dict):
                raise TypeError(f"Map.jbl 必须是 dict，实际为 {type(Map)}")

            Map["Evader"] = Evader.copy()

            obs = Map["obs"]
            sure = Map["sure"]
            obs_no_circle = Map["obs_no_circle"]
            obs_no_circle_in = Map["obs_no_circle_in"]
            Stepsize = float(Map["Stepsize"] if stepsize_override is None else stepsize_override)
            v_E = float(Map["v_E"])
            Trans_Point = Map["Trans_Point"]
            ValuePos = np.asarray(Map["ValuePos"], dtype=float)
            PStart_Point = Map["PStart_Point"]

            for _k in range(m_profiles):
                # 每个 profile 独立随机化
                if rand_enabled:
                    evader_k, vid_k = _randomize_evader_and_value_id(
                        Evader, value_id, rng, vid_range, x_range_epos, y_range_epos, _k
                    )
                else:
                    evader_k, vid_k = Evader, value_id

                Map["Evader"] = evader_k.copy()
                ValuePosOb = ValuePos[vid_k - 1, :]
                S = evader_k[:, 0:2]
                E = ValuePosOb[:, 0:2]

                time_epath = _time_tag(with_seconds=with_seconds)
                save_dir = _ensure_unique_dir(os.path.join(tm_dir, time_epath))
                time_epath = os.path.basename(save_dir)
                os.makedirs(save_dir, exist_ok=True)

                fig, ax = plt.subplots(figsize=(10, 8))
                Draw_map(PStart_Point, Trans_Point, ValuePosOb, obs, sure, obs_no_circle, obs_no_circle_in)
                ax.plot(evader_k[:, 0], evader_k[:, 1], "kd", linewidth=2, label="Evader")
                for i in range(S.shape[0]):
                    ax.plot(S[i, 0], S[i, 1], "gx", markersize=8, markeredgewidth=1.5)
                    ax.plot(E[i, 0], E[i, 1], "mx", markersize=8, markeredgewidth=1.5)
                plt.draw()

                points_file = os.path.join(save_dir, points_file_template)
                background_png = os.path.join(save_dir, "WebPickerBG.png")
                fig.savefig(background_png, dpi=200)
                fig.canvas.draw()
                ax_bbox = ax.get_position().bounds
                _run_web_picker(
                    background_png=background_png,
                    points_file=points_file,
                    xlim=ax.get_xlim(),
                    ylim=ax.get_ylim(),
                    ax_bbox=ax_bbox,
                    S=S,
                    E=E,
                    host=host,
                    port=port,
                )
                mid_points_by_path = _load_mid_points(points_file, expected_paths=S.shape[0])

                PathE2Val_true: Dict[int, np.ndarray] = {}
                for i in range(S.shape[0]):
                    mid = mid_points_by_path[i]
                    if mid.size > 0:
                        pts = np.vstack([S[i, :], mid, E[i, :]])
                    else:
                        pts = np.vstack([S[i, :], E[i, :]])
                    path_xyz = _interpolate_path(pts, stepsize=Stepsize, theta0=float(evader_k[i, 2]))
                    PathE2Val_true[i] = path_xyz
                    ax.plot(path_xyz[:, 0], path_xyz[:, 1], "b.-", linewidth=1.2)
                fig.savefig(os.path.join(save_dir, "PathPreview.png"), dpi=250)
                plt.close(fig)

                _fast_save(PathE2Val_true, save_dir, "PathE2Val_true")

                Map, final_pathE2Val, vthetaAllE2Val = WH_main_obtainMapRef(Map, evader_k, ValuePos, 1)
                pathFinalE2ValIn, IsoMapE2ValIn_i_tt, _ = WH_main_obtainIso(Map, v_E, final_pathE2Val, vthetaAllE2Val, 0)
                if save_isomap:
                    _fast_save(IsoMapE2ValIn_i_tt, save_dir, "IsoMapE2ValIn_i_tt")
                _fast_save(pathFinalE2ValIn, save_dir, "pathFinalE2ValIn")

                ok.append(f"{tm}/{time_epath}")
                logger.info(f"[EPath] ok TimeMap={tm} TimeEpath={time_epath} dir={save_dir}")
                print(f"[EPath] ok TimeMap={tm} TimeEpath={time_epath} dir={save_dir}", flush=True)
    else:
        # replay 模式：可以并行
        # 预加载所有 Map 并构建 work items
        all_work_items = []
        for tm in time_maps:
            tm_dir = os.path.join(map_root, tm)
            map_path = os.path.join(tm_dir, "Map.jbl")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"缺少 {map_path}")
            Map = joblib.load(map_path)
            if not isinstance(Map, dict):
                raise TypeError(f"Map.jbl 必须是 dict，实际为 {type(Map)}")
            Map["Evader"] = Evader.copy()
            Stepsize = float(Map["Stepsize"] if stepsize_override is None else stepsize_override)

            for k in range(m_profiles):
                # 每个 profile 独立随机化
                if rand_enabled:
                    evader_k, vid_k = _randomize_evader_and_value_id(
                        Evader, value_id, rng, vid_range, x_range_epos, y_range_epos, k
                    )
                else:
                    evader_k, vid_k = Evader, value_id

                time_epath = _time_tag(with_seconds=with_seconds)
                save_dir = _ensure_unique_dir(os.path.join(tm_dir, time_epath))
                time_epath = os.path.basename(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                points_file = os.path.join(save_dir, points_file_template)

                all_work_items.append((tm, save_dir, points_file, Map, evader_k.copy(), vid_k, Stepsize, k, save_isomap))

        if num_workers == 1:
            # 串行
            for item in all_work_items:
                result = _worker_process_profile(item)
                if result["error"] is None:
                    ok.append(f"{result['tm']}/profile{result['profile']}")
                    logger.info(f"[EPath] ok TimeMap={result['tm']} profile={result['profile']} dir={result['save_dir']}")
                    print(f"[EPath] ok TimeMap={result['tm']} profile={result['profile']}", flush=True)
                else:
                    failed.append(f"{result['tm']}/profile{result['profile']}")
                    logger.error(f"[EPath] FAIL TimeMap={result['tm']} profile={result['profile']} error={result['error']}")
                    print(f"[EPath] FAIL TimeMap={result['tm']} profile={result['profile']} error={result['error']}", flush=True)
        else:
            # 并行模式：使用 fork（Linux 默认），避免 spawn 重复导入
            ctx = get_context("fork")
            total = len(all_work_items)
            print(f"[EPath] 启动 {num_workers} 个 worker，共 {total} 个任务 ...", flush=True)
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                futures = {executor.submit(_worker_process_profile, item): item for item in all_work_items}
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    result = future.result()
                    if result["error"] is None:
                        ok.append(f"{result['tm']}/profile{result['profile']}")
                        logger.info(f"[EPath] ok TimeMap={result['tm']} profile={result['profile']} dir={result['save_dir']}")
                    else:
                        failed.append(f"{result['tm']}/profile{result['profile']}")
                        logger.error(f"[EPath] FAIL TimeMap={result['tm']} profile={result['profile']} error={result['error']}")
                    status = "ok" if result["error"] is None else "FAIL"
                    print(f"[EPath] [{done_count}/{total}] {status} TimeMap={result['tm']} profile={result['profile']}", flush=True)

    # 汇总
    logger.info(f"[EPath] done. ok={len(ok)}, failed={len(failed)}")
    print(f"\n[EPath] 完成。成功: {len(ok)}, 失败: {len(failed)}", flush=True)
    if failed:
        print(f"[EPath] 失败列表: {failed}", flush=True)
    print(f"[EPath] 详细日志: {log_path}", flush=True)


if __name__ == "__main__":
    main()

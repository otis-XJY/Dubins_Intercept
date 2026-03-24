import os
import sys
from datetime import datetime
import argparse
import json
import threading
import webbrowser
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import matplotlib

# 服务器无图形界面时自动切换到 Agg，避免 Tk 报错
if os.environ.get('DISPLAY', '') == '' and os.name != 'nt':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import joblib
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调用转换后的函数
from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainMapP,WH_main_obtainMapRef,WH_main_obtainIso
from Draw.Draw_map import Draw_map


def fast_save(obj, name_, save_dir):
    # 使用 .jbl 后缀区分普通 pickle
    filename = os.path.join(save_dir, f"{name_}.jbl")
    print(f"正在压缩保存 {filename} ...")
    joblib.dump(obj, filename, compress=3) 


def load_mid_points(points_file, expected_paths):
        with open(points_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        mid_points_raw = data.get('mid_points', data) if isinstance(data, dict) else data
        if not isinstance(mid_points_raw, list):
                raise ValueError('点文件格式错误：mid_points 必须是列表')
        if len(mid_points_raw) != expected_paths:
                raise ValueError(f'点文件路径数量不匹配：期望 {expected_paths}，实际 {len(mid_points_raw)}')

        parsed = []
        for idx, path_pts in enumerate(mid_points_raw):
                if not path_pts:
                        parsed.append(np.empty((0, 2)))
                        continue
                arr = np.asarray(path_pts, dtype=float)
                if arr.ndim != 2 or arr.shape[1] != 2:
                        raise ValueError(f'第 {idx + 1} 条路径点格式错误，必须是 [x, y] 列表')
                parsed.append(arr)
        return parsed


def run_web_picker(background_png, points_file, xlim, ylim, ax_bbox, S, E, host, port):
        state = {'saved': False}

        html = f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>路径网页选点</title>
    <style>
        body {{ font-family: sans-serif; margin: 16px; }}
        .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
        .panel {{ min-width: 320px; }}
        canvas {{ border: 1px solid #888; cursor: crosshair; max-width: 100%; height: auto; }}
        button {{ margin-right: 8px; margin-top: 8px; }}
        #status {{ margin-top: 12px; color: #333; }}
    </style>
</head>
<body>
    <h3>网页选点（路径中间点）</h3>
    <div class="row">
        <div class="panel">
            <canvas id="canvas"></canvas>
            <div>
                <button id="prevBtn">上一条路径</button>
                <button id="nextBtn">下一条路径</button>
                <button id="undoBtn">撤销当前点</button>
                <button id="clearBtn">清空当前路径点</button>
                <button id="saveBtn">保存并结束</button>
            </div>
            <div id="status"></div>
        </div>
    </div>
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
            if (!insideAxes(px, py)) {{
                return;
            }}
            const [x, y] = pixelToData(px, py);
            points[current].push([x, y]);
            draw();
        }});

        document.getElementById('prevBtn').onclick = () => {{
            current = Math.max(0, current - 1);
            draw();
        }};
        document.getElementById('nextBtn').onclick = () => {{
            current = Math.min(numPaths - 1, current + 1);
            draw();
        }};
        document.getElementById('undoBtn').onclick = () => {{
            points[current].pop();
            draw();
        }};
        document.getElementById('clearBtn').onclick = () => {{
            points[current] = [];
            draw();
        }};

        document.getElementById('saveBtn').onclick = async () => {{
            const resp = await fetch('/save', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ mid_points: points }})
            }});
            if (resp.ok) {{
                statusEl.textContent = '已保存，服务器将自动结束。';
            }} else {{
                statusEl.textContent = '保存失败，请查看终端日志。';
            }}
        }};
    </script>
</body>
</html>
"""

        class PickerHandler(BaseHTTPRequestHandler):
                def log_message(self, format_, *args):
                        return

                def do_GET(self):
                        if self.path == '/':
                                body = html.encode('utf-8')
                                self.send_response(200)
                                self.send_header('Content-Type', 'text/html; charset=utf-8')
                                self.send_header('Content-Length', str(len(body)))
                                self.end_headers()
                                self.wfile.write(body)
                                return

                        if self.path == '/background.png':
                                with open(background_png, 'rb') as f:
                                        body = f.read()
                                self.send_response(200)
                                self.send_header('Content-Type', 'image/png')
                                self.send_header('Content-Length', str(len(body)))
                                self.end_headers()
                                self.wfile.write(body)
                                return

                        self.send_response(404)
                        self.end_headers()

                def do_POST(self):
                        if self.path != '/save':
                                self.send_response(404)
                                self.end_headers()
                                return

                        length = int(self.headers.get('Content-Length', '0'))
                        body = self.rfile.read(length)
                        payload = json.loads(body.decode('utf-8'))

                        with open(points_file, 'w', encoding='utf-8') as f:
                                json.dump(payload, f, ensure_ascii=False, indent=2)

                        state['saved'] = True
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'OK')

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
                bind_errors.append(f'{p}: {e}')

        if httpd is None:
            raise RuntimeError('网页选点服务启动失败，端口绑定失败：' + ' | '.join(bind_errors))

        bind_url = f'http://{host}:{final_port}/'
        if final_port != port:
            print(f'端口 {port} 不可用，已自动切换到 {final_port}。')

        # 0.0.0.0 仅用于监听，不能直接在浏览器访问
        if host == '0.0.0.0':
            open_url = f'http://127.0.0.1:{final_port}/'
            try:
                local_ip = socket.gethostbyname(socket.gethostname())
                remote_hint = f'http://{local_ip}:{final_port}/'
            except Exception:
                remote_hint = f'http://<服务器IP>:{final_port}/'
            print(f'网页选点服务监听地址: {bind_url}')
            print(f'本机访问地址: {open_url}')
            print(f'远程访问地址(示例): {remote_hint}')
        else:
            open_url = bind_url
            print(f'网页选点已启动: {bind_url}')

        print('完成选点后请点击“保存并结束”。')

        try:
            webbrowser.open(open_url)
        except Exception:
                pass

        httpd.serve_forever()
        if not state['saved']:
                raise RuntimeError('网页选点未完成，未检测到保存操作。')


parser = argparse.ArgumentParser()
parser.add_argument('--web-pick',default=True, action='store_true', help='启用网页选点模式（浏览器中选择中间点）')
parser.add_argument('--host', default='127.0.0.1', help='网页选点服务监听地址，默认 127.0.0.1')
parser.add_argument('--port', type=int, default=8765, help='网页选点服务端口，默认 8765')
parser.add_argument('--points-file', default='', help='网页选点文件路径(JSON)，为空时自动生成')
args = parser.parse_args()

TimeMap='0320_0920'

time_tag = datetime.now().strftime('%m%d_%H%M')
save_dir = os.path.join('.', 'map', TimeMap, time_tag)
os.makedirs(save_dir, exist_ok=True)

web_pick_mode = args.web_pick

# 删除“直线路径”兜底：默认必须交互选点。
# 无图形后端且未启用网页选点时，直接提示用户切换到 --web-pick。
if (not web_pick_mode) and ('agg' in matplotlib.get_backend().lower()):
    backend_errors = []
    switched = False
    for candidate in ('TkAgg', 'QtAgg'):
        try:
            plt.switch_backend(candidate)
            print(f'检测到 Agg 后端，已自动切换为 {candidate} 以支持本地图形交互。')
            switched = True
            break
        except Exception as e:
            backend_errors.append(f'{candidate}: {e}')

    if not switched:
        error_msg = ' | '.join(backend_errors) if backend_errors else '未知原因'
        raise RuntimeError(
            '当前为非图形后端，且自动切换 TkAgg/QtAgg 失败，无法进行鼠标交互选点。'
            f'失败信息: {error_msg}。'
            '请安装 Tk 或 Qt 图形依赖，或使用 --web-pick 网页选点模式。'
        )

Map = joblib.load(f'./map/{TimeMap}/Map.jbl')	
# with open('Map'+TimeMap+'.pkl', 'rb') as f:
# 	Map = pickle.load(f)


# 1. 变量提取（假设 Map 是一个字典）
obs = Map['obs']
sure = Map['sure']
r = Map['r']
obs_no_circle = Map['obs_no_circle']
obs_no_circle_in = Map['obs_no_circle_in']
outline_all = Map['outline_all']
Stepsize = Map['Stepsize']
resolution = Map['resolution']
v_P = Map['v_P']
v_E = Map['v_E']
Trans_Point = Map['Trans_Point']
ValuePos=Map['ValuePos']
PStart_Point=Map['PStart_Point']

# 2. 定义 Evader 和 目标 id
Evader = np.array([
    [200, 1950, -np.pi/2],
    [1000, 1950, -np.pi/2],
    [1800, 1950, -np.pi/2]
])

# MATLAB id = [3, 2, 1]，对应 Python 索引为 [2, 1, 0]
id_list = np.array([3, 2, 1])
ValuePosOb = ValuePos[id_list - 1, :] # 注意：ValuePos 需要预先定义

# 3. 绘图准备
fig, ax = plt.subplots(figsize=(10, 8))
# 假设 Draw_map 已经转化为了 Python 函数
Draw_map(PStart_Point, Trans_Point, ValuePosOb, obs, sure, obs_no_circle, obs_no_circle_in)

ax.plot(Evader[:, 0], Evader[:, 1], 'kd', linewidth=2, label='Evader')

# 设置起点 S 和 终点 E
S = Evader[:, 0:2]
E = ValuePosOb[:, 0:2]

mid_points_by_path = None
if web_pick_mode:
    for i in range(S.shape[0]):
        ax.plot(S[i, 0], S[i, 1], 'gx', markersize=8, markeredgewidth=1.5)
        ax.plot(E[i, 0], E[i, 1], 'mx', markersize=8, markeredgewidth=1.5)
    plt.draw()

    background_png = os.path.join(save_dir, 'WebPickerBG.png')
    fig.savefig(background_png, dpi=200)

    points_file = args.points_file.strip() or os.path.join(save_dir, 'PathMidPts.json')
    fig.canvas.draw()
    ax_bbox = ax.get_position().bounds
    run_web_picker(
        background_png=background_png,
        points_file=points_file,
        xlim=ax.get_xlim(),
        ylim=ax.get_ylim(),
        ax_bbox=ax_bbox,
        S=S,
        E=E,
        host=args.host,
        port=args.port,
    )
    mid_points_by_path = load_mid_points(points_file, S.shape[0])
    print(f'已加载网页点文件: {points_file}')

PathE2Val_true = {} # 存储路径的字典

# 4. 路径选取与插值循环
for i in range(S.shape[0]):
    print(f'请为第 {i+1} 条路径选取中间点（左键选择，右键或回车结束）')
    
    # 绘制当前起终点
    ax.plot(S[i, 0], S[i, 1], 'gx', markersize=8, markeredgewidth=1.5)
    ax.plot(E[i, 0], E[i, 1], 'gx', markersize=8, markeredgewidth=1.5)
    plt.draw()

    # 鼠标点选中间点 (matplotlib 的 ginput)
    # n=-1 表示不限点数，直到按回车或右键
    # 注意：在某些 IDE（如 PyCharm）中 ginput 可能需要开启弹窗模式
    if web_pick_mode:
        MidPts = mid_points_by_path[i]
    else:
        pts = plt.ginput(n=-1, timeout=-1, show_clicks=True)
        MidPts = np.array(pts) if pts else np.empty((0, 2))

    # 构建完整路径点序列：S -> MidPts -> E
    if MidPts.size > 0:
        PathPts = np.vstack([S[i, :], MidPts, E[i, :]])
    else:
        PathPts = np.vstack([S[i, :], E[i, :]])

    SmoothPath_list = []

    # 5. 线性插值生成 SmoothPath
    for j in range(PathPts.shape[0] - 1):
        p1 = PathPts[j, :]
        p2 = PathPts[j+1, :]
        
        # 计算距离和需要的点数
        dist = np.linalg.norm(p2 - p1)
        N = int(max(2, np.ceil(dist / Stepsize)))
        
        # 生成插值点
        x = np.linspace(p1[0], p2[0], N)
        y = np.linspace(p1[1], p2[1], N)
        
        # 拼接点（避免重复添加分段连接处点）
        segment = np.column_stack((x, y))
        if j == 0:
            SmoothPath_list.append(segment)
        else:
            SmoothPath_list.append(segment[1:]) # 跳过首点，防止重复

    # 合并为最终矩阵
    SmoothPath = np.vstack(SmoothPath_list)

    # 6. 计算路径方向 (朝向角)
    delta = np.diff(SmoothPath, axis=0)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    
    # 保持数量一致，首位补上 Evader 的初始角度 (i, 2)
    # MATLAB: thetaAll = [Evader(i,3); theta]
    # 如果 SmoothPath 有 N 个点，diff 后 theta 有 N-1 个点，补一个后回到 N 个点
    thetaAll = np.concatenate(([Evader[i, 2]], theta))

    # 存储结果：[x, y, theta]
    # 注意 thetaAll 需要转置为列向量
    PathE2Val_true[i] = np.column_stack((SmoothPath, thetaAll))

    # 可视化生成的路径
    ax.plot(SmoothPath[:, 0], SmoothPath[:, 1], 'b.-', linewidth=1.5)
    plt.draw()

print("路径采集完成。")

# import pickle
# with open('PathE2Val_true'+TimeIso+'.pkl', 'wb') as f:
# 	pickle.dump(PathE2Val_true, f)

fast_save(PathE2Val_true, 'PathE2Val_true', save_dir)
print('路径采集完成')

# 5. 调用路径获取函数
# 计算从 Evader 起点到 ValuePos 的所有路径 (flagAll=1)
# 假设 ValuePos 已经在当前作用域中定义
Map, final_pathE2Val, vthetaAllE2Val = WH_main_obtainMapRef(
    Map, Evader, ValuePos, 1
)

# 6. 调用等时线生成函数
# 注意：最后一个参数 verse = 0
pathFinalE2ValIn, IsoMapE2ValIn_i_tt,_ = WH_main_obtainIso(
    Map, v_E, final_pathE2Val, vthetaAllE2Val, 0
)

# 打印或后续处理

# with open('IsoMapE2ValIn_i_tt'+TimeIso+'.pkl', 'wb') as f:
# 	pickle.dump(IsoMapE2ValIn_i_tt, f)

fast_save(IsoMapE2ValIn_i_tt, 'IsoMapE2ValIn_i_tt', save_dir)

# with open('pathFinalE2ValIn'+TimeIso+'.pkl', 'wb') as f:
# 	pickle.dump(pathFinalE2ValIn, f)
     
fast_save(pathFinalE2ValIn, 'pathFinalE2ValIn', save_dir)
print("IsoMapE2ValIn_i_tt 计算完成")
if web_pick_mode:
    fig.savefig(os.path.join(save_dir, "PathPreview.png"), dpi=300)
else:
    plt.show()
import os
# 解决 KMeans 在 Windows 上的内存泄漏警告
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString  # 确保导入了 LineString
from scipy.spatial import Voronoi, Delaunay
from sklearn.cluster import KMeans

# ==========================================
# 1. 环境定义
# ==========================================
map_dim = 2000
# 模拟多边形障碍物
obs_defs = [
    [(600, 1400), (900, 1450), (850, 1200), (650, 1150)],
    [(1200, 1000), (1500, 1100), (1450, 800), (1150, 750)],
    [(500, 500), (800, 600), (700, 350), (450, 400)],
    [(1000, 1500), (1200, 1600), (1300, 1400), (1000, 1300)]
]
polygons = [Polygon(p) for p in obs_defs]

def is_valid(p_arr):
    """检查点是否在地图内且不在障碍物内"""
    if not (0 <= p_arr[0] <= map_dim and 0 <= p_arr[1] <= map_dim):
        return False
    p = Point(p_arr)
    # buffer(20) 增加 20 米安全裕度
    return not any(poly.buffer(20).contains(p) for poly in polygons)

# ==========================================
# 2. 混合采样算法 (Hybrid Sampling)
# ==========================================
def hybrid_sampling(polygons, n_topological=15, n_blank_space=25):
    all_samples = []

    # --- A部分：拓扑采样 (利用 GVD 提取咽喉点) ---
    boundary_pts = []
    for poly in polygons:
        coords = list(poly.exterior.coords)
        for i in range(len(coords)-1):
            p1, p2 = np.array(coords[i]), np.array(coords[i+1])
            for t in np.linspace(0, 1, 6): 
                boundary_pts.append(p1*(1-t) + p2*t)
    
    # 加入地图边界点以稳定 Voronoi 骨架
    boundary_pts.extend([[0,0], [map_dim,0], [0,map_dim], [map_dim,map_dim], 
                         [map_dim/2, 0], [map_dim/2, map_dim], [0, map_dim/2], [map_dim, map_dim/2]])
    
    vor = Voronoi(boundary_pts)
    topo_candidates = [v for v in vor.vertices if is_valid(v)]
    
    if len(topo_candidates) > n_topological:
        kmeans = KMeans(n_clusters=n_topological, n_init=10)
        topo_samples = kmeans.fit(topo_candidates).cluster_centers_
    else:
        topo_samples = np.array(topo_candidates)
    
    # --- B部分：均匀空白区采样 (覆盖迂回路径) ---
    blank_samples = []
    attempts = 0
    while len(blank_samples) < n_blank_space and attempts < 2000:
        candidate = np.random.uniform(50, 1950, 2)
        if is_valid(candidate):
            # 确保与已有采样点保持距离，形成“蓝噪声”分布
            existing_pts = np.vstack([topo_samples, np.array(blank_samples)]) if blank_samples else topo_samples
            if np.min(np.linalg.norm(existing_pts - candidate, axis=1)) > 220:
                blank_samples.append(candidate)
        attempts += 1

    return topo_samples, np.array(blank_samples)

# ==========================================
# 3. 结果可视化与图构建
# ==========================================
topo_pts, blank_pts = hybrid_sampling(polygons)
combined_nodes = np.vstack([topo_pts, blank_pts])

plt.figure(figsize=(10, 10))

# 绘制障碍物
for poly in polygons:
    plt.fill(*poly.exterior.xy, color='gray', alpha=0.4, edgecolor='black', linewidth=1.5, zorder=2)

# 绘制“拦截图网络” (Interception Graph)
# 使用 Delaunay 剖分来连接采样点，形成潜在的拦截路径网
tri = Delaunay(combined_nodes)
edge_count = 0
for simplex in tri.simplices:
    for i, j in [(0,1), (1,2), (2,0)]:
        start, end = combined_nodes[simplex[i]], combined_nodes[simplex[j]]
        # 核心逻辑：只有不穿过障碍物的边才是有效的“编织线”
        line = LineString([start, end])
        if not any(poly.intersects(line) for poly in polygons):
            plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.15, zorder=1)
            edge_count += 1

# 绘制拓扑咽喉点
plt.scatter(topo_pts[:,0], topo_pts[:,1], c='red', s=120, marker='h', 
            edgecolors='white', linewidths=1, label='Topological Bottlenecks (Forced)', zorder=5)

# 绘制空白区采样点
plt.scatter(blank_pts[:,0], blank_pts[:,1], c='blue', s=80, marker='o', 
            edgecolors='white', linewidths=1, alpha=0.8, label='Open Space Nodes (Evasive)', zorder=4)

plt.title("Nature-Level Hybrid Sampling & Interception Graph", fontsize=14)
plt.legend(loc='upper right')
plt.xlim(0, 2000); plt.ylim(0, 2000)
plt.xlabel("X (meters)"); plt.ylabel("Y (meters)")
plt.grid(True, linestyle=':', alpha=0.5)
# plt.show()

print(f"采样完成！生成咽喉点: {len(topo_pts)}, 空白区点: {len(blank_pts)}, 有效拦截边: {edge_count}")



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString

# 假设已有采样点 (来自上一步)
# nodes = combined_nodes 
# polygons = polygons

def run_interception_game(nodes, polygons, enemy_start, enemy_target, uav_start):
    # 1. 构建图网络 (Graph Construction)
    G = nx.Graph()
    for i, p1 in enumerate(nodes):
        for j, p2 in enumerate(nodes):
            if i >= j: continue
            line = LineString([p1, p2])
            # 只有不穿过障碍物的边才有效
            if not any(poly.intersects(line) for poly in polygons):
                dist = np.linalg.norm(p1 - p2)
                G.add_edge(i, j, weight=dist)

    # 2. 找到最靠近起/终点的采样点索引
    def get_nearest_node(pt):
        return np.argmin(np.linalg.norm(nodes - pt, axis=1))

    e_start_idx = get_nearest_node(enemy_start)
    e_target_idx = get_nearest_node(enemy_target)
    u_start_idx = get_nearest_node(uav_start)

    # 3. 预测敌方“盲目最优路径” (Predicting Enemy's Blind Path)
    try:
        e_path_indices = nx.dijkstra_path(G, e_start_idx, e_target_idx, weight='weight')
        e_path_pts = nodes[e_path_indices]
    except nx.NetworkXNoPath:
        return "No Path"

    # 4. 寻找最优拦截点 (Interception Point Decision)
    # 计算敌方到达路径上每个点的时间 (假设速度 Ve=15, Vu=20)
    v_e, v_u = 15.0, 20.0
    best_int_idx = -1
    
    # 敌方到达每个点的累计时间
    e_times = [0]
    curr_t = 0
    for i in range(1, len(e_path_indices)):
        dist = G[e_path_indices[i-1]][e_path_indices[i]]['weight']
        curr_t += dist / v_e
        e_times.append(curr_t)

    # 我方到达这些点的时间
    for i, node_idx in enumerate(e_path_indices):
        u_dist = np.linalg.norm(nodes[node_idx] - uav_start)
        u_time = u_dist / v_u
        
        # 拦截条件：我方比敌方早到，且给末端捕获留出余裕 (如 5秒)
        if u_time + 5 < e_times[i]:
            best_int_idx = node_idx
            break # 找到第一个可拦截点即止 (拦截效率最高)

    return e_path_pts, nodes[best_int_idx] if best_int_idx != -1 else None, G

# --- 模拟执行 ---
enemy_s, enemy_t = np.array([100, 1900]), np.array([1900, 100])
uav_s = np.array([100, 100])
nodes = combined_nodes # 使用你图中生成的 combined_nodes

e_path, int_pt, graph = run_interception_game(nodes, polygons, enemy_s, enemy_t, uav_s)

# 可视化结果
plt.figure(figsize=(10, 10))
for poly in polygons: plt.fill(*poly.exterior.xy, color='gray', alpha=0.3)

# 画出整个网络
for u, v in graph.edges():
    plt.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], 'k-', alpha=0.05)

# 画出预测的敌方路径
plt.plot(e_path[:,0], e_path[:,1], 'r--', linewidth=2, label="Predicted Enemy Path")
plt.scatter(enemy_s[0], enemy_s[1], c='red', s=100, label="Enemy Start")
plt.scatter(enemy_t[0], enemy_t[1], c='gold', marker='*', s=200, label="Enemy Target")

# 画出我方拦截动作
plt.scatter(uav_s[0], uav_s[1], c='blue', s=100, label="UAV Start")
if int_pt is not None:
    plt.scatter(int_pt[0], int_pt[1], edgecolors='blue', facecolors='none', s=300, linewidths=3, label="Optimal Interception Point")
    plt.arrow(uav_s[0], uav_s[1], int_pt[0]-uav_s[0], int_pt[1]-uav_s[1], 
              color='blue', head_width=40, length_includes_head=True, alpha=0.6)

plt.title("Step 2: Predictive Interception on Topological Graph", fontsize=14)
plt.legend()
plt.show()
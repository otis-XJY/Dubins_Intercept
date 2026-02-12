import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import Voronoi, Delaunay
from sklearn.cluster import KMeans

# ==========================================
# 1. 环境与战略锚点定义 (用户提供的点)
# ==========================================
map_dim = 2000
# 示例战略点：ValuePos (我方保护目标), PPStart (拦截机起点)
ValuePos = np.array([[1800, 1800], [1800, 200]]) # 敌方可能攻击的目标
PPStart_Point = np.array([[100, 100], [100, 1000]]) # 我方起飞点

obs_defs = [
    [(600, 1400), (900, 1450), (850, 1200), (650, 1150)],
    [(1200, 1000), (1500, 1100), (1450, 800), (1150, 750)],
    [(500, 500), (800, 600), (700, 350), (450, 400)],
    [(1000, 1500), (1200, 1600), (1300, 1400), (1000, 1300)]
]
polygons = [Polygon(p) for p in obs_defs]

def is_valid(p_arr):
    if not (0 <= p_arr[0] <= map_dim and 0 <= p_arr[1] <= map_dim): return False
    return not any(poly.buffer(15).contains(Point(p_arr)) for poly in polygons)

# ==========================================
# 2. 改进的混合采样算法
# ==========================================
def multi_source_sampling(polygons, v_pos, pp_start, n_topo=12, n_blank=20):
    # --- A. 战略锚点 (直接加入) ---
    anchor_nodes = np.vstack([v_pos, pp_start])
    
    # --- B. 拓扑采样 (咽喉点) ---
    boundary_pts = []
    for poly in polygons:
        for i in range(len(poly.exterior.coords)-1):
            p1, p2 = np.array(poly.exterior.coords[i]), np.array(poly.exterior.coords[i+1])
            for t in np.linspace(0, 1, 5): boundary_pts.append(p1*(1-t) + p2*t)
    
    vor = Voronoi(boundary_pts + [[0,0], [2000,0], [0,2000], [2000,2000]])
    topo_candidates = [v for v in vor.vertices if is_valid(v)]
    topo_nodes = KMeans(n_clusters=n_topo, n_init=10).fit(topo_candidates).cluster_centers_

    # --- C. 空白覆盖采样 (剔除离锚点和拓扑点太近的点) ---
    blank_nodes = []
    fixed_nodes = np.vstack([anchor_nodes, topo_nodes])
    attempts = 0
    while len(blank_nodes) < n_blank and attempts < 2000:
        candidate = np.random.uniform(100, 1900, 2)
        if is_valid(candidate):
            # 保持 250m 的最小间距以防过度密集
            if np.min(np.linalg.norm(fixed_nodes - candidate, axis=1)) > 250:
                blank_nodes.append(candidate)
                fixed_nodes = np.vstack([fixed_nodes, candidate])
        attempts += 1

    return anchor_nodes, topo_nodes, np.array(blank_nodes)

# 执行采样
anchors, topo, blank = multi_source_sampling(polygons, ValuePos, PPStart_Point)
all_nodes = np.vstack([anchors, topo, blank])

# ==========================================
# 3. 构建立体拦截图 (Graph Construction)
# ==========================================
plt.figure(figsize=(10, 10))
for poly in polygons: plt.fill(*poly.exterior.xy, color='gray', alpha=0.3, edgecolor='black')

# 连接图并可视化
tri = Delaunay(all_nodes)
for simplex in tri.simplices:
    for i, j in [(0,1), (1,2), (2,0)]:
        p1, p2 = all_nodes[simplex[i]], all_nodes[simplex[j]]
        if not any(poly.intersects(LineString([p1, p2])) for poly in polygons):
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.1, zorder=1)

# 不同颜色标注不同属性的点
plt.scatter(topo[:,0], topo[:,1], c='red', s=100, marker='h', label='Topological Bottlenecks')
plt.scatter(blank[:,0], blank[:,1], c='blue', s=60, alpha=0.6, label='Coverage Nodes')
plt.scatter(anchors[:,0], anchors[:,1], c='gold', s=200, marker='*', edgecolors='black', label='Strategic Anchors (Value/Start)')

plt.title("Integrated Manifold Sampling: Topo + Coverage + Strategic Anchors")
plt.legend(loc='upper right')
plt.show()
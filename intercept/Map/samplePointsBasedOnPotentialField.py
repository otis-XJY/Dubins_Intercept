import os
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

# 屏蔽 KMeans 在 Windows 上的内存泄露警告
os.environ["OMP_NUM_THREADS"] = "1"

def samplePointsBasedOnPotentialField(
        obs_mapsize_x, 
        obs_mapsize_y, 
        obstacle_list, 
        value_positions, 
        start_positions,    
        n_topo = 15,     # 咽喉点数量
        n_blank = 25,    # 空白区覆盖点数量
        safety = 25.0,   # 避障安全距离
        min_node_spacing = None # 自动计算节点间距):
        ):
    """
    Nature/Science 级采样算法：返回全域拦截采样点坐标
    
    返回:
    - all_coords: np.array (N x 2), 所有采样点的坐标总表
    - info: dict, 包含分类好的坐标，便于你在拦截逻辑中区分优先级
    """
    min_x, max_x = min(obs_mapsize_x), max(obs_mapsize_x)
    min_y, max_y = min(obs_mapsize_y), max(obs_mapsize_y)
    width, height = max_x - min_x, max_y - min_y
    
    # 参数设置
    # n_topo = 15     # 咽喉点数量
    # n_blank = 25    # 空白区覆盖点数量
    # safety = 25.0   # 避障安全距离
    # spacing = np.sqrt(width**2 + height**2) / 12.0 # 自动计算节点间距

        # 自动计算最小间距：以地图对角线的比例作为基准，确保采样均匀性
    if min_node_spacing is None:
        min_node_spacing = np.sqrt(width**2 + height**2) / 12.0

    polygons = [Polygon(p) for p in obstacle_list] if obstacle_list else []

    def is_valid(pt):
        if not (min_x <= pt[0] <= max_x and min_y <= pt[1] <= max_y): return False
        p_geo = Point(pt)
        return not any(poly.buffer(safety).contains(p_geo) for poly in polygons)

    # 1. 战略锚点坐标 (Strategic Anchors)
    anchors = []
    if value_positions is not None: anchors.extend(value_positions)
    if start_positions is not None: anchors.extend(start_positions)
    anchors = np.array(anchors) if len(anchors) > 0 else np.empty((0, 2))

    # 2. 拓扑咽喉坐标 (Topological Bottlenecks)
    seeds = []
    for poly in polygons:
        coords = list(poly.exterior.coords)
        for i in range(len(coords)-1):
            p1, p2 = np.array(coords[i]), np.array(coords[i+1])
            for t in np.linspace(0, 1, 6): seeds.append(p1*(1-t) + p2*t)
    # 加入边界点稳定骨架
    seeds.extend([[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y], [min_x+width/2, min_y+height/2]])
    
    vor = Voronoi(seeds)
    topo_candidates = [v for v in vor.vertices if is_valid(v)]
    if len(topo_candidates) >= n_topo:
        topo_nodes = KMeans(n_clusters=n_topo, n_init=10).fit(topo_candidates).cluster_centers_
    else:
        topo_nodes = np.array(topo_candidates)

    # 3. 全域覆盖坐标 (Open Space Nodes)
    blank_nodes = []
    existing = topo_nodes if len(anchors) == 0 else np.vstack([topo_nodes, anchors])
    for _ in range(3000):
        if len(blank_nodes) >= n_blank: break
        cand = np.array([np.random.uniform(min_x+safety, max_x+safety), np.random.uniform(min_y+safety, max_y+safety)])
        if is_valid(cand):
            if len(existing) == 0 or np.min(np.linalg.norm(existing - cand, axis=1)) > min_node_spacing:
                blank_nodes.append(cand)
                existing = np.vstack([existing, cand])
    blank_nodes = np.array(blank_nodes) if len(blank_nodes) > 0 else np.empty((0, 2))

    # 4. 汇总所有坐标
    all_coords = np.vstack([n for n in [anchors, topo_nodes, blank_nodes] if len(n) > 0])
    sampled_points = np.vstack([n for n in [topo_nodes, blank_nodes] if len(n) > 0])
    
    return sampled_points, {
        "all_nodes": all_coords,
        "anchors": anchors,
        "bottlenecks": topo_nodes,
        "coverage": blank_nodes
    }

# ==========================================
# 示例调用 (你可以直接在你的主程序里写):
# ==========================================
# my_x = [0, 2000]
# my_y = [0, 2000]
# # 得到最终的坐标矩阵 coords
# coords, detail_info = get_interception_samples(my_x, my_y, obstacle_list, ValuePos, PPStart_Point)
# print(coords) # 这是一个 N x 2 的数组，包含所有你需要的采样点坐标
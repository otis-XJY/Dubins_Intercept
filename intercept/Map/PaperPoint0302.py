"""
局部搜索优化的采样点和覆盖率优化模块

功能：
1. 多源分层采样 - 生成包含拓扑、覆盖和策略锚点的初始采样点
2. 基于局部搜索的可见性微调 - 通过位置和朝向微调实现覆盖优化
3. 可视化比较 - 对比优化前后的效果
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi, Delaunay
from sklearn.cluster import KMeans
from matplotlib.patches import Polygon as MplPolygon, Wedge
import time

# ==============================================================================
# 阶段一：分层流形采样 (维持您的优秀原始逻辑不变)
# ==============================================================================

def multi_source_sampling(polygons, v_pos, pp_start, map_dim, n_topo=15, n_blank=25):
    def is_valid(p_arr, safety_margin=15):
        if not (0 <= p_arr[0] <= map_dim and 0 <= p_arr[1] <= map_dim): return False
        return not any(poly.buffer(safety_margin).contains(Point(p_arr)) for poly in polygons)

    anchor_nodes = np.vstack([v_pos, pp_start])
    
    boundary_pts =[]
    for poly in polygons:
        for i in range(len(poly.exterior.coords)-1):
            p1, p2 = np.array(poly.exterior.coords[i]), np.array(poly.exterior.coords[i+1])
            for t in np.linspace(0, 1, 5): boundary_pts.append(p1*(1-t) + p2*t)
    
    vor = Voronoi(boundary_pts + [[0,0],[map_dim,0], [0,map_dim],[map_dim,map_dim]])
    topo_candidates =[v for v in vor.vertices if is_valid(v)]
    if len(topo_candidates) < n_topo:
        topo_nodes = np.array(topo_candidates)
    else:
        topo_nodes = KMeans(n_clusters=n_topo, n_init=10).fit(topo_candidates).cluster_centers_

    blank_nodes =[]
    fixed_nodes = np.vstack([anchor_nodes, topo_nodes]) if topo_nodes.size > 0 else anchor_nodes
    attempts = 0
    while len(blank_nodes) < n_blank and attempts < 3000:
        candidate = np.random.uniform(50, map_dim-50, 2)
        if is_valid(candidate):
            if np.min(np.linalg.norm(fixed_nodes - candidate, axis=1)) > 250:
                blank_nodes.append(candidate)
                fixed_nodes = np.vstack([fixed_nodes, candidate])
        attempts += 1

    return anchor_nodes, topo_nodes, np.array(blank_nodes) if blank_nodes else np.empty((0,2))

# ==============================================================================
# 阶段二：基于局部搜索的可见性微调优化器 (修复边界截断问题)
# ==============================================================================

class CoverageOptimizer:
    def __init__(self, bounds, obstacles, camera_depth, camera_fov_deg):
        self.bounds = bounds
        self.obstacles_poly = obstacles
        self.camera_depth = camera_depth
        self.camera_fov = camera_fov_deg
        self.obstacle_union = unary_union(self.obstacles_poly)
        
        # 核心修复 1: 显式定义地图的物理边界多边形
        self.map_boundary_poly = Polygon([
            (bounds[0], bounds[2]),
            (bounds[1], bounds[2]),
            (bounds[1], bounds[3]),
            (bounds[0], bounds[3])
        ])

    def _create_wedge(self, position, angle_deg):
        """生成被地图边界严格裁剪后的相机视场多边形"""
        x, y = position
        fov_rad = np.deg2rad(self.camera_fov)
        angle_rad = np.deg2rad(angle_deg)
        
        num_segments = 20
        angles = np.linspace(angle_rad - fov_rad / 2, angle_rad + fov_rad / 2, num_segments)
        points = [(x, y)]
        points.extend([(x + self.camera_depth * np.cos(a), y + self.camera_depth * np.sin(a)) for a in angles])
        
        raw_wedge = Polygon(points)
        # 核心修复 2: 任何超出地图范围的视场都会被切除
        return raw_wedge.intersection(self.map_boundary_poly)

    def fine_tune(self, initial_points, max_iterations=8, safety_margin=15.0):
        positions = initial_points.copy()
        n_samplers = len(positions)
        
        orientations = np.full(n_samplers, 90.0) 
        
        step = 30.0 
        pos_offsets = np.array([
            [0, 0], [step, 0],[-step, 0],[0, step], [0, -step],
            [step, step],[-step, -step],[step, -step], [-step, step]
        ])
        
        print("\n--- 开始对初始采样点进行稳健的局部精细优化 ---")
        for i in range(max_iterations):
            start_time = time.time()
            max_movement = 0.0

            for k in range(n_samplers):
                best_score = -float('inf')
                best_angle = orientations[k]
                best_pos = positions[k]
                
                current_wedges = [self._create_wedge(positions[j], orientations[j]) for j in range(n_samplers) if j != k]
                other_coverage = unary_union(current_wedges)
                
                for offset in pos_offsets:
                    cand_pos = positions[k] + offset
                    
                    if not (0 <= cand_pos[0] <= self.bounds[1] and 0 <= cand_pos[1] <= self.bounds[3]):
                        continue
                    if self.obstacle_union.buffer(safety_margin).contains(Point(cand_pos)):
                        continue
                        
                    for angle in np.arange(0, 360, 15): 
                        test_wedge = self._create_wedge(cand_pos, angle)
                        
                        # 此时 test_wedge 已经被限制在地图内了。
                        marginal_gain = test_wedge.difference(other_coverage).difference(self.obstacle_union).area
                        overlap_area = test_wedge.intersection(other_coverage).area
                        
                        score = marginal_gain - 1.0 * overlap_area
                        
                        if score > best_score:
                            best_score = score
                            best_angle = angle
                            best_pos = cand_pos
                
                dist_moved = np.linalg.norm(best_pos - positions[k])
                if dist_moved > max_movement:
                    max_movement = dist_moved

                positions[k] = best_pos
                orientations[k] = best_angle
            
            final_wedges =[self._create_wedge(p, o) for p, o in zip(positions, orientations)]
            total_coverage_poly = unary_union(final_wedges).difference(self.obstacle_union)
            print(f"迭代 {i+1}/{max_iterations} | 有效覆盖: {total_coverage_poly.area:.0f} | 最大微调位移: {max_movement:.1f}m | 耗时: {time.time()-start_time:.2f}s")
            
            if max_movement < 1.0:
                print("--- 优化已稳定收敛 ---")
                break

        self.final_positions = positions
        self.final_orientations = orientations
        self.total_coverage_poly = total_coverage_poly
        return self.final_positions, self.final_orientations

    def visualize_comparison(self, initial_points):
        fig, ax = plt.subplots(figsize=(14, 14))

        if isinstance(self.total_coverage_poly, Polygon):
            ax.add_patch(MplPolygon(np.array(self.total_coverage_poly.exterior.coords), facecolor='lightgreen', alpha=0.3, zorder=1))
        elif isinstance(self.total_coverage_poly, MultiPolygon):
            for poly in self.total_coverage_poly.geoms:
                 ax.add_patch(MplPolygon(np.array(poly.exterior.coords), facecolor='lightgreen', alpha=0.3, zorder=1))

        for poly_shapely in self.obstacles_poly:
            ax.add_patch(MplPolygon(np.array(poly_shapely.exterior.coords), facecolor='dimgray', edgecolor='black', alpha=0.8, zorder=10))

        ax.scatter(initial_points[:,0], initial_points[:,1], c='red', s=40, marker='x', label='Initial Strategic Points (Red X)', zorder=14)

        for pos, orient in zip(self.final_positions, self.final_orientations):
            # 绘制真实的视场（如果有截断，会画出被截断的形状以示真实覆盖）
            real_wedge = self._create_wedge(pos, orient)
            if isinstance(real_wedge, Polygon):
                ax.add_patch(MplPolygon(np.array(real_wedge.exterior.coords), facecolor='deepskyblue', alpha=0.3, edgecolor='darkblue', zorder=12))
            
            ax.plot(pos[0], pos[1], 'o', color='navy', markersize=6, markeredgecolor='white', zorder=15)
        
        for p_initial, p_final in zip(initial_points, self.final_positions):
            ax.plot([p_initial[0], p_final[0]], [p_initial[1], p_final[1]], 'r--', alpha=0.6, linewidth=1.5)

        # 绘制地图边界线，强调其约束作用
        ax.plot([0, 2000, 2000, 0, 0],[0, 0, 2000, 2000, 0], 'k-', linewidth=2, zorder=20)

        ax.set_xlim(-100, 2100) # 稍微留点白边，以便看清边界行为
        ax.set_ylim(-100, 2100)
        ax.set_aspect('equal')
        ax.set_title("Robust Visibility Optimization (Map Boundary Constrained)", fontsize=16)
        ax.legend()
        plt.grid(True, linestyle=':', alpha=0.4)
        
# ==============================================================================
# 其他集成函数保持不变
# ==============================================================================

def samplePointsBasedOnPotentialField(
        obs_mapsize_x,
        obs_mapsize_y,
        obstacle_list,
        value_positions,
        start_positions,
        n_topo=15,
        n_blank=25,
        safety=15.0,
        min_node_spacing=None,
        camera_depth=350,
        camera_fov_deg=95,
        max_iterations=8,
        visualize=True,
):
    min_x, max_x = min(obs_mapsize_x), max(obs_mapsize_x)
    min_y, max_y = min(obs_mapsize_y), max(obs_mapsize_y)
    map_dim = max(max_x - min_x, max_y - min_y)

    polygons = [Polygon(p) for p in obstacle_list] if obstacle_list else[]

    value_positions = np.array(value_positions) if value_positions is not None else np.empty((0, 2))
    start_positions = np.array(start_positions) if start_positions is not None else np.empty((0, 2))

    anchors, bottlenecks, coverage = multi_source_sampling(
        polygons=polygons,
        v_pos=value_positions,
        pp_start=start_positions,
        map_dim=map_dim,
        n_topo=n_topo,
        n_blank=n_blank
    )

    initial_all_nodes = np.vstack([n for n in [anchors, bottlenecks, coverage] if len(n) > 0])

    optimizer = CoverageOptimizer(
        bounds=[0, map_dim, 0, map_dim],
        obstacles=polygons,
        camera_depth=camera_depth,
        camera_fov_deg=camera_fov_deg
    )

    final_positions, final_orientations = optimizer.fine_tune(
        initial_points=initial_all_nodes,
        max_iterations=max_iterations,
        safety_margin=safety
    )

    n_anchor = len(anchors)
    n_topo_actual = len(bottlenecks)
    sampled_points = final_positions[n_anchor:n_anchor + n_topo_actual + len(coverage)]

    setAll = {
        "all_nodes": final_positions,
        "anchors": final_positions[:n_anchor],
        "bottlenecks": final_positions[n_anchor:n_anchor + n_topo_actual],
        "coverage": final_positions[n_anchor + n_topo_actual:],
        "orientations": final_orientations,
        "total_coverage_poly": optimizer.total_coverage_poly,
    }

    if visualize:
        optimizer.visualize_comparison(initial_all_nodes)

    return sampled_points, setAll

def optimizeSamplingPoints(SamplingConfig):
    map_dim = SamplingConfig.get('map_dim', 2000)
    value_pos = SamplingConfig.get('ValuePos')
    start_pos = SamplingConfig.get('PPStart_Point')
    obstacles = SamplingConfig.get('obstacles',[])
    camera_depth = SamplingConfig.get('camera_depth', 350)
    camera_fov_deg = SamplingConfig.get('camera_fov_deg', 80)
    n_topo = SamplingConfig.get('n_topo', 15)
    n_blank = SamplingConfig.get('n_blank', 25)
    max_iterations = SamplingConfig.get('max_iterations', 8)
    safety_margin = SamplingConfig.get('safety_margin', 15.0)
    visualize = SamplingConfig.get('visualize', True)

    start_time = time.time()

    obstacle_list =[list(poly.exterior.coords)[:-1] for poly in obstacles]
    sampled_points, setAll = samplePointsBasedOnPotentialField(
        obs_mapsize_x=[0, map_dim],
        obs_mapsize_y=[0, map_dim],
        obstacle_list=obstacle_list,
        value_positions=value_pos,
        start_positions=start_pos,
        n_topo=n_topo,
        n_blank=n_blank,
        safety=safety_margin,
        min_node_spacing=None,
        camera_depth=camera_depth,
        camera_fov_deg=camera_fov_deg,
        max_iterations=max_iterations,
        visualize=visualize,
    )

    total_coverage_area = setAll['total_coverage_poly'].area
    map_area = map_dim * map_dim
    coverage_ratio = 100.0 * total_coverage_area / (map_area + 1e-9)

    end_time = time.time()
    elapsed_time = end_time - start_time

    SamplingConfig['initial_all_nodes'] = np.vstack([
        n for n in [setAll['anchors'], setAll['bottlenecks'], setAll['coverage']] if len(n) > 0
    ])
    SamplingConfig['final_positions'] = setAll['all_nodes']
    SamplingConfig['final_orientations'] = setAll['orientations']
    SamplingConfig['total_coverage_poly'] = setAll['total_coverage_poly']
    SamplingConfig['coverage_area'] = total_coverage_area
    SamplingConfig['coverage_ratio'] = coverage_ratio
    SamplingConfig['sampled_points'] = sampled_points
    SamplingConfig['setAll'] = setAll
    SamplingConfig['elapsed_time'] = elapsed_time

    print(f"\n✓ 采样点优化完成，总耗时: {elapsed_time:.2f} 秒")
    return SamplingConfig

if __name__ == '__main__':
    map_dim = 2000
    ValuePos = np.array([[1800, 1800],[1800, 200]])
    PPStart_Point = np.array([[100, 100], [100, 1000]])
    
    obs_defs = [[(600, 1400), (900, 1450), (850, 1200), (650, 1150)],[(1200, 1000), (1500, 1100), (1450, 800), (1150, 750)],[(500, 500), (800, 600), (700, 350), (450, 400)],[(1000, 1500), (1200, 1600), (1300, 1400), (1000, 1300)]
    ]
    polygons = [Polygon(p) for p in obs_defs]
    
    CAMERA_DEPTH = 350
    CAMERA_FOV = 80
    
    SamplingConfig = {
        'map_dim': map_dim,
        'ValuePos': ValuePos,
        'PPStart_Point': PPStart_Point,
        'obstacles': polygons,
        'camera_depth': CAMERA_DEPTH,
        'camera_fov_deg': CAMERA_FOV,
        'n_topo': 15,
        'n_blank': 25,
        'max_iterations': 6,
        'safety_margin': 15.0,
        'visualize': True
    }
    
    result = optimizeSamplingPoints(SamplingConfig)
    
    print("\n" + "="*80)
    print("优化结果摘要")
    print("="*80)
    print(f"初始采样点数: {len(result['initial_all_nodes'])}")
    print(f"最终采样点数: {len(result['final_positions'])}")
    print(f"覆盖面积: {result['coverage_area']:.0f}")
    print(f"覆盖率: {result['coverage_ratio']:.2f}%")
    print(f"总耗时: {result['elapsed_time']:.2f} 秒")

    plt.show()
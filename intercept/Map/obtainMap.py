import numpy as np
import matplotlib.pyplot as plt
from Draw.Draw_map import Draw_map
from intercept.Map.obtain_obs_no_circle_all import obtain_obs_no_circle_all
from intercept.Map.computePotentialField import computePotentialField
# from Map.samplePointsBasedOnPotentialField import samplePointsBasedOnPotentialField
from intercept.Map.PaperPoint0302 import samplePointsBasedOnPotentialField
from intercept.Map.obtain_finalObs import obtain_finalObs
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import Voronoi, Delaunay
from sklearn.cluster import KMeans
def obtainMap(Map, n_topo=10, n_blank=100, safety=20.0):
    """
    生成地图环境，包括障碍物、势场和采样点

    参数:
    Map: 包含地图参数的字典

    返回:
    Map: 更新后的地图字典
    """

    # 从输入Map中提取参数
    r = Map['r']  # 最小转弯半径
    Stepsize = Map['Stepsize']  # 路径生成间隔
    sure = Map['sure']  # 安全距离
    obs_mapsize_x = Map['obs_mapsize_x']  # 障碍物区域x尺寸
    obs_mapsize_y = Map['obs_mapsize_y']  # 障碍物区域y尺寸
    obs_mapsize = np.array([obs_mapsize_x, obs_mapsize_y])

    num = 0  # 圆形随机障碍物数量
    R = Map['R']  # 障碍物半径范围
    num_obs_nocircle = Map['num_obs_nocircle']  # 多边形随机障碍物数量
    num_steps = Map['num_steps']  # 边界采样点数

    ValuePos = Map['ValuePos']  # 价值点位置
    PStart_Point = Map['PStart_Point']  # 起点位置
    # E_TranPoint = Map['E_TranPoint']  # 终点位置
    Trans_Point = Map['Trans_Point']  # 过渡点位置
    resolution_map_pos = Map['resolution_map_pos']  # 地点生成分辨率

    # 计算总位置数
    numbers = np.arange(1, resolution_map_pos[0] * resolution_map_pos[1])
    result = np.sum(numbers)

    # 计算地图尺寸
    mapsize = np.array([
    max(np.max(Trans_Point[:, 0]), obs_mapsize_x[1]),
    max(np.max(Trans_Point[:, 1]), obs_mapsize_y[1])
])

    resolution = np.array([max(mapsize) / 500, max(mapsize)])

    # 生成障碍物
    start_time = time.time()
    outline_all, obs_no_circle, obs_no_circleTP, obs_no_circle_in = obtain_obs_no_circle_all(
        num_obs_nocircle, obs_mapsize, R, num_steps, PStart_Point, ValuePos, sure, Map['r']
    )

    # 绘制初始地图
    plt.figure(figsize=(10, 8))
    Draw_map(PStart_Point, Trans_Point, ValuePos, [], sure, obs_no_circle, obs_no_circle_in)

    # 处理障碍物和终点
    removeEndPointAll = []
    PPStart_Point = np.vstack([PStart_Point, ValuePos])

    for i in range(len(PPStart_Point)):
        for j in range(len(Trans_Point)):
            obs, obs_no_circle, obs_no_circleTP, obs_no_circle_in, outline_all, removeEndPoint = obtain_finalObs(
                num, obs_mapsize, PPStart_Point[i], Trans_Point[j], R, r,
                obs_no_circle, obs_no_circleTP, obs_no_circle_in, outline_all
            )
            if i == 0:
                removeEndPointAll.append(removeEndPoint)

    # 移除无效终点
    removeEndPointAll = np.array(removeEndPointAll)
    Trans_Point = Trans_Point[removeEndPointAll == 0]

    # 绘制处理后的地图
    plt.figure(figsize=(10, 8))
    Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)

    # 计算势场
    # X_full, Y_full, PotentialField, X_grid, Y_grid, PotentialField_ds = computePotentialField(
    #     obs_no_circleTP, E_TranPoint, PPStart_Point, Trans_Point, obs_mapsize, 10, 30 * r
    # )

    # 混合采样点
    num_samples = 36
    sampled_points, setAll = samplePointsBasedOnPotentialField(
    obs_mapsize_x, 
    obs_mapsize_y,
    obstacle_list=obs_no_circleTP, 
    value_positions=ValuePos[:,:2], 
    start_positions=PStart_Point[:,:2], 
    n_topo=n_topo, 
    n_blank=n_blank, 
    safety=safety,
    min_node_spacing=None
)

    # # 绘制势场和采样点
    # plt.figure(figsize=(12, 10))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X_grid, Y_grid, PotentialField_ds, cmap='viridis', alpha=0.8)
    # ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_potential_values,
    #            c='red', s=50, marker='o')
    # plt.colorbar(ax.collections[0], ax=ax, shrink=0.5, aspect=5)
    # plt.title('Downsampled Potential Field with Sampled Points')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # ==========================================
    # 3. 构建立体拦截图 (Graph Construction)
    # ==========================================
    plt.figure(figsize=(10, 8))
    Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)


    # 连接图并可视化
    tri = Delaunay(setAll['all_nodes'])
    for simplex in tri.simplices:
        for i, j in [(0,1), (1,2), (2,0)]:
            p1, p2 = setAll['all_nodes'][simplex[i]], setAll['all_nodes'][simplex[j]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.1, zorder=1)

    # 不同颜色标注不同属性的点
    plt.scatter(setAll['bottlenecks'][:,0], setAll['bottlenecks'][:,1], c='red', s=100, marker='h', label='Topological Bottlenecks')
    plt.scatter(setAll['coverage'][:,0], setAll['coverage'][:,1], c='blue', s=60, alpha=0.6, label='Coverage Nodes')
    plt.scatter(setAll['anchors'][:,0], setAll['anchors'][:,1], c='gold', s=200, marker='*', edgecolors='black', label='Strategic Anchors (Value/Start)')

    plt.title("Integrated Manifold Sampling: Topo + Coverage + Strategic Anchors")
    plt.legend(loc='upper right')
    # plt.show()

    # 再次处理障碍物和采样点
    removeEndPointAll = []
    for i in range(len(PPStart_Point)):
        for j in range(len(sampled_points)):
            obs, obs_no_circle, obs_no_circleTP, obs_no_circle_in, outline_all, removeEndPoint = obtain_finalObs(
                num, obs_mapsize, PPStart_Point[i], sampled_points[j], R, r,
                obs_no_circle, obs_no_circleTP, obs_no_circle_in, outline_all
            )
            if i == 0:
                removeEndPointAll.append(removeEndPoint)

    # 移除无效采样点
    removeEndPointAll = np.array(removeEndPointAll)
    sampled_points = sampled_points[removeEndPointAll == 0]

    # 设置过渡点角度
    # angles = (np.pi / 2) * np.ones(len(sampled_points))
    # Trans_Point = np.column_stack([sampled_points, angles])
  
    # 提取对应于 sampled_points 的优化后朝向 (从 setAll 中)
    # setAll['orientations'] 对应于 all_nodes 的顺序: anchors + bottlenecks + coverage
    n_anchors = len(setAll['anchors'])
    orientations_sampled = setAll['orientations'][n_anchors:]  # bottlenecks + coverage 的朝向
    orientations_sampled = orientations_sampled[removeEndPointAll == 0]  # 同步过滤
    
    # 将度数转换为弧度
    angles = np.deg2rad(orientations_sampled)
    Trans_Point = np.column_stack([sampled_points, angles])
    print('Trans_Point shape:', Trans_Point.shape)

    # 绘制最终地图
    plt.figure(figsize=(10, 8))
    Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)

    # 更新Map字典
    Map['obs'] = obs
    Map['sure'] = sure
    Map['r'] = r
    Map['obs_no_circle'] = obs_no_circle
    Map['obs_no_circle_in'] = obs_no_circle_in
    Map['outline_all'] = outline_all
    Map['Stepsize'] = Stepsize
    Map['resolution'] = resolution
    Map['Trans_Point'] = Trans_Point

    end_time = time.time()
    print(f"地图生成完成，耗时: {end_time - start_time:.2f} 秒")

    return Map


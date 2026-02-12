import numpy as np
from shapely.geometry import Polygon, Point
from shapely.prepared import prep

def obtain_finalObs(num, mapsize, Start_Point, End_Point, R, r, obs_no_circle, obs_no_circleTP, obs_no_circle_in, outline_all):
    """
    专家优化版：解决 np.isin 判定失效问题，提升计算效率
    """
    # 1. 生成随机圆形障碍物
    obs = np.column_stack([
        np.round(np.random.rand(num) * (mapsize[0][1] - mapsize[0][0]) + mapsize[0][0], 2),
        np.round(np.random.rand(num) * (mapsize[1][1] - mapsize[1][0]) + mapsize[1][0], 2),
        np.random.randint(R[0], R[1], num)
    ])

    removeInd_circles = []
    removeEndPoint = 0

    # 矢量化检查起点和终点与圆的碰撞
    if num > 0:
        dists_start = np.sqrt((Start_Point[0] - obs[:, 0])**2 + (Start_Point[1] - obs[:, 1])**2)
        dists_end = np.sqrt((End_Point[0] - obs[:, 0])**2 + (End_Point[1] - obs[:, 1])**2)
        
        removeInd_circles = np.where(dists_start < (obs[:, 2] + 2 * r))[0]
        if np.any(dists_end < (obs[:, 2] + 2 * r)):
            removeEndPoint = 1
        
        obs = np.delete(obs, removeInd_circles, axis=0)

    # 2. 处理多边形障碍物
    # 创建起点和终点的“安全判定圆”
    start_buffer = Point(Start_Point).buffer(2 * r)
    end_buffer = Point(End_Point).buffer(2 * r)
    
    ll = len(obs_no_circle)
    removeInd_poly = []

    for j in range(ll):
        if obs_no_circle[j] is not None:
            # 将 numpy 数组转为 Shapely 多边形
            poly_coords = obs_no_circle[j]
            if len(poly_coords) < 3: continue
            
            current_poly = Polygon(poly_coords)
            # 使用 prepared geometry 极大地加速空间判定
            prepared_poly = prep(current_poly)

            # --- 判定1: 障碍物是否覆盖起点区域 ---
            in1 = prepared_poly.intersects(start_buffer)
            
            # --- 判定2: 障碍物之间是否相互重叠 (保持原逻辑) ---
            in3 = False
            for k in range(j + 1, ll):
                if obs_no_circle[k] is not None:
                    next_poly = Polygon(obs_no_circle[k])
                    if current_poly.intersects(next_poly):
                        in3 = True
                        break
            
            # --- 判定3: 障碍物是否覆盖终点区域 ---
            in2 = prepared_poly.intersects(end_buffer)

            if in1 or in3:
                removeInd_poly.append(j)
                # 标记该索引下的障碍物失效
                obs_no_circle[j] = None
                obs_no_circleTP[j] = None
                obs_no_circle_in[j] = None
                # 同步删除轮廓线
                outline_all = outline_all[outline_all[:, 2] != j]
            elif in2:
                removeEndPoint = 1

    # 3. 移除超出地图范围的障碍物
    for j in range(len(obs_no_circle)):
        if obs_no_circle[j] is not None:
            x_coords = obs_no_circle[j][:, 0]
            y_coords = obs_no_circle[j][:, 1]
            if np.any(x_coords < 10) or np.any(y_coords < 10) or \
               np.any(x_coords > mapsize[0][1]) or np.any(y_coords > mapsize[1][1]):
                if j not in removeInd_poly:
                    obs_no_circle[j] = None
                    obs_no_circleTP[j] = None
                    obs_no_circle_in[j] = None
                    outline_all = outline_all[outline_all[:, 2] != j]

    return obs, obs_no_circle, obs_no_circleTP, obs_no_circle_in, outline_all, removeEndPoint
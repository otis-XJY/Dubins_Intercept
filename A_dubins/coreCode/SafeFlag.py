from matplotlib.path import Path
import numpy as np


def if_safe_no_circle(r, path, obs_no_circle_, outline, from_path, to_path, point):
    Distance = {}

    num = len(path)

    if from_path == 1:
        from_path = 1
    else:
        from_path = num - 2

    if to_path == 2:
        to_path = num -3
        # 修改为适用于 list 结构的索引方式
        start_sqre = point[:,num -3]
    elif to_path == 3:
        to_path = 3
        # 修改为适用于 list 结构的索引方式
        start_sqre = point[:,3]
    else:
        to_path = num
        # 修改为适用于 list 结构的索引方式
        start_sqre = point[:,num]

    from_path=from_path-1
    to_path = to_path - 1
    # 修改为适用于 list 结构的索引方式
    end_sqre = point[:,0]
    R_MAX = 2 * r

    # 计算路径的角度和四个点的坐标
    angle2 = np.arctan2(start_sqre[1] - end_sqre[1], start_sqre[0] - end_sqre[0])
    if angle2 + np.pi / 2 > np.pi:
        angle1 = angle2 + np.pi / 2 - np.pi
    else:
        angle1 = angle2 + np.pi / 2

    x = [
        start_sqre[0] + np.cos(angle2) * R_MAX + np.cos(angle1) * R_MAX,
        start_sqre[0] + np.cos(angle2) * R_MAX - np.cos(angle1) * R_MAX,
        end_sqre[0] - np.cos(angle2) * R_MAX - np.cos(angle1) * R_MAX,
        end_sqre[0] - np.cos(angle2) * R_MAX + np.cos(angle1) * R_MAX
    ]
    y = [
        start_sqre[1] + np.sin(angle2) * R_MAX + np.sin(angle1) * R_MAX,
        start_sqre[1] + np.sin(angle2) * R_MAX - np.sin(angle1) * R_MAX,
        end_sqre[1] - np.sin(angle2) * R_MAX - np.sin(angle1) * R_MAX,
        end_sqre[1] - np.sin(angle2) * R_MAX + np.sin(angle1) * R_MAX
    ]

    x.append(x[0])
    y.append(y[0])

    x_point = outline[:, 0]
    y_point = outline[:, 1]

    # bounding box for faster check
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    # check if points of outline are within the bounding box
    in_box = (x_point >= x_min) & (x_point <= x_max) & (y_point >= y_min) & (y_point <= y_max)

    id = np.where(in_box == 1)[0]
    obs_id = outline[id, 2]

    obs_to_avoid= np.unique(obs_id)

    obs_no_circle = [obs_no_circle_[int(i)] for i in obs_to_avoid]

    obs_id=[]
    flag_safe = [1] * len(obs_no_circle)
    for k, obs in enumerate(obs_no_circle):
        for j in range(from_path, to_path + 1):
            poly = Path(obs)
            edge_points = np.array(path[j]).T
            inside = poly.contains_points(edge_points, radius=-1e-10)

            # Use Path.contains_points to check if any of the path points are inside the polygon
            if np.any(inside):
                flag_safe[k] = 0
                obs_id.append(obs_to_avoid[k])
                break

    if not flag_safe:
        flag_safe = 1

    return flag_safe, np.array(obs_id)



def obtain_safe_point(point, outline, r):
    # 计算路径的角度和四个点的坐标
    R_MAX = 2 * r

    x_point = outline[:, 0]
    y_point = outline[:, 1]

    # bounding box for faster check
    x_min = point[0] - R_MAX
    x_max = point[0] + R_MAX
    y_min = point[1] - R_MAX
    y_max = point[1] + R_MAX

    # check if points of outline are within the bounding box
    in_box = (x_point >= x_min) & (x_point <= x_max) & (y_point >= y_min) & (y_point <= y_max)

    id = np.where(in_box == 1)[0]

    return len(id)

def if_safe_point(obs_no_circle, center, R):
    """
    判断圆（或点）是否在所有非圆障碍物区域之外
    :param obs_no_circle: list，每个元素是 (N,2) numpy 数组，表示障碍物多边形的顶点坐标
    :param center: (2,) numpy 数组或 list，圆心坐标
    :param R: float，圆半径。如果 R=0 就检查圆心本身
    :return: flag_safe = 1 (安全) / 0 (不安全)
    """

    if R != 0:
        th = np.arange(0, 2*np.pi+1e-6, np.pi/4)  # 0:pi/4:2pi
        edge1x = center[0] + R * np.cos(th)
        edge1y = center[1] + R * np.sin(th)
        edge_points = np.vstack([edge1x[:8], edge1y[:8]]).T
    else:
        edge_points = np.array([center])  # 单点

    flag_safe = 1
    for obs in obs_no_circle:
        if obs is not None and len(obs) > 0:
            poly = Path(obs)  # 构建多边形
            inside = poly.contains_points(edge_points,radius=-1e-10)  # 判断是否在内部
            if np.any(inside):
                flag_safe = 0
                break

    return flag_safe
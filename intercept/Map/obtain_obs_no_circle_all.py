import numpy as np
import random
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

def obtain_obs_no_circle_all(num_obs_nocircle, center, R, num_steps, Start_Point, End_Point, sure, MAPr):
    """
    获取所有不包含圆形障碍物的多边形，并进行扩展和插值。
    num_obs_nocircle: 障碍物的数量
    center: 障碍物中心的位置
    R: 障碍物大小的随机范围
    num_steps: 轮廓采样的步数
    Start_Point: 起点坐标
    End_Point: 终点坐标
    sure: 用于扩展障碍物的参数
    MAPr: 用于控制更大安全区域的扩展参数
    """
    outline_all = []
    obs_no_circle = []
    obs_no_circle_in = []
    obs_no_circleTP = []

    for i in range(num_obs_nocircle):
        outline, vertex, expandedPolygon, expandedPolygonTP = obtain_obs_nocircle(1, center, R, num_steps, Start_Point,
                                                                                  End_Point, sure, MAPr)

        if outline.size > 0:
            # 为每个障碍物附加编号
            # print(outline)
            outline_with_id = np.column_stack((outline, np.ones(len(outline[:, 0])) * (i)))
            outline_all.append(outline_with_id)
            obs_no_circle.append(expandedPolygon)
            obs_no_circleTP.append(expandedPolygonTP)
            obs_no_circle_in.append(vertex)

    return np.vstack(outline_all), obs_no_circle, obs_no_circleTP, obs_no_circle_in


def expandshape(polygon, radius):
    """
    扩展多边形形状

    参数:
    polygon: numpy数组，形状为(n, 2)，表示多边形的顶点坐标
    radius: 扩展半径

    返回:
    expandedPolygon: 扩展后的多边形顶点坐标
    """
    numVertices = polygon.shape[0]
    expandedPolygon = np.array([]).reshape(0, 2)

    for i in range(numVertices - 1):
        # 获取当前和下一个顶点
        currentVertex = polygon[i, :]
        nextVertex = polygon[(i + 1) % numVertices, :]

        # 计算边向量
        edgeVector = nextVertex - currentVertex

        # 计算边法向量（旋转90度）
        edgeNormal = -np.array([-edgeVector[1], edgeVector[0]])
        edgeNormal = edgeNormal / np.linalg.norm(edgeNormal)

        # 计算扩展顶点位置
        expandedVertex1 = currentVertex + radius * edgeNormal
        expandedVertex2 = nextVertex + radius * edgeNormal

        # 存储扩展顶点
        expandedPolygon = np.vstack([expandedPolygon, expandedVertex1])
        expandedPolygon = np.vstack([expandedPolygon, expandedVertex2])

    # 添加第一个点以闭合多边形
    expandedPolygon = np.vstack([expandedPolygon, expandedPolygon[0, :]])

    # # 可选：绘制原始多边形和扩展后的多边形
    # plt.figure()
    # plt.plot(polygon[:, 0], polygon[:, 1], 'r.-')  # 绘制原始多边形
    # plt.plot(expandedPolygon[:, 0], expandedPolygon[:, 1], 'b.-')  # 绘制扩展多边形
    #
    # # 在原始顶点处绘制圆
    # theta = np.linspace(0, 2 * np.pi, 100)
    # for i in range(numVertices):
    #     center = polygon[i, :]
    #     circleX = center[0] + radius * np.cos(theta)
    #     circleY = center[1] + radius * np.sin(theta)
    #     plt.plot(circleX, circleY, 'g-')
    #
    # plt.axis('equal')
    # plt.show()

    return expandedPolygon


def obtain_obs_nocircle(id, mapsize, R, num_steps, Start_Point, End_Point, sure, MAPr):
    """
    获取不含圆形障碍物的多边形障碍物，并对其进行扩展和插值。
    id: 障碍物的标识
    mapsize: 地图大小
    R: 障碍物大小的随机范围
    num_steps: 轮廓采样的步数
    Start_Point: 起点坐标
    End_Point: 终点坐标
    sure: 用于扩展障碍物的参数
    MAPr: 用于控制更大安全区域的扩展参数
    """
    # 如果 id == 1, 随机生成圆心
    if id == 1:
        center_x = random.randint(mapsize[0][0], mapsize[0][1] - 1)
        center_y = random.randint(mapsize[1][0], mapsize[1][1] - 1)
        center = [center_x, center_y]
    else:
        center = mapsize  # 否则设置为地图大小

    # 随机生成障碍物的半径
    # print(R)
    r = random.randint(R[0],R[1])
    num_points = 10  # 随机点的数量
    radius = r * np.sqrt(np.random.rand(num_points))  # 随机半径
    theta = 2 * np.pi * np.random.rand(num_points)  # 随机角度

    # 转换为笛卡尔坐标系
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    # 组合坐标点
    points = np.column_stack((x, y))

    # 构建凸包
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # 保存顶点
    vertex = hull_points

    vertex = np.vstack([vertex, vertex[0, :]])

    # 扩展多边形
    expandedPolygon = expandshape(vertex, sure)

    # 再次扩展
    expandedPolygonTP = expandshape(expandedPolygon, 3 * MAPr)

    polygon = Polygon(expandedPolygon)
    # print(Start_Point)
    # pointS = Point(Start_Point[0], Start_Point[1])
    # pointE = Point(End_Point[0], End_Point[1])
    #
    # # 检查起点和终点是否在障碍物内
    # in1 = polygon.contains(pointS)
    # in2 = polygon.contains(pointE)

    # 只取 x 和 y 坐标，忽略 theta
    pointS = Start_Point[:, :2]  # 选取所有行的前两列，作为 x 和 y 坐标
    pointE = End_Point[:, :2]  # 选取所有行的前两列，作为 x 和 y 坐标

    # 创建 Point 对象的数组（不使用循环）
    pointS_objects = [Point(p[0], p[1]) for p in pointS]
    pointE_objects = [Point(p[0], p[1]) for p in pointE]

    # 使用 numpy 向量化操作批量判断哪些点在多边形内
    # 通过 Map 操作, 判断每个点是否在多边形内
    in1 = np.array([polygon.contains(p) for p in pointS_objects])
    in2 = np.array([polygon.contains(p) for p in pointE_objects])



    # in1 = plt.mlab.inside(Start_Point[0], Start_Point[1], expandedPolygon[:, 0], expandedPolygon[:, 1])
    # in2 = plt.mlab.inside(End_Point[0], End_Point[1], expandedPolygon[:, 0], expandedPolygon[:, 1])

    outline_all = []

    if not np.any(in1) and not np.any(in2):
        # 对障碍物的轮廓进行插值
        for i in range(len(expandedPolygon) - 1):
            x_interp = np.linspace(expandedPolygon[i, 0], expandedPolygon[i + 1, 0], num_steps)
            y_interp = np.linspace(expandedPolygon[i, 1], expandedPolygon[i + 1, 1], num_steps)
            outline = np.column_stack((x_interp, y_interp))
            outline_all.extend(outline)

    return np.array(outline_all), vertex, expandedPolygon, expandedPolygonTP


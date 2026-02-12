import numpy as np

def obtain_new_center(a, b, c, Start_Point, R, center_old, type, total_field, pos, resolution):
    """
    获取沿直线垂直方向移动的中心点。

    Parameters:
        a, b, c : float
            直线的参数 ax + by + c = 0
        Start_Point : list of float
            起点坐标 [x, y]
        R : float
            半径
        center_old : list of float
            原始中心点 [x, y]
        type : int
            1：靠近已知直线；2：远离已知直线
        total_field : object
            总场景（暂时未使用）
        pos : int
            位置ID（暂时未使用）
        resolution : float
            分辨率

    Returns:
        center_new : np.array
            新的中心点
        poit_num : int
            中心点的数量
    """
    distance = np.arange(-R, 1.5 * R, resolution)
    k_vertical = -1 / a

    norm_vertical1 = np.array([1, k_vertical]) / np.linalg.norm([1, k_vertical])
    displacement1 = distance[:, None] * norm_vertical1
    norm_vertical2 = -norm_vertical1
    displacement2 = distance[:, None] * norm_vertical2

    new_point = Start_Point[:2] + displacement1[0]
    distances = (a * new_point[0] + b * new_point[1] + c) / np.sqrt(a ** 2 + b ** 2)
    singal = (a * center_old[0] + b * center_old[1] + c) / np.sqrt(a ** 2 + b ** 2)

    if type == 1:
        if singal * distances >= 0:
            center_new = center_old + displacement2
        else:
            center_new = center_old + displacement1

    elif type == 2:
        if singal * distances >= 0:
            center_new = center_old + displacement1
        else:
            center_new = center_old + displacement2

    poit_num = len(center_new)
    return center_new, poit_num

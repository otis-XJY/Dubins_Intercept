import numpy as np

def obtain_h(start, goal):
    """
    计算起始点到目标点的欧氏距离。

    Parameters:
        start : list of float
            起始点坐标 [x, y]
        goal : list of float
            目标点坐标 [x, y]

    Returns:
        h : float
            起始点到目标点的欧氏距离
    """
    h = 1.414 * np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
    return h
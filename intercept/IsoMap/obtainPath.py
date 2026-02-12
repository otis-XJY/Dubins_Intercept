import numpy as np


def obtainPath(close, vthetaAllP, verse):
    """
    从close列表中提取路径点

    参数:
    close: 包含路径节点的列表
    vthetaAllP: 方向角数组
    verse: 方向标志 (0: 正向, 1: 反向)

    返回:
    path: 路径点数组，形状为 (3, N) [x坐标, y坐标, 方向角]
    """
    numNode = len(close)
    path_x = []
    path_y = []

    # 处理中间节点 (从第2个到倒数第2个)
    for i in range(1, numNode - 1):  # Python是0-based索引
        if 'path' in close[i]:
            for le in range(len(close[i]['path'])):
                path_x.extend(close[i]['path'][le][0])
                path_y.extend(close[i]['path'][le][1])

    # 处理最后一个节点
    if numNode > 0 and 'path' in close[numNode - 1]:
        for le in range(len(close[numNode - 1]['path'])):
            path_x.extend(close[numNode - 1]['path'][le][0])
            path_y.extend(close[numNode - 1]['path'][le][1])

    # 转换为numpy数组
    path_x = np.array(path_x)
    path_y = np.array(path_y)

    if verse == 0:
        # 正向
        path = np.vstack([path_x, path_y, vthetaAllP])
    else:
        # 反向
        path = np.vstack([np.flip(path_x), np.flip(path_y), np.flip(vthetaAllP)])

    return path


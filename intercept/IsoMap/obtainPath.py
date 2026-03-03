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

    # 转换为numpy数组，使用float32节省内存，并四舍五入到小数点后两位
    path_x = np.round(np.array(path_x, dtype=np.float32), 2)
    path_y = np.round(np.array(path_y, dtype=np.float32), 2)
    
    # 处理vthetaAllP，转换为float32并四舍五入
    vthetaAllP = np.round(np.asarray(vthetaAllP, dtype=np.float32), 4)

    if verse == 0:
        # 正向
        path = np.vstack([path_x, path_y, vthetaAllP])
    else:
        # 反向
        path = np.vstack([np.round(np.flip(path_x), 2), 
                         np.round(np.flip(path_y), 2), 
                         np.round(np.flip(vthetaAllP), 4)])

    # 确保最终结果是float32格式
    path = np.asarray(path, dtype=np.float32)
    
    return path


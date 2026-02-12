import numpy as np
from A_dubins.coreCode.obtain_h import obtain_h

def update_close(path_safe_flag, param_all, End_Point, close):
    """
    更新关闭列表。

    Parameters:
        path_safe_flag : int
            被选中的安全路径（扁平索引）
        param_all : list of lists
            每个元素为一个障碍/列的候选参数列表
        End_Point : list of float
            目标点坐标 [x, y]
        close : list of dict
            关闭列表，每个字典包含一个节点

    Returns:
        close : list of dict
            更新后的关闭列表
    """
    # map flattened index to (col, idx_in_col)
    num = path_safe_flag
    col_lengths = [len(col) for col in param_all]
    cum = np.cumsum(col_lengths)
    col = int(np.searchsorted(cum, num, side='right'))
    prev = int(cum[col - 1]) if col > 0 else 0
    idx_in_col = int(num - prev)

    param = param_all[col][idx_in_col]

    # 构建新的节点
    node = {
        'point': param['point'],
        'vtheta': param['vtheta'][1:],
        'vtheta_plot': param['vtheta_plot'],
        'vtheta_all': param['vtheta_all'],
        'g': close[-1]['g'] + sum(param['length']),
        'f': close[-1]['g'] + sum(param['length']) + obtain_h(param['point'][:, -1], End_Point),
        'alllength': close[-1]['g'] + param['length'],
        'path': param['path'],
        'r': param['r'],
        'center': param['center'],
        'pos_id': param['pos_id'],
        'parent_id': param['parent_id']
    }
    close = np.append(close, node)

    return close

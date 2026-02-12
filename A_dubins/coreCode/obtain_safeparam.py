# core/obtain_safeparam.py
import numpy as np
def obtain_safeparam(path_safe_flag, param_all):
    """
    根据安全路径标志从param_all中提取安全路径的参数。
    
    参数:
        path_safe_flag: 安全路径的标志列表
        param_all: 路径参数
    
    返回:
        param_safe: 安全路径的参数
    """
    # Build a list-of-lists where each index corresponds to an obstacle (column)
    param_safe = []
    if not param_all:
        return param_safe

    # param_all is list of columns; prepare param_safe with same number of columns
    param_safe = [[] for _ in param_all]

    if not path_safe_flag:
        return param_safe

    # compute cumulative lengths to map flattened indices to (col, idx)
    col_lengths = [len(col) for col in param_all]
    cum = np.cumsum(col_lengths)

    for num in path_safe_flag:
        col = int(np.searchsorted(cum, num, side='right'))
        prev = int(cum[col - 1]) if col > 0 else 0
        idx_in_col = int(num - prev)
        param_safe[col].append(param_all[col][idx_in_col])

    return param_safe
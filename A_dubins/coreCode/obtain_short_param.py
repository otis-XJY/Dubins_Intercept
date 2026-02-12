import numpy as np
def obtain_short_param(param_all):
    """
    该函数根据路径长度筛选出最短的路径参数。

    Parameters:
        param_all : list
            包含多个路径参数的列表（每个元素为一列候选）。

    Returns:
        param_all_new : list
            筛选后的路径参数列表，仅保留每列中最短的一部分路径。
    """
    param_all_new = []

    for param_group in param_all:
        # 对于每个列（参数组），单独收集长度
        lengths = []
        for param in param_group:
            if 'Length' not in param or param['Length'] is None:
                lengths.append(float('inf'))
            else:
                lengths.append(param['Length'])

        lengths = np.array(lengths)

        # 找到非无穷的有效索引
        non_inf_indices = np.where(lengths != np.inf)[0]
        if non_inf_indices.size == 0:
            # 本列没有有效路径，返回空列表占位
            param_all_new.append([])
            continue

        # 按长度排序，取前一半的候选（保留原行为的意图）
        sorted_idx = np.argsort(lengths)
        take = max(1, len(non_inf_indices) // 2)
        min_two_index = sorted_idx[:take]

        # 筛选出最短路径的参数（按列内索引）
        param_group_new = [param_group[int(idx)] for idx in min_two_index]
        param_all_new.append(param_group_new)

    return param_all_new

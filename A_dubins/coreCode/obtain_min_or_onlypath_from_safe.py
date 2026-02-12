# core/obtain_min_or_onlypath_from_safe.py
import numpy as np

def obtain_min_or_onlypath_from_safe(mode, param_all, flag_all):
    """
    根据不同的类型选择路径：
    type=1: 选择最短路径
    type=2: 仅选择安全路径
    type=3: 选择最短路径

    参数:
        type: 类型决定选择方式
        param_all: 路径参数
        flag_all: 路径安全标志

    返回:
        min_path: 最短路径
        path_safe_flag: 安全路径的标志
        flag_min: 最短路径的标志
    """
    # param_all is expected to be a list of columns (each column is a list of params)
    # path_safe_flag contains flattened indices into the concatenated columns

    # flag_all is expected as a flat list of 0/1 flags for all candidates (flattened across columns)
    path_safe_flag = []
    for i in range(len(flag_all)):
        try:
            if np.all(np.array(flag_all[i]) == 1):
                path_safe_flag.append(i)
        except Exception:
            # In some cases flag_all entries may be scalars (0/1)
            if flag_all[i] == 1:
                path_safe_flag.append(i)

    min_path = float('inf')
    flag_min = 0

    # Helper: map a flattened index to (col, idx_in_col)
    col_lengths = [len(col) for col in param_all]
    cum = np.cumsum(col_lengths)

    # Defensive validation: ensure each candidate is a dict with expected keys
    for col_i, col in enumerate(param_all):
        for idx_in_col, item in enumerate(col):
            if not isinstance(item, dict):
                # Build a safe type-only description and include a short summary of column element types
                item_desc = item.__class__.__name__
                col_types = [e.__class__.__name__ for e in col]
                # For nested lists, collect a short inner-type summary for each element
                nested_summary = []
                for e in col[:6]:
                    if isinstance(e, list):
                        nested_summary.append([x.__class__.__name__ for x in e[:6]])
                    else:
                        nested_summary.append(e.__class__.__name__)
                raise TypeError(f'Invalid candidate type in param_all at column {col_i}, index {idx_in_col}: '
                                f'{item_desc}. Column types (first 10): {col_types[:10]}. Nested sample: {nested_summary}')
            if 'Length' not in item:
                raise KeyError(f"Candidate dict missing 'Length' at column {col_i}, index {idx_in_col}: {type(item).__name__}")

    if mode == 3:
        # choose the absolute shortest path among all candidates
        flat_idx = 0
        for col_i in range(len(param_all)):
            for item in param_all[col_i]:
                if item['Length'] < min_path:
                    min_path = item['Length']
                    flag_min = flat_idx
                flat_idx += 1

    if mode == 1:
        # choose shortest among safe paths (path_safe_flag are flattened indices)
        if path_safe_flag:
            best = None
            for i, num in enumerate(path_safe_flag):
                col = int(np.searchsorted(cum, num, side='right'))
                prev = int(cum[col - 1]) if col > 0 else 0
                idx_in_col = int(num - prev)
                candidate = param_all[col][idx_in_col]
                # candidate is validated above; safe to access 'Length'
                if candidate['Length'] < min_path:
                    min_path = candidate['Length']
                    flag_min = i  # index within path_safe_flag
                    best = candidate

    return min_path, path_safe_flag, flag_min


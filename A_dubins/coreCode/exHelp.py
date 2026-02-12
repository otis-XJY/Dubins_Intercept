def effective_len(x):
    # None 或空列表 → 0
    if x is None or len(x) == 0:
        return 0
    # 特判 [[]]
    if len(x) == 1 and isinstance(x[0], list) and len(x[0]) == 0:
        return 0
    return len(x)

import numpy as np

def obtainNearETP(Trans_Point_2D, PosE, ValuePos, pairsE2Val, weight_perp=1.0, weight_path=1.0):
    """
    全向量化寻找最接近路径段的转移点 (TP)
    
    综合考虑两个因素：
    1. Trans_Point 到线段 PosE-ValuePos 的垂直距离 (dist_perp)
    2. 折线距离：PosE -> Trans_Point -> ValuePos 的总路径距离 (dist_path)
    
    参数:
    - weight_perp: 垂直距离的权重 (默认1.0)
    - weight_path: 折线距离的权重 (默认1.0)
    """
    # 1. 提取坐标 (调整为 0-based 索引)
    # pairsE2Val 第一列是 Eid, 第二列是 Valid
    A_all = PosE[:, :2]  # (M, 2)
    B_all = ValuePos[pairsE2Val[:, 1].astype(int), :2] # (M, 2)
    
    num_lines = A_all.shape[0]
    num_points = Trans_Point_2D.shape[0]
    eps = 1e-12

    # 2. 准备向量化计算所需的分量
    # D: 方向向量 (M, 2)
    D = B_all - A_all
    D_mag_sq = np.sum(D**2, axis=1, keepdims=True) + eps # (M, 1)

    # 提取转移点坐标 (N, 2)
    Px = Trans_Point_2D[:, 0]
    Py = Trans_Point_2D[:, 1]

    # --- 3. 广播计算投影系数 t (核心优化) ---
    # 利用维度扩展实现 (M, 1) 与 (1, N) 的运算，生成 (M, N) 矩阵
    # t[i, j] 表示第 i 条线段相对于第 j 个 TP 的投影系数
    dx_tp = Px[np.newaxis, :] - A_all[:, [0]] # (M, N)
    dy_tp = Py[np.newaxis, :] - A_all[:, [1]] # (M, N)
    
    t = (dx_tp * D[:, [0]] + dy_tp * D[:, [1]]) / D_mag_sq # (M, N)

    # --- 4. 广播计算点到直线的垂直距离 dist_perp ---
    # 公式: |Dx*(Ay - Py) - Dy*(Ax - Px)| / sqrt(D_mag_sq)
    num = np.abs(D[:, [0]] * (A_all[:, [1]] - Py[np.newaxis, :]) - 
                 D[:, [1]] * (A_all[:, [0]] - Px[np.newaxis, :])) # (M, N)
    dist_perp = num / np.sqrt(D_mag_sq) # (M, N)

    # --- 4b. 计算折线距离：PosE -> Trans_Point -> ValuePos ---
    # Trans_Point 到 PosE 的距离: (M, N)
    dist_PE = np.sqrt((Px[np.newaxis, :] - A_all[:, [0]])**2 + 
                      (Py[np.newaxis, :] - A_all[:, [1]])**2)
    
    # Trans_Point 到 ValuePos 的距离: (M, N)
    dist_TV = np.sqrt((Px[np.newaxis, :] - B_all[:, [0]])**2 + 
                      (Py[np.newaxis, :] - B_all[:, [1]])**2)
    
    # 折线总距离
    dist_path = dist_PE + dist_TV

    # --- 5. 引入惩罚项 ---
    # outside_mask: (M, N) 类型的布尔矩阵
    outside_mask = (t < 0) | (t > 1)
    
    # 使用 np.where 替代 if/else 循环赋值
    # 只要在 [0, 1] 之外，增加 1e6 的极大偏移量
    penalty = np.where(outside_mask, 1e6 + 100 * dist_perp, 0.0)
    
    # --- 5b. 归一化两个距离指标 (考虑量纲差异) ---
    # 计算 dist_perp 和 dist_path 的最大值用于归一化
    dist_perp_max = np.max(dist_perp) + eps
    dist_path_max = np.max(dist_path) + eps
    
    # 归一化到 [0, 1] 范围
    dist_perp_norm = dist_perp / dist_perp_max
    dist_path_norm = dist_path / dist_path_max
    
    # --- 6. 组合两个指标 ---
    total_score = weight_perp * dist_perp_norm + weight_path * dist_path_norm + penalty

    # --- 7. 寻优 (axis=1 代表在 TP 维度找最小值) ---
    min_idx = np.argmin(total_score, axis=1) # (M,)
    
    # 提取结果
    # 注意：返回的 ID 保持 1-based 以兼容原 MATLAB 后续逻辑
    NearTPid_E = min_idx
    NearTPpos_E = Trans_Point_2D[min_idx, :]

    return NearTPpos_E, NearTPid_E.reshape(-1, 1)


import numpy as np

def obtainNearTP(Trans_Point, PosP, PosE):
    """
    向量化实现：从 Trans_Point 中找到每组 PosP 视野范围内最靠近 PosP 的点。
    """
    # 提取坐标和角度
    PosP_ = PosP[:, :2]  # (n, 2)
    PosE_ = PosE[:, :2]  # (n, 2)
    thetaP = PosP[:, 2]  # (n,)
    thetaE = PosE[:, 2]  # (n,)
    
    n = PosP_.shape[0]
    num_trans = Trans_Point.shape[0]

    # 1. 计算朝向向量 vP: (n, 2)
    vP = np.column_stack([np.cos(thetaP), np.sin(thetaP)])

    # 2. 计算相对位移向量: (n, num_trans, 2)
    # 利用广播：Trans_Point (num_trans, 2) 扩展为 (1, num_trans, 2)
    # PosP_ (n, 2) 扩展为 (n, 1, 2)
    rel_pos = Trans_Point[np.newaxis, :, :] - PosP_[:, np.newaxis, :]

    # 3. 判断是否在区域 B (inP): (n, num_trans)
    # 向量化点积：sum(rel_pos * vP_expanded, axis=2)
    inP = np.sum(rel_pos * vP[:, np.newaxis, :], axis=2) >= 0

    # 4. 计算所有距离平方: (n, num_trans)
    dists_sq = np.sum(rel_pos**2, axis=2)

    # 5. 核心逻辑：寻找最佳索引
    # 创建带惩罚的距离矩阵：不在区域内的点距离设为无穷大
    masked_dists = np.where(inP, dists_sq, np.inf)

    # 找到每行的最小索引
    BestIndex = np.argmin(masked_dists, axis=1)

    # 6. 处理 Fallback (退路) 逻辑
    # 找出那些在区域内完全没有候选点的行
    no_candidates = np.all(~inP, axis=1)
    
    if np.any(no_candidates):
        # 对于这些行，重新在原始 dists_sq 中找最小值
        BestIndex[no_candidates] = np.argmin(dists_sq[no_candidates], axis=1)

    # 7. 提取坐标
    BestPoints = Trans_Point[BestIndex, :]

    # 转换回 1-based 索引以保持与 MATLAB 逻辑一致 (可选)
    # 如果后续 Python 代码已改为 0-based，请删除 + 1
    return BestPoints, (BestIndex).reshape(-1, 1)
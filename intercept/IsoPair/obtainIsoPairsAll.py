import numpy as np

def obtainIsoPairs(IsoEpos, IsoPpos, PosE, PosP, threshold, ValuePos, pairs, Eid, ETP2Val_IsoPosId):
    """
    Python 版拦截点对计算函数
    """
    # 如果输入为空，直接返回
    if IsoEpos is None or IsoPpos is None or IsoEpos.size == 0 or IsoPpos.size == 0:
        return np.array([])

    # --- 索引筛选 (对应 MATLAB find 逻辑) ---
    # MATLAB: idx1 = find(pairs(Eid,2) == ETP2Val_IsoPosId(:,2))
    # 注意：Python 索引 0 对应 MATLAB 1，此处假设输入已做相应对齐
    target_val = pairs[Eid, 1]
    idx1 = np.where(ETP2Val_IsoPosId[:, 1] == target_val)[0]
    idx2 = np.where(ETP2Val_IsoPosId[:, 1] == -1)[0]
    idx = np.concatenate([idx1, idx2])

    if idx.size == 0:
        return np.array([])

    # --- Step 1. 提取坐标与朝向 ---
    # IsoEpos: [x; y; theta], 大小为 3 x n
    Epos = IsoEpos[:2, idx].T       # 形状 (nE, 2)
    Ppos = IsoPpos[:2, :].T        # 形状 (nP, 2)
    # 提取朝向角 (弧度)
    Eori = IsoEpos[2, idx]          # (nE,)
    Pori = IsoPpos[2, :]           # (nP,)

    # --- Step 2. 角度差与代价 (向量化广播) ---
    # dx, dy 均为 (nE, nP) 矩阵
    dx = Epos[:, 0:1] - Ppos[:, 0]
    dy = Epos[:, 1:2] - Ppos[:, 1]
    
    # 计算 P 到 E 的方位角 (转换为度)
    angPE = np.degrees(np.arctan2(dy, dx))
    
    # 计算朝向偏差: angPE - Pori (Pori 转换为度)
    # 利用广播: (nE, nP) - (nP,) -> (nE, nP)
    angleDiff_ = angPE - np.degrees(Pori)
    
    # 角度归一化到 [-180, 180]
    angleDiff = (angleDiff_ + 180) % 360 - 180
    
    # 计算方向代价
    costDirection = np.abs(angleDiff) / 43.5
    # 注意：在 Python 中我们后面直接用 Mask 过滤，不必特意转成 NaN

    validAngleMask = np.abs(angleDiff) <= 43.5
    if not np.any(validAngleMask):
        return np.array([])

    # --- Step 3. 距离判断 ---
    dist = np.hypot(dx, dy)
    validMask = validAngleMask & (dist <= threshold)

    # 计算 E 到目标点 (ValuePos) 的距离
    target_pos_idx = int(pairs[Eid, 1])
    # 如果 MATLAB 中 pairs(Eid,2) 是 1-based 索引，这里需要 -1，视项目习惯而定
    Valuepos_single = ValuePos[target_pos_idx, :2] 
    distE2Value = np.hypot(Epos[:, 0] - Valuepos_single[0], Epos[:, 1] - Valuepos_single[1])

    # --- Step 4 & 5. 结果收集 ---
    if not np.any(validMask):
        # 对应 MATLAB: 即使 matchpairs 计算了也不返回结果
        return np.array([])
    else:
        # 找到所有符合条件的索引
        # idxE_rel 是在 Epos 里的索引，idxP 是在 IsoPpos 里的索引
        idxE_rel, idxP = np.where(validMask)
        
        # 提取对应数据
        distances_val = dist[idxE_rel, idxP]
        angle_val = angleDiff[idxE_rel, idxP]
        distE2ValueF = distE2Value[idxE_rel]
        
        # 将相对索引 idxE_rel 映射回原始 IsoEpos 的索引 idx
        original_idxE = idx[idxE_rel]

        # 构造结果矩阵: [idxE, idxP, dist, angle, distE2Value]
        # 注意：若后续代码需要 MATLAB 风格的 1-based 索引，请给 original_idxE 和 idxP 加 1
        pairsAns = np.column_stack((
            original_idxE, 
            idxP, 
            distances_val, 
            angle_val, 
            distE2ValueF
        ))
        
    return pairsAns


import numpy as np

def obtainPTP2TP_IsoPos_timeShift2(Pid, Eid, tp, te, PosP, PosE, CapDist, IsoMap_i_tt_insertedP, IsoMap_i_tt_insertedE, ValuePos, pairs):
    """
    根据时间偏移提取等时面点对 (Time-Shifted IsoPairs)
    """
    # --- 1. 辅助安全提取函数 ---
    def obtainsafeGetIsoPos(s):
        # 模拟 MATLAB 的 obtainsafeGetIsoPos
        return s.IsoPos if s is not None and hasattr(s, 'IsoPos') else None

    def obtainsafeGetpathid(s):
        # 模拟 MATLAB 的 obtainsafeGetpathid
        return s.pathid if s is not None and hasattr(s, 'pathid') else None

    # --- 2. 提取特定 Agent 的时间序列数据 ---
    # MATLAB: PTP2TP_ = IsoMap_i_tt_insertedP(Pid)
    # Python: Pid, Eid 为 1-based 索引，转换为 0-based
    P_agent_data = IsoMap_i_tt_insertedP[int(Pid)]
    E_agent_data = IsoMap_i_tt_insertedE[int(Eid)]

    # --- 3. 提取特定时间步的数据 (te, tp) ---
    # MATLAB: ETP2Val_IsoPos{te}, PTP2TP_IsoPos{tp}
    # Python: te, tp 为 1-based 索引，转换为 0-based
    
    # 提取 Evader 在 te 时刻的数据
    s_e = E_agent_data[int(te)] if int(te) < len(E_agent_data) else None
    ETP2Val_IsoPos_te = obtainsafeGetIsoPos(s_e)
    ETP2Val_IsoPosId_te = obtainsafeGetpathid(s_e)

    # 提取 Pursuer 在 tp 时刻的数据
    s_p = P_agent_data[int(tp)] if int(tp) < len(P_agent_data) else None
    PTP2TP_IsoPos_tp = obtainsafeGetIsoPos(s_p)

    # --- 4. 计算拦截对 (obtainIsoPairs) ---
    # 注意：PosE(Eid,:) 和 PosP(Pid,:) 的索引映射
    # pairs 为已转换好的任务配对矩阵
    output = obtainIsoPairs(
        ETP2Val_IsoPos_te, 
        PTP2TP_IsoPos_tp, 
        PosE[int(Eid), :], 
        PosP[int(Pid), :], 
        CapDist, 
        ValuePos, 
        pairs, 
        Eid, 
        ETP2Val_IsoPosId_te
    )

    return output
import numpy as np

import numpy as np

import numpy as np

def obtainCost(flattened_pairs, pathidP, CapRef, Pid_ref_flat, reward_max=1.0):
    """
    计算代价函数：数值越小表示方案越优。
    计算代价函数：路径长度、空间距离、时间差距、朝向偏差以及安全距离的加权综合。
    各子 reward 由原始量（Delta_t、Delta_d、Delta_L、|Delta_v|、Delta_D）与阈值直接计算，
    与对应子代价的语义一致且 reward ∈ [0, reward_max]，避免先算 cost 再反推带来的量化损失。
    
    参数:
    flattened_pairs: NumPy 矩阵，包含 [te, tp, ..., dist, Angle, distE2ValueF]
    pathidP: NumPy 矩阵，包含 [..., ..., path_len]
    CapRef: 含 CapDist、CapDistTime、CapAngle、CapAngleTime、v_P、timeIsoRes 等
    Pid_ref_flat: 用于映射原始 Pid 到索引的数组
    reward_max: 各子 reward 的上界（下界为 0），默认 1.0
    
    返回:
    cost: 综合代价向量 (M,)
    costAll: 特征矩阵 (M, 15)：前 5 列为 Delta_*，中 5 列为各子 cost，后 5 列为对应 reward
    """
    # --- 0. 参数提取与广播 ---
    CapDist = CapRef['CapDist']
    CapDistTime_ = np.array(CapRef['CapDistTime'])
    # 根据 Pid 索引映射对应的 CapDistTime
    CapDistTime = CapDistTime_[Pid_ref_flat.astype(int)]

    CapAngle = CapRef['CapAngle']/2
    CapAngleTime_ = np.array(CapRef['CapAngleTime'])/2
    CapAngleTime = CapAngleTime_[Pid_ref_flat.astype(int)]
    
    v_P = CapRef['v_P']
    timeIsoRes = CapRef['timeIsoRes']
    
    # --- 1. 权重参数设置 ---
    param_L = 1.0    # 路径长度权重
    param_d = 1.5    # 拦截空间距离权重 (重要)
    param_t = 1.5    # 时间匹配权重 (重要)
    param_v = 1.0    # 朝向偏差权重
    param_D = 2.0    # 目标安全距离权重 (关键：防止逃避者接近目标)

    # --- 2. 特征提取 ---
    Delta_t = flattened_pairs[:, 0] - flattened_pairs[:, 1]  # te - tp
    Delta_d = flattened_pairs[:, 6]                         # 拦截点空间距离
    Delta_L = pathidP[:, 2]                                 # 追捕者路径长度
    Delta_v = flattened_pairs[:, 7]                         # 角度偏差
    Delta_D = flattened_pairs[:, 8]                         # 拦截点距离目标的距离

    eps = np.finfo(float).eps

    # --- 3. 代价函数计算 (越小越优) ---
    
    # A. 路径长度代价: 越短越小
    max_L = np.max(Delta_L)
    cost_L = Delta_L / (max_L + eps)
    
    # B. 拦截点距离代价: 
    # Delta_d < CapDist 时为 0 (理想)
    # CapDist ~ CapDistTime 之间从 0 线性增加到 1
    # 大于 CapDistTime 为 1 (差)
    cost_d = np.where(
        Delta_d <= CapDist,
        0.0,
        np.where(
            Delta_d <= CapDistTime,
            (Delta_d - CapDist) / (CapDistTime - CapDist + eps),
            1.0
        )
    )
    
    # C. 朝向角度代价: 偏差越小 sin(0)=0 越小
    # B. 朝向角度代价（分段线性）
    # 0 ~ CapAngle: 0
    # CapAngle ~ CapAngleTime: 0→1 线性
    # CapAngleTime ~ 180: 1
    abs_delta_v=np.abs(Delta_v)
    cost_v = np.where(
        abs_delta_v <= CapAngle,
        0.0,
        np.where(
            abs_delta_v <= CapAngleTime,
            (abs_delta_v - CapAngle) / (CapAngleTime - CapAngle + eps),
            1.0
        )
    )
    
    # D. 时间匹配代价:
    # Delta_t >= 0 (追捕者比逃避者早到或同时到) 代价为 0
    # 稍微晚到 (-threshold ~ 0) 代价从 0 线性增加到 1
    # 太晚 ( < -threshold) 代价为 1
    time_threshold = (CapDistTime / v_P) / timeIsoRes
    cost_t = np.where(
        Delta_t >= 0,
        0.0,
        np.where(
            Delta_t >= -time_threshold,
            np.abs(Delta_t) / (time_threshold + eps), # 越晚(负值越大)代价越高
            1.0
        )
    )
    
    # E. 目标安全代价:
    # Delta_D <= CapDist：拦截点离目标过近，代价取满 1。
    # Delta_D > CapDist：代价随距离增大从 1 连续衰减至 0（指数衰减，与 reward 对偶）。
    k_D = 0.022
    cost_D = np.where(
        Delta_D <= CapDist,
        1.0,
        np.exp(-k_D * (Delta_D - CapDist)),
    )

    # --- 4. 综合总代价 ---
    cost = (param_L * cost_L + 
            param_d * cost_d + 
            param_v * cost_v + 
            param_t * cost_t + 
            param_D * cost_D)

    # --- 5. 子 reward：由原始量直接计算（与上式各 cost_* 满足 reward = reward_max*(1-cost_*)，但不经过 cost 中间量）---
    # L：路径越短越好 — reward_L = reward_max * (1 - Delta_L/(max_L+eps))
    reward_L = reward_max * (max_L - Delta_L) / (max_L + eps)

    # d：Delta_d 在 [CapDist, CapDistTime] 内对 (CapDistTime - Delta_d) 线性；≤CapDist 满分；>CapDistTime 为 0
    reward_d = np.where(
        Delta_d <= CapDist,
        reward_max,
        np.where(
            Delta_d <= CapDistTime,
            reward_max * (CapDistTime - Delta_d) / (CapDistTime - CapDist + eps),
            0.0,
        ),
    )

    # v：角度 — 与 cost_v 分段线性对偶
    reward_v = np.where(
        abs_delta_v <= CapAngle,
        reward_max,
        np.where(
            abs_delta_v <= CapAngleTime,
            reward_max * (CapAngleTime - abs_delta_v) / (CapAngleTime - CapAngle + eps),
            0.0,
        ),
    )

    # t：Delta_t>=0 满分；[-time_threshold,0) 内对 (time_threshold+eps+Delta_t) 线性；再晚为 0
    reward_t = np.where(
        Delta_t >= 0,
        reward_max,
        np.where(
            Delta_t >= -time_threshold,
            reward_max * (time_threshold + eps + Delta_t) / (time_threshold + eps),
            0.0,
        ),
    )

    # D：由 Delta_D、CapDist、k_D 直接计算（与 cost_D 对偶，但不经过 cost_D 以免舍入误差）
    reward_D = np.where(
        Delta_D <= CapDist,
        0.0,
        reward_max * (1.0 - np.exp(-k_D * (Delta_D - CapDist))),
    )

    # 结果拼接：Delta → 子 cost → 子 reward（与 cost 列顺序一致：t,d,v,L,D）
    costAll = np.column_stack((
        Delta_t, Delta_d, Delta_v, Delta_L, Delta_D,
        reward_t, reward_d, reward_v, reward_L, reward_D,
    ))

    return cost, costAll

import numpy as np

def obtainTask(IsoPairs_time_Eid_Pid_Posid, IsoMapP_i_tt, IsoMapEin_i_tt,CapRef):
    """
    代价函数计算与任务候选表生成
    
    参数:
    IsoPairs_time_Eid_Pid_Posid: 包含 {te, tp, Eid, Pid, pairs_matrix, eid_ref, pid_ref} 的列表或对象数组
    IsoMapP_i_tt: 嵌套列表，存放 Pursuer 的 IsoMap 结构体
    IsoMapEin_i_tt: 嵌套列表，存放 Evader 的 IsoMap 结构体
    CapRef: 包含拦截参数的字典
    """
    
    if len(IsoPairs_time_Eid_Pid_Posid) == 0:
        return np.empty((0, 16))

    # --- 1. 展开所有 pairs 行 (替代 cellfun + vertcat) ---
    # 获取每一行中 pairs 矩阵的行数，用于快速平铺元数据
    # IsoPairs_... 的列索引：0:te, 1:tp, 2:Eid, 3:Pid, 4:pairs, 5:eid_ref, 6:pid_ref
    row_counts = [row[4].shape[0] for row in IsoPairs_time_Eid_Pid_Posid]
    
    # 向量化平铺元数据
    te_flat = np.repeat([row[0] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    tp_flat = np.repeat([row[1] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    eid_logical_flat = np.repeat([row[2] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    pid_logical_flat = np.repeat([row[3] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    eid_ref_flat = np.repeat([row[5] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    pid_ref_flat = np.repeat([row[6] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)

    # 垂直堆叠所有的 pairs 矩阵 (取前 5 列: IsoidxE, IsoidxP, dist, Angle1, Angle2)
    pairs_matrix_all = np.vstack([row[4][:, :5] for row in IsoPairs_time_Eid_Pid_Posid])

    # 构造 flattened_pairs [te, tp, Eid, Pid, IsoidxE, IsoidxP, dist, Angle1, Angle2]
    # 注意：Python 0-based 索引，对应 MATLAB 的 1-9 列
    flattened_pairs = np.column_stack((
        te_flat, tp_flat, eid_logical_flat, pid_logical_flat, pairs_matrix_all
    ))
    
    # 构造 PEid [eid_ref, pid_ref]
    PEid = np.column_stack((eid_ref_flat, pid_ref_flat))

    # --- 2. 提取路径信息 (替代 arrayfun) ---
    # 由于 IsoMap 是非规整结构，我们使用列表推导式进行快速属性访问
    # MATLAB 索引参考：flattened_pairs(:, 2)->tp, (:, 4)->Pid, (:, 6)->IsoidxP
    
    # 获取 Pursuer 的路径信息
    # 减 1 是因为 MATLAB 存储的索引是 1-based，Python 访问数组需 0-based
    pathidP = np.array([
        IsoMapP_i_tt[int(p)][int(t)].pathid[int(idxP), :]
        for t, p, idxP in zip(tp_flat, pid_logical_flat, flattened_pairs[:, 5])
    ])

    # 获取 Evader 的路径信息
    pathidE = np.array([
        IsoMapEin_i_tt[int(p)][int(t)].pathid[int(idxE), :]
        for t, p, idxE in zip(te_flat, eid_logical_flat, flattened_pairs[:, 4])
    ])

    # --- 3. 构造代价函数 ---
    # 调用外部定义的 obtainCost 函数 (假设该函数已转换)
    # 传入 flattened_pairs 和 pathidP
    cost, costAll = obtainCost(flattened_pairs, pathidP, CapRef,pid_ref_flat)

    # --- 4. 拼接最终候选表 ---
    # 对应 MATLAB: [flattened_pairs(:,1:7), pathidP(:,3), cost, pathidE(:,2), pathidP(:,2), PEid, pathidE(:,3), pathidP(:,3), costAll]
    # 列含义：
    # 0-6: te, tp, Eid, Pid, IsoidxE, IsoidxP, dist
    # 7: path_len (pathidP 第 3 列)
    # 8: cost
    # 9: ValPosid (pathidE 第 2 列)
    # 10: TPid (pathidP 第 2 列)
    # 11-12: Eid_ref, Pid_ref
    # 13: Eiso (pathidE 第 3 列)
    # 14: Piso (pathidP 第 3 列)
    # 15+: costAll（15 列：Delta×5 + 子 cost×5 + 子 reward×5）
    
    InterceptCandidates = np.column_stack((
        flattened_pairs[:, :7],   # te, tp, Eid, Pid, IsoidxE, IsoidxP, dist
        pathidP[:, 2],            # path_len
        cost,                     # cost
        pathidE[:, 1],            # Eid对应的ValPosid
        pathidP[:, 1],            # Pid对应的TPid
        PEid,                     # 原始 Eid, Pid 引用
        pathidE[:, 2],            # Eiso
        pathidP[:, 2],            # Piso
        costAll                   # Delta、子代价与子 reward
    ))

    return InterceptCandidates


import numpy as np

def obtainTask_timeShift2(IsoPairs_time_Eid_Pid_Posid, IsoMap_i_tt_insertedP, IsoMap_i_tt_insertedE, CapRef):
    """
    代价函数计算与任务候选表生成 (Time-Shift 版本)
    
    参数:
    IsoPairs_time_Eid_Pid_Posid: 包含 {te, tp, ETPid, PTPid, pairs_matrix, Eid, Pid} 的列表
    IsoMap_i_tt_insertedP: 拼接后的 Pursuer 等时面嵌套列表 [agent][time]
    IsoMap_i_tt_insertedE: 拼接后的 Evader 等时面嵌套列表 [agent][time]
    """
    
    if len(IsoPairs_time_Eid_Pid_Posid) == 0:
        return np.empty((0, 18))

    # --- 1. 展开所有 pairs 行 (元数据扩张) ---
    # IsoPairs 列定义: 0:te, 1:tp, 2:ETPid, 3:PTPid, 4:pairs_sub, 5:Eid, 6:Pid
    row_counts = [row[4].shape[0] for row in IsoPairs_time_Eid_Pid_Posid]
    
    # 扩张 te, tp, ETPid, PTPid
    te_flat = np.repeat([row[0] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    tp_flat = np.repeat([row[1] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    ETPid_flat = np.repeat([row[2] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    PTPid_flat = np.repeat([row[3] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    
    # 扩张 原始 ID: Eid, Pid (对应 MATLAB 的 PEid) 真实值
    Eid_flat = np.repeat([row[5] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    Pid_flat = np.repeat([row[6] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    # 扩张 原始 ID: Eid, Pid (对应 MATLAB 的 PEid) 相对值
    Eid_ref_flat = np.repeat([row[7] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)
    Pid_ref_flat = np.repeat([row[8] for row in IsoPairs_time_Eid_Pid_Posid], row_counts)

    # 垂直堆叠所有的 pairs 子矩阵 (取前 5 列: IsoidxE, IsoidxP, dist, Angle1, Angle2)
    pairs_matrix_all = np.vstack([row[4][:, :5] for row in IsoPairs_time_Eid_Pid_Posid])

    # 构造 flattened_pairs [te, tp, ETPid, PTPid, IsoidxE, IsoidxP, dist] (7列)
    flattened_pairs = np.column_stack((
        te_flat, tp_flat, ETPid_flat, PTPid_flat, pairs_matrix_all
    ))
    
    # 构造 PEid [Eid, Pid] (对应 MATLAB 的 12, 13 列)
    PEid = np.column_stack((Eid_ref_flat, Pid_ref_flat))

    # --- 2. 提取路径长度 (替代 arrayfun) ---
    # Pursuer 路径信息提取 (注意 0-based 索引)
    # 索引参数: t_idx (tp_flat), agent_id (Pid_ref_flat), iso_idx (flattened_pairs[:, 5])
    pathidP = np.array([
        IsoMap_i_tt_insertedP[int(pid)][int(t)].pathid[int(idxP), :]
        for t, pid, idxP in zip(tp_flat, Pid_ref_flat, flattened_pairs[:, 5])
    ])

    # Evader 路径信息提取
    # 索引参数: t_idx (te_flat), agent_id (Eid_ref_flat), iso_idx (flattened_pairs[:, 4])
    pathidE = np.array([
        IsoMap_i_tt_insertedE[int(eid)][int(t)].pathid[int(idxE), :]
        for t, eid, idxE in zip(te_flat, Eid_ref_flat, flattened_pairs[:, 4])
    ])

    
    # --- 3. 构造代价函数 ---
    # 调用之前转换好的 obtainCost 函数
    cost, costAll = obtainCost(flattened_pairs, pathidP, CapRef,Pid_ref_flat)

    # --- 4. 拼接最终候选表 ---
    # MATLAB 拼接顺序：
    # [flattened_pairs(:,1:7), pathidP(:,3), cost, pathidE(:,2), pathidP(:,2), PEid, pathidE(:,3), pathidP(:,3), costAll]
    
    # 列解释 (18列):
    # 0-1: te, tp
    # 2-3: ETPid, PTPid
    # 4-5: IsoPosidxE, IsoPosidxP
    # 6: Iso_dist
    # 7: path_len (pathidP[:, 2])
    # 8: cost
    # 9: Eid对应的ValPosid (pathidE[:, 1])
    # 10: Pid对应的TPid (pathidP[:, 1])
    # 11-12: Eid, Pid (原始ID)/[0 1]而不是[0 2]
    # 13: Eiso (pathidE[:, 2])
    # 14: Piso (pathidP[:, 2])
    # 15+: costAll（15 列：Delta×5 + 子 cost×5 + 子 reward×5）
    
    InterceptCandidates = np.column_stack((
        flattened_pairs[:,:7],          # 0-6 (te, tp, ETPid, PTPid, idxE, idxP, dist)
        pathidP[:, 2],            # 7: path_len
        cost,                     # 8: cost
        pathidE[:, 1],            # 9: Eid对应的ValPosid
        pathidP[:, 1],            # 10: Pid对应的TPid
        PEid,                     # 11-12: Eid, Pid
        pathidE[:, 2],            # 13: Eiso
        pathidP[:, 2],            # 14: Piso
        costAll                   # 15-29: Delta、子代价与子 reward
    ))

    return InterceptCandidates
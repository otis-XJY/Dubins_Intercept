import numpy as np

import numpy as np

def obtainCost(flattened_pairs, pathidP):
    """
    计算代价函数：路径长度、空间距离、时间差距、朝向偏差以及安全距离的加权综合。
    
    参数:
    flattened_pairs: NumPy 矩阵，包含 [te, tp, ..., dist, Angle, distE2ValueF]
    pathidP: NumPy 矩阵，包含 [..., ..., path_len]
    
    返回:
    cost: 综合代价向量 (M,)
    costAll: 原始特征矩阵 (M, 3)
    """
    
    # --- 1. 权重参数设置 ---
    param_L = 1.0    # 路径长度权重
    param_d = 1.2    # 拦截点之间距离权重
    param_t = 1.5    # 时间权重
    param_v = 1.2    # 朝向权重
    param_D = 2.0    # 打击安全权重

    # --- 2. 特征提取 (0-based 索引转换) ---
    # MATLAB: flattened_pairs(:,1) - flattened_pairs(:,2)
    Delta_t = flattened_pairs[:, 0] - flattened_pairs[:, 1]
    
    # MATLAB: flattened_pairs(:,7) (拦截点距离)
    Delta_d = flattened_pairs[:, 6]
    
    # MATLAB: pathidP(:,3) (路径长度)
    Delta_L = pathidP[:, 2]
    
    # MATLAB: flattened_pairs(:,8) (偏离中心角度/朝向)
    Delta_v = flattened_pairs[:, 7]
    
    # MATLAB: flattened_pairs(:,9) (距离目标的距离)
    Delta_D = flattened_pairs[:, 8]

    # --- 3. 代价函数计算 (完全向量化) ---
    
    # 防止除以 0 的小常数
    eps = np.finfo(float).eps
    
    # 路径长度部分: 线性归一化
    # 注意：如果 Delta_L 全部相同，max 为 0，需处理
    max_L = np.max(Delta_L)
    cost_L = param_L * Delta_L / (max_L + eps)
    
    # 拦截点距离部分: Sigmoid 函数
    cost_d = param_d * 1.0 / (1.0 + np.exp(-0.1381 * (Delta_d - 50.0)))
    
    # 朝向部分: 正弦映射
    cost_v = param_v * np.sin(np.pi * np.abs(Delta_v) / (2.0 * 43.5))
    
    # 时间差距部分: 指数衰减 (时间差越大，代价越小)
    cost_t = param_t * np.exp(-0.5365 * Delta_t)
    
    # 安全距离部分: Sigmoid 反向映射
    cost_D = param_D * 1.0 / (1.0 + np.exp(0.022 * (Delta_D - 199.99)))

    # 综合代价
    cost = cost_L + cost_d + cost_v + cost_t + cost_D

    # --- 4. 结果拼接 ---
    # MATLAB: [Delta_t, Delta_d, Delta_D]
    costAll = np.column_stack((Delta_t, Delta_d, Delta_D))

    return cost, costAll

import numpy as np

def obtainTask(IsoPairs_time_Eid_Pid_Posid, IsoMapP_i_tt, IsoMapEin_i_tt):
    """
    代价函数计算与任务候选表生成
    
    参数:
    IsoPairs_time_Eid_Pid_Posid: 包含 {te, tp, Eid, Pid, pairs_matrix, eid_ref, pid_ref} 的列表或对象数组
    IsoMapP_i_tt: 嵌套列表，存放 Pursuer 的 IsoMap 结构体
    IsoMapEin_i_tt: 嵌套列表，存放 Evader 的 IsoMap 结构体
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
    cost, costAll = obtainCost(flattened_pairs, pathidP)

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
    # 15: costAll
    
    InterceptCandidates = np.column_stack((
        flattened_pairs[:, :7],   # te, tp, Eid, Pid, IsoidxE, IsoidxP, dist
        pathidP[:, 2],            # path_len
        cost,                     # cost
        pathidE[:, 1],            # Eid对应的ValPosid
        pathidP[:, 1],            # Pid对应的TPid
        PEid,                     # 原始 Eid, Pid 引用
        pathidE[:, 2],            # Eiso
        pathidP[:, 2],            # Piso
        costAll                   # 原始全量代价
    ))

    return InterceptCandidates


import numpy as np

def obtainTask_timeShift2(IsoPairs_time_Eid_Pid_Posid, IsoMap_i_tt_insertedP, IsoMap_i_tt_insertedE, ValuePos=None):
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
    cost, costAll = obtainCost(flattened_pairs, pathidP)

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
    # 11-12: Eid, Pid (原始ID)
    # 13: Eiso (pathidE[:, 2])
    # 14: Piso (pathidP[:, 2])
    # 15-17: costAll (如果是向量/矩阵，拼接全量)
    
    InterceptCandidates = np.column_stack((
        flattened_pairs[:,:7],          # 0-6 (te, tp, ETPid, PTPid, idxE, idxP, dist)
        pathidP[:, 2],            # 7: path_len
        cost,                     # 8: cost
        pathidE[:, 1],            # 9: Eid对应的ValPosid
        pathidP[:, 1],            # 10: Pid对应的TPid
        PEid,                     # 11-12: Eid, Pid
        pathidE[:, 2],            # 13: Eiso
        pathidP[:, 2],            # 14: Piso
        costAll                   # 15-17: costAll
    ))

    return InterceptCandidates
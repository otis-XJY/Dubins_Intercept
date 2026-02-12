import numpy as np
from intercept.IsoPair.obtainIsoPath import obtainPE2IsoPath,obtainIsoMapP2TP,obtainIsoMapIso2TP
def obtainNearTPIso(Stepsize, TPIso_IsoPos, Iso2TP_pathid, PosP, PosTP):
    """
    寻找距离当前位置 PosP 最近的等时面点 (Iso Point)
    """
    # 1. 判空处理
    if TPIso_IsoPos is None or TPIso_IsoPos.size == 0:
        return []

    # 2. 提取 2D 坐标 (确保为 2xN)
    TPIso_IsoPos_sub = TPIso_IsoPos[:2, :] 
    
    # 3. 计算等时面点到 TP 的预存路径长度
    # MATLAB: Iso2TP_pathid(:, 3) -> Python: Iso2TP_pathid[:, 2]
    Iso2TP_length = Iso2TP_pathid[:, 2] * Stepsize

    # === 朝向向量计算 (保留逻辑，如需启用可取消注释) ===
    # vTP = np.array([np.cos(PosTP[2]), np.sin(PosTP[2])])
    # vP = np.array([np.cos(PosP[2]), np.sin(PosP[2])])
    # (此处省略了原代码中被注释掉的区域合法性过滤)

    # 4. 向量化计算 PosP 到所有等时面点的欧氏距离
    # 利用 NumPy 广播: (2, N) - (2, 1) -> (2, N)
    diff = TPIso_IsoPos_sub - PosP[:2].reshape(2, 1)
    # 计算范数 (欧氏距离)，axis=0 得到 (N,) 向量
    dist_euclidean = np.linalg.norm(diff, axis=0)

    # 5. 总距离 = PosP到Iso的直线距离 + Iso到TP的路径距离
    dist = dist_euclidean + Iso2TP_length

    # 6. 寻找最小值索引
    idx = np.argmin(dist)
    dmin = dist[idx]

    # 返回 [索引, 距离]
    # 注意：返回 idx + 1 (1-based)，因为调用函数 obtainPE2TP 
    # 使用了 int(iso_pid) - 1 进行反向索引
    return [idx, dmin]

import numpy as np

import numpy as np

def obtainPE2TP(IsoMapPTP2Iso_i_tt, NearTPid_P, PosP, Trans_Point, pathFinalMapPTP2Iso, Map, v):
    """
    获得当前位置 P/E 到最近的 TP 的路径和时间信息。
    结构与 main 函数中的 results 保持完全一致：使用列表嵌套列表。
    """
    # --- 1. 数据准备 ---
    num_TPs = len(IsoMapPTP2Iso_i_tt)
    num_Times = len(IsoMapPTP2Iso_i_tt[0])
    num_Agents = len(NearTPid_P)
    
    # 提取 Iso 属性列表
    Iso2TP_IsoPos = [[item.IsoPos for item in row] for row in IsoMapPTP2Iso_i_tt]
    Iso2TP_pathid = [[item.pathid for item in row] for row in IsoMapPTP2Iso_i_tt]

    # --- 2. 构造扁平化索引 (修正为 Agent-major 顺序) ---
    # meshgrid(Agents, Times) -> indexing='ij' 保证 a_flat 是 [0,0,0, 1,1,1...]
    a_grid, t_grid = np.meshgrid(np.arange(num_Agents), np.arange(num_Times), indexing='ij')
    a_flat = a_grid.ravel()
    t_flat = t_grid.ravel()
    
    # 获取每个 Agent 对应的 TP 索引 (0-based)
    tp_indices = NearTPid_P.flatten().astype(int)
    tp_flat = tp_indices[a_flat]

    # --- 3. 构造 results (列表嵌套列表结构) ---
    # 结构: [ [time, TPid, PE_id, [IsoPosId, dist]], ... ]
    # 严格对应 MATLAB: {time, TPid, PE_id, obtainNearTPIso(...)}
    results = [
        [
            t,                               # time (1-based)
            tp,                              # TPid (1-based)
            a,                               # PE_id (1-based)
            obtainNearTPIso(                     # 内部包含 [IsoPosId, dist]
                Map['Stepsize'], Iso2TP_IsoPos[tp][t], Iso2TP_pathid[tp][t], 
                PosP[a, :], Trans_Point[tp, :]
            )
        ]
        for a, t, tp in zip(a_flat, t_flat, tp_flat)
    ]

    # --- 4. 寻优与提取 (同步 obtainResult 安全逻辑) ---
    # 距离矩阵 (Agent x Time): 对应 c{4}(2)
    all_dist = np.array([
        r[3][1] if (r[3] is not None and len(r[3]) >= 2) else np.inf 
        for r in results
    ]).reshape(num_Agents, num_Times)

    # ID 矩阵 (Agent x Time): 对应 c{4}(1)
    all_IsoPosId = np.array([
        r[3][0] if (r[3] is not None and len(r[3]) >= 1) else np.nan 
        for r in results
    ]).reshape(num_Agents, num_Times)

    # 找到每个 Agent 距离最小的时间索引
    TimeId_idx = np.argmin(all_dist, axis=1) 
    
    # 提取 P2TPPairs_... (对应 MATLAB 的 results(linear_indices))
    agent_row_idx = np.arange(num_Agents)
    linear_indices = agent_row_idx * num_Times + TimeId_idx
    P2TPPairs_time_TPid_PEid_IsoPosid = [results[idx] for idx in linear_indices]
    
    # 获取选中的最优 ID
    IsoPosId_win = all_IsoPosId[agent_row_idx, TimeId_idx]

    # --- 5. 路径参数提取 ---
    # TP_win 为每个 Agent 选定的 TP (0-based)
    TP_win = (NearTPid_P.flatten()).astype(int)
    
    # 获取 pathid 矩阵: [pid1, pid2, isoid]
    PathId = np.array([
        IsoMapPTP2Iso_i_tt[tp][tid].pathid[int(iso_id), :]
        for tp, tid, iso_id in zip(TP_win, TimeId_idx, IsoPosId_win)
    ])

    # --- 6. 路径切片 ---
    PathIso2TP = [
        pathFinalMapPTP2Iso[int(p[0])][int(p[1])][:, -int(p[2]):]
        for p in PathId
    ]
    
    PosIso2TP = [
        pathFinalMapPTP2Iso[int(p[0])][int(p[1])][:, -int(p[2])]
        for p in PathId
    ]
    PIsoPos = np.vstack([p.reshape(1, -1) for p in PosIso2TP])

    # --- 7. 调用局部规划与合并函数 ---
    PathP2Iso, IsoMapP2Iso_i_tt, IsoMapP2Iso_EndTime = obtainPE2IsoPath(PosP, PIsoPos, Map, v)
    P2IsoL = [path.shape[1] for path in PathP2Iso]

    IsoMapIso2TP_i_tt, IsoMapIso2TP_EndTime = obtainIsoMapIso2TP(v, PathIso2TP, Map)
    
    # 合并 IsoMap
    IsoMapP2TP_i_tt = obtainIsoMapP2TP(
        IsoMapP2Iso_i_tt, IsoMapIso2TP_i_tt, 
        IsoMapP2Iso_EndTime, IsoMapIso2TP_EndTime, 
        P2IsoL
    )

    # --- 8. 路径拼接 ---
    PathP2TP = [
        np.hstack((p_iso, p_tp)) 
        for p_iso, p_tp in zip(PathP2Iso, PathIso2TP)
    ]

    return PathP2TP, IsoMapP2TP_i_tt, P2TPPairs_time_TPid_PEid_IsoPosid



import numpy as np
import time
from intercept.IsoMap.obtainPath import obtainPath
from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm, _PLANNER_TIMEOUT_SEC
# 假设已经定义了存储结构的类，与你示例中的 IsoMapData 对应
from dataclasses import dataclass
@dataclass
class IsoMapData:
    """存储等值线数据的结构类"""
    IsoPos: np.ndarray      # 等值线位置坐标
    pathid: np.ndarray      # 路径ID
    IsoVtheta: np.ndarray   # 等值线速度角度

def obtainPE2IsoPath(PosP, PIsoPos, Map, v):
    """
    计算从当前位置 PosP 到 PIsoPos 的 Dubins 路径，并生成对应的等时面映射。
    """
    # --- 1. 参数提取 ---
    obs = Map.get('obs')
    sure = Map.get('sure')
    r = Map.get('r')
    obs_no_circle = Map.get('obs_no_circle')
    outline_all = Map.get('outline_all')
    step_size = Map.get('Stepsize')
    resolution = Map.get('resolution')
    num_time = Map.get('numTime')
    time_plot = Map.get('timePlot')
    
    num_agents = PosP.shape[0]
    
    # 临时存储
    final_pathP2Iso_segments = [None] * num_agents
    vthetaAllP2Iso = [None] * num_agents
    
    # --- 2. 路径规划 (Dubins A*) ---
    # 此处必须使用循环处理每个 Agent 的独立路径
    start_time = time.time()
    # 设置全局规划截止时间，防止路径规划无限循环
    deadline = time.monotonic() + _PLANNER_TIMEOUT_SEC
    for i in range(num_agents):
        # 调用 Dubins 路径规划 (对应 MATLAB 的 A_dubins_nocircle_swarm)
        # 注意：PIsoPos[i, :] 对应 i 代理的目标 Iso 位置
        path_segments, _, _, _ = A_dubins_nocircle_swarm(
            PosP[i, :],
            PIsoPos[i, :],
            obs, sure, r,
            obs_no_circle,
            outline_all,
            step_size,
            1,
            resolution,
            deadline=deadline
        )

        final_pathP2Iso_segments[i] = path_segments

        # Dubins_A 路径构建失败时，使用直线作为兜底路径
        if path_segments is None:
            print(f'[等时面路径] 代理 {i} Dubins_A规划失败，使用直线兜底')
            sx, sy = PosP[i, 0], PosP[i, 1]
            ex, ey = PIsoPos[i, 0], PIsoPos[i, 1]
            dx, dy = ex - sx, ey - sy
            dist = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)
            n_pts = max(int(np.ceil(dist / step_size)) + 1, 2)
            xs = np.linspace(sx, ex, n_pts, dtype=np.float32)
            ys = np.linspace(sy, ey, n_pts, dtype=np.float32)
            thetas = np.full(n_pts, direction, dtype=np.float32)
            vthetaAllP2Iso[i] = thetas
            # 标记为直线兜底，第3步直接构造路径矩阵
            final_pathP2Iso_segments[i] = ('straight', np.vstack([xs, ys, thetas]))
            continue

        # 超时检查：若已超过截止时间则剩余代理使用直线兜底
        # 注意：必须从 i 开始（包含当前代理），否则 final_pathP2Iso_segments[i] 非 None
        # 但 vthetaAllP2Iso[i] 仍为 None，导致 obtainPath 崩溃
        if time.monotonic() > deadline:
            print(f'[等时面路径] 规划超时，已完成 {i}/{num_agents} 个代理，剩余使用直线兜底')
            for j in range(i, num_agents):
                sx, sy = PosP[j, 0], PosP[j, 1]
                ex, ey = PIsoPos[j, 0], PIsoPos[j, 1]
                dx, dy = ex - sx, ey - sy
                dist = np.sqrt(dx**2 + dy**2)
                direction = np.arctan2(dy, dx)
                n_pts = max(int(np.ceil(dist / step_size)) + 1, 2)
                xs = np.linspace(sx, ex, n_pts, dtype=np.float32)
                ys = np.linspace(sy, ey, n_pts, dtype=np.float32)
                thetas = np.full(n_pts, direction, dtype=np.float32)
                vthetaAllP2Iso[j] = thetas
                final_pathP2Iso_segments[j] = ('straight', np.vstack([xs, ys, thetas]))
            break

        # 提取并展平 vtheta_all (速度角集合)
        vtheta_list = []
        if path_segments is not None:
            # 模拟 MATLAB 的 tt_=1:length-1 配合 tt_+1 逻辑
            for tt in range(1, len(path_segments)):
                segment_vtheta = path_segments[tt].get('vtheta_all', [])
                for k_theta in segment_vtheta:
                    if isinstance(k_theta, (list, np.ndarray)):
                        vtheta_list.extend(k_theta)
                    else:
                        vtheta_list.append(k_theta)

        vthetaAllP2Iso[i] = np.array(vtheta_list)

    # --- 3. 轨迹点提取 (obtainPath) ---
    pathFinalP2Iso = [None] * num_agents
    for i in range(num_agents):
        seg = final_pathP2Iso_segments[i]
        if seg is not None:
            if isinstance(seg, tuple) and seg[0] == 'straight':
                # 直线兜底路径，直接使用预构造的矩阵
                pathFinalP2Iso[i] = seg[1]
            else:
                # 转换路径段为连续的坐标点矩阵
                pathFinalP2Iso[i] = obtainPath(seg, vthetaAllP2Iso[i], 0)

    # --- 4. 构造等时面映射 (IsoMapP2Iso_i_tt) ---
    IsoMapP2Iso_i_tt = [[{} for _ in range(num_time)] for _ in range(num_agents)]
    IsoMapP2Iso_EndTime = [None] * num_agents
    EndTimeFlag = np.zeros(num_agents)
    v_E = v  # 速度 m/s

    for tt in range(num_time):
        # 计算当前时间步对应的路径点索引
        # MATLAB: round(v_E * Map.timePlot(tt)) 
        # Python 索引从 0 开始，因此需要处理
        target_idx = int(round(v_E * time_plot[tt])) - 1
        
        for i in range(num_agents):
            current_path = pathFinalP2Iso[i]
            
            if current_path is not None:
                path_len = current_path.shape[1]
                
                # 如果当前路径长度足够覆盖预测时间点
                if path_len > target_idx and target_idx >= 0:
                    # 提取对应索引的位置和角度
                    IsoPos = current_path[:, target_idx].reshape(-1, 1)
                    IsoVtheta = vthetaAllP2Iso[i][target_idx] if len(vthetaAllP2Iso[i]) > target_idx else 0
                    
                    # 构造 pathid: [-1, -1, 对应索引] (保持 MATLAB 逻辑)
                    pathid = np.array([[-1, -1, target_idx]]) # +1 保持 1-based 参考
                    
                    IsoMapP2Iso_i_tt[i][tt] = IsoMapData(
                        IsoPos=IsoPos,
                        pathid=pathid,
                        IsoVtheta=np.array([[IsoVtheta]])
                    )
                else:
                    # 路径不足，记录结束时间步
                    if EndTimeFlag[i] == 0:
                        IsoMapP2Iso_EndTime[i] = tt  # 1-based 
                        EndTimeFlag[i] = 1

    print(f"PE2IsoPath 计算完成，耗时: {time.time() - start_time:.4f} 秒")
    return pathFinalP2Iso, IsoMapP2Iso_i_tt, IsoMapP2Iso_EndTime


import numpy as np

def obtainIsoMapIso2TP(v, PathIso2TP, Map):
    """
    根据给定的 Iso 到 TP 的路径，生成对应的等时面映射。
    
    参数:
        v: 速度 (m/s)
        PathIso2TP: 路径列表，每个元素为 3xL 的矩阵 (0:1行为坐标, 2行为角度)
        Map: 包含时间信息的字典 (numTime, timePlot)
        
    返回:
        IsoMapIso2TP_i_tt: 嵌套列表 [agent][time]，存储 IsoMapData 对象
        IsoMapIso2TP_EndTime: 记录每个 agent 路径结束的时间步索引
    """
    # --- 1. 参数初始化 ---
    num_agents = len(PathIso2TP)
    num_time = Map.get('numTime')
    time_plot = Map.get('timePlot')
    v_E = v  # 速度 m/s
    
    # 初始化输出容器
    # IsoMapIso2TP_i_tt[agent][time]
    IsoMapIso2TP_i_tt = [[{} for _ in range(num_time)] for _ in range(num_agents)]
    IsoMapIso2TP_EndTime = [None] * num_agents
    EndTimeFlag = np.zeros(num_agents)

    # --- 2. 核心提取逻辑 ---
    # 提前计算所有时间步对应的目标索引，避免在循环内重复计算
    # MATLAB: round(v_E * Map.timePlot(tt)) -> Python: 索引需减 1
    target_indices = np.round(v_E * np.array(time_plot)).astype(int) - 1

    for tt in range(num_time):
        target_idx = target_indices[tt]
        
        for i in range(num_agents):
            current_path = PathIso2TP[i]
            
            # 检查路径是否存在
            if current_path is not None and current_path.size > 0:
                path_len = current_path.shape[1]
                
                # 判断当前路径长度是否足够覆盖该时间点
                if path_len > target_idx and target_idx >= 0:
                    # 提取位置 (前两行)
                    IsoPos = current_path[:, target_idx].reshape(-1, 1)
                    # 提取角度 (第三行，MATLAB index 3 -> Python index 2)
                    IsoVtheta = current_path[2, target_idx]
                    
                    # 构造 pathid: [-1, -1, target_idx + 1] (保持 1-based 参考)
                    pathid = np.array([[-1, -1, target_idx]])
                    
                    # 存储数据
                    IsoMapIso2TP_i_tt[i][tt] = IsoMapData(
                        IsoPos=IsoPos,
                        pathid=pathid,
                        IsoVtheta=np.array([[IsoVtheta]])
                    )
                else:
                    # 如果路径长度不足且尚未记录结束标志
                    if EndTimeFlag[i] == 0:
                        IsoMapIso2TP_EndTime[i] = tt  # 记录 1-based 时间步索引
                        EndTimeFlag[i] = 1

    return IsoMapIso2TP_i_tt, IsoMapIso2TP_EndTime


import numpy as np

def obtainIsoMapP2TP(IsoMapP2Iso_i_tt, IsoMapIso2TP_i_tt, IsoMapP2Iso_EndTime, IsoMapIso2TP_EndTime, P2IsoL):
    """
    将两段等时面映射拼接为完整的 P 到 TP 的等时面序列。
    
    参数:
        IsoMapP2Iso_i_tt: 第一阶段等时面数据 [agent][time]
        IsoMapIso2TP_i_tt: 第二阶段等时面数据 [agent][time]
        IsoMapP2Iso_EndTime: 第一阶段结束的时间步 (1-based)
        IsoMapIso2TP_EndTime: 第二阶段结束的时间步 (1-based)
        P2IsoL: 第一段路径的长度列表，用于索引偏移
        
    返回:
        IsoMapP2TP_i_tt: 拼接后的等时面序列 [agent][time_list]
    """
    num_agents = len(IsoMapP2Iso_i_tt)
    IsoMapP2TP_i_tt = [None] * num_agents

    for i in range(num_agents):
        # 获取结束时间步（如果为 None 则取最大范围）
        # MATLAB 1:P2IsoTime-1 对应 Python 切片 [:P2IsoTime-1]
        p2iso_limit = int(IsoMapP2Iso_EndTime[i]) if IsoMapP2Iso_EndTime[i] is not None else len(IsoMapP2Iso_i_tt[i])
        iso2tp_limit = int(IsoMapIso2TP_EndTime[i]) if IsoMapIso2TP_EndTime[i] is not None else len(IsoMapIso2TP_i_tt[0])

        # --- 更新第二部分数据 (替代 cellfun + setfield) ---
        # 逻辑：将 pathid 的第三个元素（本地路径索引）加上第一段路径的长度 P2IsoL[i]
        updated_second_part = [
            IsoMapData(
                IsoPos=item.IsoPos,
                # pathid 结构为 [[i, j, idx]]，更新 idx
                pathid=np.array([[item.pathid[0, 0], item.pathid[0, 1], item.pathid[0, 2] + P2IsoL[i]]]) 
                       if item.pathid is not None and item.pathid.size > 0 else None,
                IsoVtheta=item.IsoVtheta
            )
            for item in IsoMapIso2TP_i_tt[i][:iso2tp_limit]
        ]

        # --- 提取第一部分数据 ---
        first_part = IsoMapP2Iso_i_tt[i][:p2iso_limit]

        # --- 拼接结果 ---
        # MATLAB: [IsoMapP2Iso_i_tt{i,1:P2IsoTime-1}, IsoMapIso2TP_i_tt{i,1:Iso2TPTime-1}]
        IsoMapP2TP_i_tt[i] = first_part + updated_second_part

    return IsoMapP2TP_i_tt


import numpy as np
import copy

def insertIsoMapP2TP(IsoMap_i_tt, IsoMapP2TP, E2TP_TPIdx, PathP2TP):
    """
    拼接等时面数据：将 P2TP 段插入到原始 IsoMap 的前端，并更新索引偏移。
    
    输入：
        IsoMap_i_tt : 原始等时面嵌套列表 [n_TP][Time]
        IsoMapP2TP  : 新计算的 P2TP 段 [m_Agents][Time_segment]
        E2TP_TPIdx  : 代理对应的 TP 索引 (m_Agents, )
        PathP2TP    : 路径数据，用于计算长度偏移
    """

    # 定义辅助函数处理 pathid 偏移
    def dealWithPathId(x, P2TPL):
        # 如果 x 为 None 或 pathid 为空，直接返回
        if x is None or x.pathid is None or x.pathid.size == 0:
            return x
        
        # 为了不修改原始 IsoMap_i_tt，创建一个新对象副本
        # 假设 x 是之前定义的 IsoMapData 类实例
        new_x = copy.copy(x) 
        
        # 更新 pathid: [row, col, index + P2TPL]
        # MATLAB: x.pathid(:,3) + P2TPL -> Python: x.pathid[:, 2] + P2TPL
        new_pathid = x.pathid.copy()
        new_pathid[:, 2] += P2TPL
        new_x.pathid = new_pathid
        
        return new_x

    # --- 核心拼接逻辑 (使用嵌套推导式替代循环) ---
    # 外层遍历每个代理 i
    # 逻辑流：
    # 1. 计算当前第一段路径长度 P2TPL
    # 2. 提取并处理原始 IsoMap 的对应行 (IsoTP2Iso)
    # 3. 将新段 (IsoP2TP) 与 处理后的原始段 拼接
    
    IsoMap_i_tt_inserted = [
        # 第一部分：新计算的 P2TP 段列表
        list(IsoMapP2TP[i]) + 
        # 第二部分：处理后的原始段（对应 TP 索引行的所有时间点）
        [
            dealWithPathId(x, PathP2TP[i].shape[1]) 
            for x in IsoMap_i_tt[int(E2TP_TPIdx[i])]
        ]
        for i in range(len(IsoMapP2TP))
    ]

    return IsoMap_i_tt_inserted
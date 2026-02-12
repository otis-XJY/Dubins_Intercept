import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调用转换后的函数
import pickle
import argparse
import colorsys
from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainMapP,WH_main_obtainMapRef,WH_main_obtainIso
from intercept.Prediction.analyzeEvaderIntent import analyzeEvaderIntent
from intercept.IsoPair.obtainIsoPairsAll import obtainIsoPairs,obtainPTP2TP_IsoPos_timeShift2
from intercept.IsoPair.obtainTaskAll import obtainTask,obtainTask_timeShift2
from Draw.DrawIntercept import draw_candidates
from Draw.Draw_map import Draw_map
from intercept.Prediction.obtainDWAprePath import obtainDWAprePath
from intercept.Prediction.predictLikelyTarget import predictLikelyTarget
from intercept.IsoPair.obtainNearTPall import obtainNearETP,obtainNearTP
from intercept.IsoPair.obtainPE2TP import obtainPE2TP
from intercept.IsoPair.obtainIsoPath import insertIsoMapP2TP
# parser = argparse.ArgumentParser()
# parser.add_argument('--nodraw', action='store_true', help='如果指定则不进行绘图，直接返回结果')
# args = parser.parse_args()

TimeMap='_0206_2010'
TimeIso=TimeMap


Map = joblib.load('Map'+TimeMap+'.jbl')	

IsoMapP2TP_i_tt = joblib.load('IsoMapP_i_tt'+TimeMap+'.jbl')	    
pathFinalP2TP = joblib.load('pathFinalP'+TimeMap+'.jbl')	

IsoMapPTP2Iso_i_tt = joblib.load('IsoMapPTP2Iso_i_tt'+TimeMap+'.jbl')	  
IsoMapPIso2TP_i_tt=joblib.load('IsoMapPIso2TP_i_tt'+TimeMap+'.jbl')
pathFinalMapPTP2Iso = joblib.load('pathFinalMapPTP2Iso'+TimeMap+'.jbl')

IsoMapE2ValIn_i_tt = joblib.load('IsoMapE2ValIn_i_tt'+TimeIso+'.jbl')	
pathFinalE2ValIn = joblib.load('pathFinalE2ValIn'+TimeIso+'.jbl')	
PathE2Val_true=joblib.load('PathE2Val_true'+TimeIso+'.jbl')

IsoMapTP2Val_i_tt = joblib.load('IsoMapTP2Val_i_tt'+TimeMap+'.jbl')
pathFinalTP2Val = joblib.load('pathFinalTP2Val'+TimeMap+'.jbl')

IsoMapETP2Iso_i_tt=joblib.load('IsoMapETP2Iso_i_tt'+TimeMap+'.jbl')
IsoMapEIso2TP_i_tt=joblib.load('IsoMapEIso2TP_i_tt'+TimeMap+'.jbl')
pathFinalMapETP2Iso=joblib.load('pathFinalMapETP2Iso'+TimeMap+'.jbl')

# 1. 图形窗口清理
# MATLAB: clf, close all
# plt.close('all')

# 2. 从 Map 中提取变量 (假设 Map 是一个字典)
# 严格保留变量名
obs = Map['obs']
sure = Map['sure']
r = Map['r']
obs_no_circle = Map['obs_no_circle']
obs_no_circle_in = Map['obs_no_circle_in']
outline_all = Map['outline_all']
Stepsize = Map['Stepsize']
resolution = Map['resolution']
v_P = Map['v_P']  # 注意：MATLAB中此处赋值给了 v_PASD
v_E = Map['v_E']
Trans_Point = Map['Trans_Point']
ValuePos = Map['ValuePos']
PStart_Point=Map['PStart_Point']
timeIsoRes=Map['timeIsoRes']
# 3. 定义 Evader 矩阵
Evader = np.array([
    [200, 1950, -np.pi/2],
    [1000, 1950, -np.pi/2],
    [1800, 1950, -np.pi/2]
])

# 4. 初始化 E_PreRef 结构体 (在 Python 中使用字典)
E_PreRef = {
    'num_v': 10,
    'num_w': 10,
    'v_range': [100, 150],
    'w_range': [-np.pi/6, np.pi/6],
    'Stepsize': Map['Stepsize'],
    'T_pred': 1
}

import numpy as np
from scipy.optimize import linear_sum_assignment # 替代 matchpairs

# --- 基础参数初始化 ---
CapDist = 100
PosE = Evader # 假设 Evader 已定义
PosP = PStart_Point # 假设 PStart_Point 已定义

# --- 1. 数据提取 (替代 cellfun) ---
# 使用列表推导式处理尺寸不一的结构体
ETP2Val_IsoPos = [[item.IsoPos for item in row] for row in IsoMapE2ValIn_i_tt]
ETP2Val_pathid = [[item.pathid for item in row] for row in IsoMapE2ValIn_i_tt]
PTP2TP_IsoPos = [[item.IsoPos for item in row] for row in IsoMapP2TP_i_tt]

# 获取维度信息
num_e = len(ETP2Val_IsoPos)
num_t1 = len(ETP2Val_IsoPos[0])
num_p = len(PTP2TP_IsoPos)
num_t2 = len(PTP2TP_IsoPos[0])

# --- 2. 创建索引网格 (替代 ndgrid) ---
# indexing='ij' 确保与 MATLAB ndgrid 的行为一致
eid_idx, pid_idx, te_idx, tp_idx = np.meshgrid(
    np.arange(num_e), np.arange(num_p), np.arange(num_t1), np.arange(num_t2), 
    indexing='ij'
)

# --- 3. 筛选有效组合 (向量化逻辑) ---
# te 从 tp 开始，剔除 te < tp 的情况
valid_idx = te_idx >= tp_idx

pid_flat = pid_idx[valid_idx]
eid_flat = eid_idx[valid_idx]
tp_flat = tp_idx[valid_idx]
te_flat = te_idx[valid_idx]

# --- 4. 意图分析 ---
Valid, threatMatrix = analyzeEvaderIntent(ValuePos, Evader)
# pairsE2Val = [(1:numel(Valid))' Valid]
pairsE2Val = np.column_stack((np.arange(len(Valid)), Valid))

# --- 5. 获取 IsoPairs (替代 arrayfun) ---
# 虽然是列表推导式，但在 Python 内部比显式 for 循环更优化
results = [
    [
        te, tp, eid, pid,
        obtainIsoPairs(
            ETP2Val_IsoPos[eid][te], PTP2TP_IsoPos[pid][tp], 
            PosE[eid, :], PosP[pid, :], CapDist, ValuePos, 
            pairsE2Val, eid, ETP2Val_pathid[eid][te]
        ),
        eid, pid
    ]
    for eid, pid, te, tp in zip(eid_flat, pid_flat, te_flat, tp_flat)
]

# --- 6. 展平、过滤与排序 ---
# results 是一个 list of lists，其中第 5 个元素（索引4）是 obtainIsoPairs 的返回值
# 过滤非空项 (r[4] 对应 MATLAB 的 results{:,5})
flat_results = [r for r in results if r[4] is not None and len(r[4]) > 0]

# 转换为 object 数组便于后续操作，并按第 1 列 (te) 排序

IsoPairs_time_Eid_Pid_Posid_ = np.array(flat_results, dtype=object)
sort_idx = np.argsort(IsoPairs_time_Eid_Pid_Posid_[:, 0].astype(float))
IsoPairs_time_Eid_Pid_Posid = IsoPairs_time_Eid_Pid_Posid_[sort_idx]


# --- 7. 任务获取与初步筛选 ---
InterceptCandidates = obtainTask(IsoPairs_time_Eid_Pid_Posid, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt)
IC = InterceptCandidates.copy()

# 排序并取前 100 个用于绘图
IC_Plot = IC[np.argsort(IC[:, 8])][:100]

# ax1=plt.figure(figsize=(10, 8))
# draw_candidates(IC_Plot, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt, pathFinalE2ValIn, pathFinalP2TP)
# Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)
# plt.show()

###########################################################转化从这开始

# 假设 InterceptCandidates 是一个 NumPy 矩阵
# --- 1. 基础数据提取 (0-based 索引) ---
EidAll = InterceptCandidates[:, 2]  # MATLAB 3
PidAll = InterceptCandidates[:, 3]  # MATLAB 4
cost = InterceptCandidates[:, 8]    # MATLAB 9

# --- 2. 向量化寻找 (Eid, Pid) 的最小 cost 对应的行索引 ---
# 使用 lexsort 进行分组取最小值：主键 Pid(idx 3)，次键 Eid(idx 2)，末键 cost(idx 8)
# lexsort 顺序为由次到主，故传入 (cost, EidAll, PidAll)
# 这能确保在 (Eid, Pid) 相同的行中，cost 最小的排在前面
sort_indices = np.lexsort((cost, EidAll, PidAll))
IC_sorted = InterceptCandidates[sort_indices]

# 找到唯一的 (Eid, Pid) 组合及其在排序后矩阵中第一次出现的位置
# unique 的 axis=0 配合 return_index=True 是处理“行去重”的高级方案
_, first_occur_indices = np.unique(IC_sorted[:, [2, 3]], axis=0, return_index=True)

# 核心技巧：锁定原始矩阵中每个 (Eid, Pid) 对应的最小 cost 的那一行
best_row_indices = sort_indices[first_occur_indices]

# 提取这些最优行的数据
Eid_best = InterceptCandidates[best_row_indices, 2]
Pid_best = InterceptCandidates[best_row_indices, 3]
cost_best = InterceptCandidates[best_row_indices, 8]

# --- 3. 构建 Cost 矩阵与索引映射矩阵 ---
# 映射 ID 到 0 ~ N-1 连续索引
E_set, E_map = np.unique(Eid_best, return_inverse=True)
P_set, P_map = np.unique(Pid_best, return_inverse=True)
nE, nP = len(E_set), len(P_set)

# 初始化代价矩阵和“行索引查找矩阵”
CostMat = np.full((nE, nP), 1e9)
RowIdxMat = np.zeros((nE, nP), dtype=int)

# 向量化填充：完全替代 MATLAB 的 accumarray
CostMat[E_map, P_map] = cost_best
RowIdxMat[E_map, P_map] = best_row_indices

# --- 4. 执行任务指派 ---
# linear_sum_assignment 寻找全局最小代价
e_indices, p_indices = linear_sum_assignment(CostMat)

# 过滤掉代价过高 (1e9) 的无效指派
valid_mask = CostMat[e_indices, p_indices] < 1e9
e_indices = e_indices[valid_mask]
p_indices = p_indices[valid_mask]

# --- 5. 快速提取结果 (完全消除循环) ---
# 利用 RowIdxMat 直接通过坐标映射回 InterceptCandidates 的原始行号
final_idx = RowIdxMat[e_indices, p_indices]

# 得到最终分配结果
ICFinal = InterceptCandidates[final_idx, :]
AssignedIntercepts = ICFinal.copy()  # 保持变量名一致

# 映射回原始 ID
pairs_realE2P = np.column_stack((E_set[e_indices], P_set[p_indices]))

# --- 6. 最终结果拼接 ---
# 提取第 16-18 列 (索引 15:18) 并拼接
extractedData = AssignedIntercepts[:, 15:18]
pairs_CostE2P = np.column_stack((pairs_realE2P, extractedData))
######################
# plt.figure(figsize=(10, 8))
# draw_candidates(ICFinal, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt, pathFinalE2ValIn, pathFinalP2TP)
# Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)
# plt.show()

# --- 1. 基础状态初始化 ---
PosE = Evader.copy()
PosP = PStart_Point.copy()

# 计算当前配对的距离
# pairs_realE2P: [Eid, Pid] (1-based IDs)
# 转换为 0-based 索引进行切片


# 向量化计算欧氏距离
distances = np.linalg.norm(
        PosP[pairs_realE2P[:, 1].astype(int), :2] - 
        PosE[pairs_realE2P[:, 0].astype(int), :2], 
        axis=1
    )
# --- 2. 转换 PathPtrue (替代 mat2cell) ---
# MATLAB: PathPtrue = mat2cell(PosP', size(PosP,2), ones(1,size(PosP,1)))'
# 意图：将 PosP 的每一行提取出来作为一个独立的数组存入列表，且保持为列向量(2x1 或 3x1)
PathPtrue = [p.reshape(-1, 1) for p in PosP]

# 时间参数
TimeRes = 0.01
t = 0
t_all = 0

# --- 3. 提取 Pursuer 路径信息 (替代 arrayfun) ---
# ICFinal 列定义对照 Python (MATLAB index - 1):
# 12: Pid, 10: PTPid, 14: PIsoid
PidAll = ICFinal[:, 12].astype(int)
PTPidAll = ICFinal[:, 10].astype(int)
PIsoidAll = ICFinal[:, 14].astype(int)

# 提取 PathP: pathFinalP2TP{Pid, TPid}(:, 1:Isoid)
# 使用列表推导式配合 zip 替代 arrayfun，处理非规整切片
PathP = [
    pathFinalP2TP[pid][tpid][:, :isoid] 
    for pid, tpid, isoid in zip(PidAll, PTPidAll, PIsoidAll)
]

# --- 4. 提取 Evader 路径信息 ---
EfromTPid = np.zeros(PosE.shape[0])
# 11: Eid, 9: ValuePosid, 13: EIsoid
EidAll = ICFinal[:, 11].astype(int)
ValuePosid = ICFinal[:, 9].astype(int)
EIsoidAll = ICFinal[:, 13].astype(int)

# 提取 PathE: pathFinalE2ValIn{Eid, VPid}(:, 1:Isoid)
PathE = [
    pathFinalE2ValIn[eid][vpid][:, :isoid]
    for eid, vpid, isoid in zip(EidAll, ValuePosid, EIsoidAll)
]

# --- 5. 准备绘图与循环环境 ---

# 初始化绘图
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))

# 辅助变量初始化
t = 0
t_all = 0
# 假设 steps_size, v_P, v_E, timeIsoRes 等变量已定义

while not np.all(distances < CapDist):
    # --- 1. 时间更新 ---
    t += TimeRes / Stepsize
    t_all += TimeRes / Stepsize

    # --- 2. 确定未拦截的 E 和 P ---
    Capflag = distances < CapDist
    # pairs_realE2P: [Eid, Pid] (注意保持 ID 对应关系)
    UnCapPid = pairs_realE2P[~Capflag, 1].astype(int)
    UnCapEid = pairs_realE2P[~Capflag, 0].astype(int)
    # 假设在进入循环前已初始化这些变量为 None
    # IsoMapPTP2Iso_i_tt_timeShift = None 
    # IsoMapETP2Val_i_tt_timeShift = None

    # --- 1. 动态维护 IsoMap 缓存 (TimeShift 逻辑) ---
    # exist(var, 'var') 在 Python 中通常通过检查变量是否为 None 实现
    if 'IsoMapPTP2Iso_i_tt_timeShift' in locals() and IsoMapPTP2Iso_i_tt_timeShift is not None:
        # 获取当前 TimeShift 缓存的维度
        # MATLAB: [row, col] = size(...)
        # Python: 假设是嵌套列表 [row][col]
        num_rows_shift = len(IsoMapPTP2Iso_i_tt_timeShift)
        num_cols_shift = len(IsoMapPTP2Iso_i_tt_timeShift[0]) if num_rows_shift > 0 else 0
        
        # 获取原始 IsoMap 的列数 (时间维度长度)
        num_cols_orig = len(IsoMapP2TP_i_tt[0])

        if num_cols_shift != num_cols_orig:
            # 如果当前缓存行数与未捕获的 Pursuer 数量不符
            if num_rows_shift != len(UnCapPid):
                # 按照 ID 顺序 (1, 2, 3...) 过滤缓存
                # np.sort(UnCapPid) - 1 转换为 0-based 索引
                sort_pid_idx = (np.sort(UnCapPid)).astype(int)
                IsoMapPTP2Iso_i_tt_timeShift = [IsoMapPTP2Iso_i_tt_timeShift[i] for i in sort_pid_idx]

            # 如果当前缓存行数与未捕获的 Evader 数量不符
            if len(IsoMapETP2Val_i_tt_timeShift) != len(UnCapEid):
                # 注意 MATLAB 原代码此处使用了 UnCapPid 过滤 E，需根据实际逻辑确认是否一致
                # 如果是误写，Python 建议改为 np.sort(UnCapEid)
                sort_eid_idx = (np.sort(UnCapEid)).astype(int)
                IsoMapETP2Val_i_tt_timeShift = [IsoMapETP2Val_i_tt_timeShift[i] for i in sort_eid_idx]
        else:
            # 维度匹配，则直接从原始全量 Map 初始化（每一行作为一个单元）
            # 对应 MATLAB: num2cell(..., 2)'
            IsoMapPTP2Iso_i_tt_timeShift = [list(row) for row in IsoMapP2TP_i_tt]
            IsoMapETP2Val_i_tt_timeShift = [list(row) for row in IsoMapE2ValIn_i_tt]
    else:
        # 如果不存在缓存，则进行初始化
        IsoMapPTP2Iso_i_tt_timeShift = [list(row) for row in IsoMapP2TP_i_tt]
        IsoMapETP2Val_i_tt_timeShift = [list(row) for row in IsoMapE2ValIn_i_tt]

    # --- 2. 动态过滤任务候选表 (ICFinal 逻辑) ---
    # 如果任务表行数与未捕获的 Evader 数量不一致
    if len(ICFinal) != len(UnCapEid):
        # 按照 Evader ID 顺序提取对应的分配任务
        sort_idx_ic = (np.sort(UnCapEid)).astype(int)
        # 使用 NumPy 的高级索引一次性提取所有行
        ICFinal = ICFinal[sort_idx_ic, :]

############################################################从这转化

    # --- 1. 计算当前时间步在路径中的索引 ---
    # MATLAB: t*v_P 对应 Python 索引 (注意 Python 0-based)
    curr_idx_p = int(round(t * v_P))
    # MATLAB: (t-dt)*v_P + 1 对应 Python 起点索引
    seg_start = int(round((t - TimeRes / Stepsize) * v_P))
    seg_end = curr_idx_p 

    # --- 2. 提取追捕者当前时刻位置 P_pos (替代第一个 cellfun) ---
    # PathP 是 list of arrays (2 x N)，提取特定列
    # 使用 min(..., x.shape[1]-1) 防止越界
    P_pos_list = [
        path[:, min(curr_idx_p, path.shape[1])] 
        for path in PathP
    ]

    # --- 3. 更新追捕者真实轨迹 PathPtrue (基于 UnCapPid) ---
    # 获取排序后的未捕获追捕者索引 (0-based)
    # MATLAB: sort(UnCapPid)
    p_sort_idx = (np.sort(UnCapPid)).astype(int)

    # 向量化追加路段：仅针对未捕获的追捕者更新
    # Python 切片 [start:end] 是左闭右开，刚好对应 MATLAB 的 (start+1):end
    for idx in p_sort_idx:
        # 提取规划路径中的当前步片段
        segment = PathP[idx][:, max(0, seg_start) : seg_end]
        # np.hstack 实现水平拼接 [pathtrue, segment]
        PathPtrue[idx] = np.hstack((PathPtrue[idx], segment))

    # --- 4. 整理并排序当前追捕者位置 PosP ---
    # 将列表转换为矩阵并按 ID 排序
    temp_PosP = np.array(P_pos_list) # (N, 2)
    PosP = temp_PosP[p_sort_idx, :]

    # --- 5. 提取并排序逃避者当前位置 PosE ---
    # 假设 PathE2Val_true 是 list of arrays (Time x 2 或 2 x Time)
    # 根据 MATLAB 代码 x(t_all*v_P, :)，假设是 (Time x 2)
    curr_idx_e = int(round(t_all * v_P))
    e_sort_idx = (np.sort(UnCapEid)).astype(int)

    # 提取并排序 (替代最后一个 cellfun + cell2mat + sort)
    PosE = np.array([
        PathE2Val_true[i][min(curr_idx_e, PathE2Val_true[i].shape[0] - 1), :]
        for i in e_sort_idx
    ])
#########################################################
    # 注：如果 PathE2Val_true 的结构是 (2 x Time)，则改为：
    # PathE2Val_true[i][:, min(...)]
    # --- 4. 攻击意图判断 (DWA) ---
    # 调用 DWA 子函数
    min_dists, BestPaths = obtainDWAprePath(PosE, [PathE[i] for i in UnCapEid], E_PreRef)

    flagIn = 1 if np.all(min_dists <= CapDist / 5) else 0

    # --- 5. 重新规划逻辑 ---
    if not flagIn:
        # 轨迹预测
        traj = [PathE2Val_true[i][:int(t_all * v_P), :] for i in UnCapEid]
        targets = ValuePos[:, :2]
        res = predictLikelyTarget(traj, targets, UseWindow=3*v_P, Method='poly')
        Validnew = res['rank_idx'][:, 0]
        pairsE2Val = np.column_stack((np.arange(len(Validnew)), Validnew))

        # 检查是否需要触发重规划
        NearTPpos_E, NearTPid_E = obtainNearETP(Trans_Point[:, :2], PosE, ValuePos, pairsE2Val)
        
        if not np.array_equal(NearTPid_E, EfromTPid) or not np.array_equal(Validnew, Valid):
            Valid = Validnew
            EfromTPid = NearTPid_E
##############################################从这里开始转化
            # --- 1. 获得 E2TP (逃避者到转移点路径) ---
            # 调用之前转换好的 obtainPE2TP
            PathE2TP, IsoMapE2TP, E2TP_results = obtainPE2TP(
                IsoMapEIso2TP_i_tt, EfromTPid, PosE, Trans_Point, pathFinalMapETP2Iso, Map, Map['v_E']
            )

            # 计算时间索引 (替代 cellfun)
            TimeE2TP = np.array([p.shape[1] / v_E * Stepsize for p in PathE2TP])
            TimeidE2TP = np.ceil(TimeE2TP / timeIsoRes).astype(int)

            # 提取 TP 索引：MATLAB 逻辑是 x(:,2) 后取 x{1}，在 Python 结果列表中对应索引 1
            E2TP_TPIdx = np.array([res[1] for res in E2TP_results])

            # --- 2. 获得 P2TP (追捕者到转移点路径) ---
            # 计算最近 TP
            NearTPpos_P, NearTPid_P = obtainNearTP(Trans_Point[:, :2], PosP, PosE)
            # 计算路径
            PathP2TP, IsoMapP2TP, P2TP_results = obtainPE2TP(
                IsoMapPIso2TP_i_tt, NearTPid_P, PosP, Trans_Point, pathFinalMapPTP2Iso, Map, Map['v_P']
            )

            TimeP2TP = np.array([p.shape[1] / v_P * Stepsize for p in PathP2TP])
            TimeidP2TP = np.ceil(TimeP2TP / timeIsoRes).astype(int)
            P2TP_TPIdx = np.array([res[1] for res in P2TP_results])

            # --- 3. 动态合并等时面 (insertIsoMapP2TP) ---
            IsoMap_i_tt_P2Iso_raw = insertIsoMapP2TP(IsoMapPTP2Iso_i_tt, IsoMapP2TP, P2TP_TPIdx, PathP2TP)
            IsoMap_i_tt_E2Iso_raw = insertIsoMapP2TP(IsoMapTP2Val_i_tt, IsoMapE2TP, E2TP_TPIdx, PathE2TP)

            # --- 4. 补齐等时面长度 (Padding) ---
            # 计算每个代理的最大时间长度
            TimeP_len = [len(row) for row in IsoMap_i_tt_P2Iso_raw]
            TimeE_len = [len(row) for row in IsoMap_i_tt_E2Iso_raw]
            TimeL = max(max(TimeP_len), max(TimeE_len))

            # 列表推导式快速补齐 None (替代 repmat + cellfun)
            IsoMap_i_tt_P2Iso = [row + [None] * (TimeL - len(row)) for row in IsoMap_i_tt_P2Iso_raw]
            IsoMap_i_tt_E2Iso = [row + [None] * (TimeL - len(row)) for row in IsoMap_i_tt_E2Iso_raw]

            # --- 5. 创建 4D 索引网格与筛选 (替代 ndgrid) ---
            num_e = len(IsoMap_i_tt_E2Iso)
            num_p = len(IsoMap_i_tt_P2Iso)

            # indexing='ij' 确保生成逻辑与 MATLAB ndgrid (e, p, te, tp) 完全一致
            eid_idx, pid_idx, te_idx, tp_idx = np.meshgrid(
                np.arange(num_e), np.arange(num_p), np.arange(TimeL), np.arange(TimeL), 
                indexing='ij'
            )

            # 筛选 te >= tp 的有效组合 (向量化掩码)
            valid_mask = te_idx >= tp_idx
            eid_flat = eid_idx[valid_mask]
            pid_flat = pid_idx[valid_mask]
            te_flat = te_idx[valid_mask] # 转为 1-based 供子函数使用
            tp_flat = tp_idx[valid_mask]

            # --- 6. 执行拦截对搜索 (替代 arrayfun) ---
            # 使用 zip 并行迭代，这是 Python 中替代多参数 arrayfun 的最高性能方案
            results = [
                [
                    te, tp, E2TP_TPIdx[eid], P2TP_TPIdx[pid],
                    obtainPTP2TP_IsoPos_timeShift2(
                        pid, eid, tp, te, PosP, PosE, CapDist, 
                        IsoMap_i_tt_P2Iso, IsoMap_i_tt_E2Iso, ValuePos, pairsE2Val
                    ),
                    eid, pid
                ]
                for eid, pid, te, tp in zip(eid_flat, pid_flat, te_flat, tp_flat)
            ]

            # --- 7. 过滤并排序结果 ---
            # 过滤掉第五个元素（拦截对信息）为空的项
            flat_results = [r for r in results if r[4] is not None and len(r[4]) > 0]


                # 转换为 object 数组并按第一列 (te) 排序
            IsoPairs_time_ETPid_PTPid_Posid_ = np.array(flat_results, dtype=object)
            sort_idx = np.argsort(IsoPairs_time_ETPid_PTPid_Posid_[:, 0].astype(int))
            IsoPairs_time_ETPid_PTPid_Posid = IsoPairs_time_ETPid_PTPid_Posid_[sort_idx]


            # --- 8. 生成任务候选表 ---
            InterceptCandidates = obtainTask_timeShift2(
                IsoPairs_time_ETPid_PTPid_Posid, IsoMap_i_tt_P2Iso, IsoMap_i_tt_E2Iso
            )

            IC = InterceptCandidates.copy()
            
            # --- 任务指派 (完全向量化替代 accumarray/matchpairs) ---

            # --- 1. & 2. 寻找每个 (Pid, Eid) 组合中 cost 最小的行索引 ---
            # MATLAB: sortrows(IC, [13, 12, 9]) -> 对应 Python 索引 [12, 11, 8]
            # lexsort 键顺序为从后往前：cost(8), Eid(11), Pid(12)
            sIdx = np.lexsort((IC[:, 8], IC[:, 11], IC[:, 12]))
            IC_sorted = IC[sIdx]

            # 对 (Pid, Eid) 进行唯一性筛选，保留 cost 最小的行 (stable)
            # unique 的 axis=0 配合 return_index 锁定排序后矩阵中的位置
            _, firstOccurIdx = np.unique(IC_sorted[:, [12, 11]], axis=0, return_index=True)

            # 映射回原始 IC 的行索引
            bestRowIndices = sIdx[firstOccurIdx]

            # 提取最优行对应的 Eid, Pid 和 cost
            Eid_best = IC[bestRowIndices, 11]
            Pid_best = IC[bestRowIndices, 12]
            cost_best = IC[bestRowIndices, 8]

            # --- 3. 构建用于指派的 Cost 矩阵 (替代 accumarray) ---
            # 映射 ID 到 0~N 连续索引
            E_set, E_map = np.unique(Eid_best, return_inverse=True)
            P_set, P_map = np.unique(Pid_best, return_inverse=True)
            nE, nP = len(E_set), len(P_set)

            # 初始化 CostMat 和索引查找表
            CostMat = np.full((nP, nE), 1e9)
            RowIdxMat = np.zeros((nP, nE), dtype=int)

            # 向量化填充 (由于 bestRowIndices 已预选，每个槽位仅一个值)
            CostMat[P_map, E_map] = cost_best
            RowIdxMat[P_map, E_map] = bestRowIndices

            # --- 4. 执行任务分配 (替代 matchpairs) ---
            # linear_sum_assignment 寻找最小权重匹配
            p_indices, e_indices = linear_sum_assignment(CostMat)

            # 过滤超过阈值 (1e8) 的无效分配
            mask = CostMat[p_indices, e_indices] < 1e8
            p_indices, e_indices = p_indices[mask], e_indices[mask]

            # 映射回原始 ID 得到 [Eid, Pid] 配对
            pairs_realE2P = np.column_stack((E_set[e_indices], P_set[p_indices]))

            # --- 5. 提取最终结果 (完全向量化) ---
            # 利用 RowIdxMat 查找表直接获取 IC 的行索引
            final_idx = RowIdxMat[p_indices, e_indices]

            # 得到最终分配结果
            AssignedIntercepts = IC[final_idx, :]
            ICFinal = AssignedIntercepts.copy()  # 保持变量名一致

            # 更新 TimeShift 缓存列表 (List Comprehension 处理对象)
            IsoMapPTP2Iso_i_tt_timeShift = [IsoMap_i_tt_P2Iso[i] for i in p_indices]
            IsoMapETP2Val_i_tt_timeShift = [IsoMap_i_tt_E2Iso[i] for i in e_indices]

            # --- 6. 结果拼接 (优化：无需 ismember，数据已天然对齐) ---
            # MATLAB: extractedData = AssignedIntercepts(Locb(Lia), 16:18)
            # Python: 提取第 16-18 列 (索引 15:18)
            extractedData = AssignedIntercepts[:, 15:18]

            # 最终拼接结果 [Eid, Pid, data1, data2, data3]
            pairs_CostE2P_ = np.column_stack((pairs_realE2P, extractedData))

            # 纯转移#############################################纯转移

            # --- 1. Evader 路径处理 (全推导式实现) ---
            # 提取元数据并转换为 0-based 索引
            E_idx = ICFinal[:, 11].astype(int)      # EidAns
            VP_idx = ICFinal[:, 9].astype(int)      # ValuePosid
            Iso_idx = ICFinal[:, 13].astype(int)        # EIsoidAns
            ETP_idx = ICFinal[:, 2].astype(int)    # EfromTPid

            # 使用带有条件判断的列表推导式替代 if/else 和 arrayfun
            # 逻辑：拼接 PathE2TP 和 pathFinalTP2Val，或者直接对 PathE2TP 进行切片
            PathE_active = [
                np.hstack((
                    PathE2TP[i], 
                    pathFinalTP2Val[ETP_idx[i]][VP_idx[i]][:, :int(Iso_idx[i] - PathE2TP[i].shape[1])]
                )) if VP_idx[i] >= 0 else PathE2TP[i][:, :int(Iso_idx[i])]
                for i in range(len(E_idx))
            ]

            # 全局重排 (替代 PathE_(EidAns)=PathE_)
            # 通过 argsort 找到 Eid 从小到大的顺序，一次性重构全局列表
            PathE = [PathE_active[i] for i in np.argsort(E_idx)]


            # --- 2. Pursuer 路径处理 (全推导式实现) ---
            # 提取元数据
            P_idx = ICFinal[:, 12].astype(int)     # PidAns
            TP_to_idx = ICFinal[:, 10].astype(int)  # PtoTPid
            Iso_p_idx = ICFinal[:, 14].astype(int)      # PIosidAns
            TP_from_idx = ICFinal[:, 3].astype(int) # PfromTPid

            # 同理处理 Pursuer 路径
            PathP_active = [
                np.hstack((
                    PathP2TP[i], 
                    pathFinalMapPTP2Iso[TP_from_idx[i]][TP_to_idx[i]][:, :int(Iso_p_idx[i] - PathP2TP[i].shape[1])]
                )) if TP_to_idx[i] >= 0 else PathP2TP[i][:, :int(Iso_p_idx[i])]
                for i in range(len(P_idx))
            ]

            # 全局重排 (替代 PathP_(PidAns)=PathP_)
            PathP = [PathP_active[i] for i in np.argsort(P_idx)]

            # 重置时间
            t = 0
    # --- 6. 绘图 (交互式更新) ---#

    # --- 1. 清理与基础地图绘制 ---
# --- 1. 清理与基础地图绘制 ---
    ax.cla()
    Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in, ax)

    # --- 2. 准备颜色集 (使用 tab20 提供更多对比度) ---
    numIntercepts = len(UnCapEid)
    # tab20 每两个颜色为一组同色系，适合表示 Pursuer-Evader 对
    color_map = plt.cm.get_cmap('tab20') 

    # --- 3. 向量化绘制全局位置点 (作为背景) ---
    ax.scatter(Evader[:, 0], Evader[:, 1], c='gray', marker='d', s=20, alpha=0.5, label='Evader Start')
    
    # --- 4. 遍历每个拦截任务绘图 ---
    for idx, i in enumerate(UnCapEid):
        # 为每组任务分配 tab20 中的一对颜色
        # color_p (追捕者): 深色； color_e (逃避者): 浅色
        color_p = color_map((idx * 2) % 20)
        color_e = color_map((idx * 2 + 1) % 20)
        
        # 查找分配表索引
        pid_arr = np.where(ICFinal[:, 12] == i)[0]
        if len(pid_arr) == 0: continue
        pid = pid_arr[0]

        # --- 提取拦截点数据 ---
        tp_time_idx = int(ICFinal[pid, 1])
        te_time_idx = int(ICFinal[pid, 0])
        iso_idx_p = int(ICFinal[pid, 5])
        iso_idx_e = int(ICFinal[pid, 4])

        # 追捕者与逃避者预测拦截点
        s_p = IsoMapPTP2Iso_i_tt_timeShift[pid][tp_time_idx]
        intercept_p_pos = s_p.IsoPos[:, iso_idx_p]
        intercept_p_theta = s_p.IsoVtheta[iso_idx_p]

        s_e = IsoMapETP2Val_i_tt_timeShift[pid][te_time_idx]
        intercept_e_pos = s_e.IsoPos[:, iso_idx_e]
        intercept_e_theta = s_e.IsoVtheta[iso_idx_e]

        # --- 绘制当前实时点与方向 (向量颜色跟随对象) ---
        ax.plot(PosP[i, 0], PosP[i, 1], 'o', color=color_p, markersize=8, markeredgecolor='w')
        ax.quiver(PosP[i, 0], PosP[i, 1], 150*np.cos(PosP[i, 2]), 150*np.sin(PosP[i, 2]), 
                color=color_p, angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.8)

        ax.plot(PosE[i, 0], PosE[i, 1], 'd', color=color_e, markersize=8, markeredgecolor='w')
        ax.quiver(PosE[i, 0], PosE[i, 1], 150*np.cos(PosE[i, 2]), 150*np.sin(PosE[i, 2]), 
                color=color_e, angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.8)

        # --- 绘制拦截预测点 (增加白色边框突出显示) ---
        ax.plot(intercept_p_pos[0], intercept_p_pos[1], '^', color=color_p, markersize=5, markeredgecolor='k', label=f'P-{i} Intercept')
        ax.plot(intercept_e_pos[0], intercept_e_pos[1], 's', color=color_e, markersize=5, markeredgecolor='k')

        # --- 绘制预测轨迹 (虚线表示预测) ---
        ax.plot(PathP[i][0, :], PathP[i][1, :], '--', color=color_p, linewidth=1, alpha=0.6)
        ax.plot(PathE[i][0, :], PathE[i][1, :], '--', color=color_e, linewidth=1, alpha=0.6)

        ax.plot(PathP2TP[i][0, :], PathP2TP[i][1, :], '--', color='black', linewidth=1, alpha=0.6)
        ax.plot(PathE2TP[i][0, :], PathE2TP[i][1, :], '--', color='black', linewidth=1, alpha=0.6)

        # ax.plot(PathP[i][0, :], PathP[i][1, :], '--', color=color_p, linewidth=1, alpha=0.6)
        # ax.plot(PathE[i][0, :], PathE[i][1, :], '--', color=color_e, linewidth=1, alpha=0.6)
         
        # --- 绘制真实历史轨迹 (加粗并稍微调淡灰色) ---
        len_e_true = int(t_all * v_E)
        ax.plot(PathE2Val_true[i][:len_e_true, 0], PathE2Val_true[i][:len_e_true, 1], 
                '-', color='gray', linewidth=2.5, alpha=0.3, zorder=1)
        
        len_p_true = int(t_all * v_P)
        ax.plot(PathPtrue[i][0, :len_p_true], PathPtrue[i][1, :len_p_true], 
                '-', color='gray', linewidth=2.5, alpha=0.3, zorder=1)

        # --- DWA 最优路径 (使用高亮的金黄色或对应颜色的实线) ---
        ax.plot(BestPaths[i][0, :], BestPaths[i][1, :], '-', color=color_p, linewidth=2.5, zorder=3)

    # 设置图例避免遮挡
    # ax.legend(loc='upper right', fontsize='x-small', ncol=2)
    # --- 5. 刷新画布 ---
    plt.grid(True)
    plt.draw()
    plt.pause(0.01) # 对应 drawnow 和 pause
    
    # 更新距离用于循环判断
    distances = np.linalg.norm(
        PosP[pairs_realE2P[:, 1].astype(int), :2] - 
        PosE[pairs_realE2P[:, 0].astype(int), :2], 
        axis=1
    )
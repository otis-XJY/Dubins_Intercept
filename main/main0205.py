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
TimeIso='_0206_2010'


Map = joblib.load('Map'+TimeMap+'.jbl')	

IsoMapP2TP_i_tt = joblib.load('IsoMapP_i_tt'+TimeMap+'.jbl')	    
pathFinalP2TP = joblib.load('pathFinalP'+TimeMap+'.jbl')	

IsoMapPTP2Iso_i_tt = joblib.load('IsoMapPTP2Iso_i_tt'+TimeMap+'.jbl')	  
pathFinalMapPTP2Iso = joblib.load('pathFinalMapPTP2Iso'+TimeMap+'.jbl')

IsoMapE2ValIn_i_tt = joblib.load('IsoMapE2ValIn_i_tt'+TimeIso+'.jbl')	
pathFinalE2ValIn = joblib.load('pathFinalE2ValIn'+TimeIso+'.jbl')	

IsoMapTP2Val_i_tt = joblib.load('IsoMapTP2Val_i_tt'+TimeIso+'.jbl')
pathFinalTP2Val = joblib.load('pathFinalTP2Val'+TimeIso+'.jbl')

PathE2Val_true=joblib.load('PathE2Val_true'+TimeIso+'.jbl')

IsoMapETP2Iso_i_tt=joblib.load('IsoMapETP2Iso_i_tt'+TimeIso+'.jbl')
pathFinalMapETP2Iso=joblib.load('pathFinalMapETP2Iso'+TimeIso+'.jbl')

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

##################################################
# --- 8. 构建代价矩阵与任务分配 (核心优化点) ---
EidAll = InterceptCandidates[:, 2]
PidAll = InterceptCandidates[:, 3]
cost = InterceptCandidates[:, 8]

E_set = np.unique(EidAll)
P_set = np.unique(PidAll)
nE, nP = len(E_set), len(P_set)

# 快速映射 ID 到索引 (替代 ismember)
E_map = {id_val: i for i, id_val in enumerate(E_set)}
P_map = {id_val: i for i, id_val in enumerate(P_set)}

# 为每个 (Eid, Pid) 对寻找最小 cost 的索引
# 这里使用组合键 (Eid, Pid) 进行分组优化
# 采用 lexsort 确保相同 (E, P) 下 cost 最小的排在前面
sort_keys = np.lexsort((cost, PidAll, EidAll))
IC_sorted = IC[sort_keys]

# 找到唯一的 (Eid, Pid) 组合及其第一次出现（最小 cost）的索引
_, unique_indices = np.unique(IC_sorted[:, [2, 3]].astype(float), axis=0, return_index=True)
best_IC_per_EP = IC_sorted[unique_indices]

# 构建 Cost 矩阵 (初始化为极大值)
CostMat = np.full((nE, nP), 1e9)
# 获取在矩阵中的坐标
row_indices = np.array([E_map[e] for e in best_IC_per_EP[:, 2]])
col_indices = np.array([P_map[p] for p in best_IC_per_EP[:, 3]])
CostMat[row_indices, col_indices] = best_IC_per_EP[:, 8]

# 使用线性指派算法 (替代 matchpairs)
# scipy 的 linear_sum_assignment 默认寻找最小代价
row_ind, col_ind = linear_sum_assignment(CostMat)

# 过滤掉代价过高的无效指派 (对应 matchpairs 的 threshold 参数)
mask = CostMat[row_ind, col_ind] < 1e9
row_ind = row_ind[mask]
col_ind = col_ind[mask]

# 得到最终的分配 ID 对
pairs_realE2P = np.column_stack((E_set[row_ind], P_set[col_ind]))

# --- 9. 提取最终分配结果 ---
# 在 best_IC_per_EP 中根据分配结果提取完整的行数据
final_results_mask = np.array([
    np.where((best_IC_per_EP[:, 2] == e) & (best_IC_per_EP[:, 3] == p))[0][0]
    for e, p in pairs_realE2P
])
AssignedIntercepts = best_IC_per_EP[final_results_mask]
ICFinal = AssignedIntercepts
# # --- 绘图部分 2 (ICFinal 最终分配结果) ---
plt.figure(figsize=(10, 8))
draw_candidates(InterceptCandidates, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt, pathFinalE2ValIn, pathFinalP2TP)
Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)
plt.show()
# --- 10. 拼接结果 pairs_CostE2P ---
# 提取 AssignedIntercepts 的第 16 到 18 列 (Python 索引 15:18)
# 并与 pairs_realE2P 拼接
extractedData = AssignedIntercepts[:, 15:18] 
pairs_CostE2P = np.column_stack((pairs_realE2P, extractedData))
######################


# --- 1. 基础状态初始化 ---
PosE = Evader.copy()
PosP = PStart_Point.copy()

# 计算当前配对的距离
# pairs_realE2P: [Eid, Pid] (1-based IDs)
# 转换为 0-based 索引进行切片
idx_E_dist = pairs_realE2P[:, 0].astype(int)
idx_P_dist = pairs_realE2P[:, 1].astype(int)

# 向量化计算欧氏距离
distances = np.sqrt(np.sum((PosP[idx_P_dist, :2] - PosE[idx_E_dist, :2])**2, axis=1))

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

    # --- 3. 更新位置 (向量化替代 cellfun) ---
    # P_pos: 提取当前时刻 P 的位置
    P_pos_list = [
        path[:, min(int(t * v_P), path.shape[1] - 1)] 
        for path in PathP
    ]
    PosP_full = np.array(P_pos_list) # 此时是 N x 2 或 N x 3
    # 只取未捕获且按 ID 排序的 P
    sorted_UnCapPid_idx = np.argsort(UnCapPid)
    PosP = PosP_full[sorted_UnCapPid_idx]

    # 更新 PathPtrue (历史轨迹)
    for i, pid in enumerate(UnCapPid):
        # 模拟 [pathtrue, x(:, index)]
        seg_start = int((t - TimeRes / Stepsize) * v_P) + 1
        seg_end = int(t * v_P) + 1
        path_seg = PathP[i][:, min(PathP[i].shape[1] - 1, np.arange(seg_start, seg_end))]
        PathPtrue[pid] = np.hstack((PathPtrue[pid], path_seg))

    # 更新 PosE (从预定义的真实路径提取)
    # 假设 PathE2Val_true 是包含所有 E 真实轨迹的 list
    E_pos_list = [
        path[min(int(t_all * v_P), path.shape[0] - 1), :] 
        for path in PathE2Val_true
    ]
    PosE_all = np.array(E_pos_list)
    # 按未捕获 Eid 排序提取
    PosE = PosE_all[np.argsort(UnCapEid) - 1]

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
        pairsE2Val = np.column_stack((np.arange(len(Validnew)) + 1, Validnew))

        # 检查是否需要触发重规划
        NearTPpos_E, NearTPid_E = obtainNearETP(Trans_Point[:, :2], PosE, ValuePos, pairsE2Val)
        
        if not np.array_equal(NearTPid_E, EfromTPid) or not np.array_equal(Validnew, Valid):
            Valid = Validnew
            EfromTPid = NearTPid_E

            # 获取 E2TP 和 P2TP
            PathE2TP, IsoMapE2TP, E2TP_Timeid_TPid_Pid_IsoPosid = obtainPE2TP(IsoMapETP2Iso_i_tt, EfromTPid, PosE, Trans_Point, pathFinalMapETP2Iso, Map, Map.v_P)
            
            E2TP_TPIdx = np.array([res[1] for res in E2TP_Timeid_TPid_Pid_IsoPosid]).flatten()
           
            NearTPpos_P, NearTPid_P = obtainNearTP(Trans_Point[:, :2], PosP, PosE)
            PathP2TP, IsoMapP2TP, P2TP_Timeid_TPid_Pid_IsoPosid = obtainPE2TP(IsoMapPTP2Iso_i_tt, NearTPid_P, PosP, Trans_Point, pathFinalMapPTP2Iso, Map, Map.v_P)

            P2TP_TPIdx = np.array([res[1] for res in P2TP_Timeid_TPid_Pid_IsoPosid]).flatten()

            # 插入并对齐 IsoMap (补齐长度)
            IsoMap_i_tt_P2Iso_ = insertIsoMapP2TP(IsoMapPTP2Iso_i_tt, IsoMapP2TP, P2TP_TPIdx, PathP2TP)
            IsoMap_i_tt_E2Iso_ = insertIsoMapP2TP(IsoMapTP2Val_i_tt, IsoMapE2TP, E2TP_TPIdx, PathE2TP)
            
            # 对齐 IsoMap 长度 (Padding)
            TimeL = max([len(row) for row in IsoMap_i_tt_P2Iso_] + [len(row) for row in IsoMap_i_tt_E2Iso_])
            # Python 列表推导式填充
            IsoMap_i_tt_P2Iso = [row + [None]*(TimeL - len(row)) for row in IsoMap_i_tt_P2Iso_]
            IsoMap_i_tt_E2Iso = [row + [None]*(TimeL - len(row)) for row in IsoMap_i_tt_E2Iso_]

            # 创建 4D 索引网格并筛选有效组合
            eid_grid, pid_grid, te_grid, tp_grid = np.meshgrid(
                np.arange(len(IsoMap_i_tt_E2Iso)), np.arange(len(IsoMap_i_tt_P2Iso)), 
                np.arange(TimeL), np.arange(TimeL), indexing='ij'
            )
            v_mask = te_grid >= tp_grid
            
            # 调用 obtainPTP2TP_IsoPos_timeShift2 (通过列表推导式处理)
            # 这里是计算密集区，results 的生成使用列表推导
            results = [
                [te, tp, E2TP_TPIdx[e], P2TP_TPIdx[p], 
                 obtainPTP2TP_IsoPos_timeShift2(p, e, tp, te, PosP, PosE, CapDist, IsoMap_i_tt_P2Iso, IsoMap_i_tt_E2Iso, ValuePos, pairsE2Val), 
                 e, p]
                for e, p, te, tp in zip(eid_grid[v_mask], pid_grid[v_mask], te_grid[v_mask], tp_grid[v_mask])
            ]

            # 过滤、排序并获取候选任务
            flat_results = [r for r in results if r[4] is not None and len(r[4]) > 0]
            flat_results.sort(key=lambda x: x[0]) # 按 te 排序
            IsoPairs_time_ETPid_PTPid_Posid = np.array(flat_results, dtype=object)

            InterceptCandidates = obtainTask_timeShift2(IsoPairs_time_ETPid_PTPid_Posid, IsoMap_i_tt_P2Iso, IsoMap_i_tt_E2Iso)

            IC = InterceptCandidates.copy()
            # --- 任务指派 (完全向量化替代 accumarray/matchpairs) ---
            # IC: [..., cost(index 8), Eid(11), Pid(12)] (Python 索引从 0 开始需对应)
            # 使用排序 + unique 寻找每个 (P, E) 的最小 cost
            sort_idx = np.lexsort((IC[:, 8].astype(float), IC[:, 12].astype(float), IC[:, 11].astype(float)))
            IC_s = IC[sort_idx]
            _, first_idx = np.unique(IC_s[:, [11, 12]].astype(float), axis=0, return_index=True)
            best_IC = IC_s[first_idx]

            # 构建指派矩阵
            E_u = np.unique(best_IC[:, 11])
            P_u = np.unique(best_IC[:, 12])
            CostMat = np.full((len(E_u), len(P_u)), 1e9)
            # 建立映射
            e_map = {id: i for i, id in enumerate(E_u)}
            p_map = {id: i for i, id in enumerate(P_u)}

            for row in best_IC:
                CostMat[e_map[row[11]], p_map[row[12]]] = row[8]

            
            row_indices = np.array([e_map[e] for e in best_IC[:, 11]])
            col_indices = np.array([p_map[p] for p in best_IC[:, 12]])
            CostMat[row_indices, col_indices] = best_IC[:, 8]

            # 匈牙利算法
            e_idx, p_idx = linear_sum_assignment(CostMat)
            # 过滤掉代价过高的无效指派 (对应 matchpairs 的 threshold 参数)
            mask = CostMat[e_idx, p_idx] < 1e9
            e_idx = e_idx[mask]
            p_idx = p_idx[mask]
            pairs_realE2P = np.column_stack((E_u[e_idx], P_u[p_idx]))

            ICFinal = np.array([best_IC[(best_IC[:, 11] == E_u[ei]) & (best_IC[:, 12] == P_u[pi])][0] 
                               for ei, pi in zip(e_idx, p_idx)])
            
            # --- 2. 提取对应的等时面数据 (List Comprehension 是处理对象的最佳方式) ---
            # assignments[:, 0] 是 P 的索引，assignments[:, 1] 是 E 的索引
            IsoMapPTP2Iso_i_tt_timeShift = [IsoMap_i_tt_P2Iso[p_idx]]
            IsoMapETP2Val_i_tt_timeShift = [IsoMap_i_tt_E2Iso[p_idx]]

            # --- 3. 简化拼接 pairs_CostE2P_ ---
            # 在 MATLAB 中使用 ismember 是为了找回数据，但在这里 ICFinal 已经包含了所有列
            # 我们只需要直接从 ICFinal 中提取：
            # 第 12, 13 列 (ID) 和 第 16, 17, 18 列 (Extracted Data)
            # Python 索引：11, 12 和 15, 16, 17
            pairs_CostE2P_ = ICFinal[:, [11, 12, 15, 16, 17]]


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
    ax.cla() # 对应 cla
    # 假设 Draw_map 已封装，且接受 ax 对象作为参数
    Draw_map(ax, PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)

    # --- 2. 向量化绘制全局位置点与方向 ---
    # 绘制逃避者起始点、追捕者当前点、逃避者当前点
    ax.plot(Evader[:, 0], Evader[:, 1], 'kd', linewidth=2, label='Evader Start')
    ax.plot(PosP[:, 0], PosP[:, 1], 'mo', linewidth=2, label='Pursuer Pos')
    ax.plot(PosE[:, 0], PosE[:, 1], 'md', linewidth=2, label='Evader Pos')

    # 向量化绘制追捕者和逃避者的方向向量 (Quiver)
    # 注意：PosP[:, 2] 对应 MATLAB 的 PosP(:,3)
    ax.quiver(PosP[:, 0], PosP[:, 1], 200*np.cos(PosP[:, 2]), 200*np.sin(PosP[:, 2]), 
            color='c', angles='xy', scale_units='xy', scale=1, width=0.003)
    ax.quiver(PosE[:, 0], PosE[:, 1], 200*np.cos(PosE[:, 2]), 200*np.sin(PosE[:, 2]), 
            color='c', angles='xy', scale_units='xy', scale=1, width=0.003)

    # --- 3. 准备颜色集 ---
    numIntercepts = len(UnCapEid)
    colorSet = plt.cm.get_cmap('tab10', numIntercepts) # 替代 lines(numIntercepts)

    # --- 4. 遍历每个拦截任务绘图 ---
    for i in UnCapEid:
        color_p = colorSet(i)
        color_e = colorSet((i + 2) % numIntercepts)
        
        # 查找当前代理在分配表中的索引 (i+1 因为 MATLAB ID 从 1 开始)
        # 对应 MATLAB: pid=find(ICFinal(:,13)==i) -> ICFinal(:,12) 是 Python 0-based
        pid_arr = np.where(ICFinal[:, 12] == (i))[0]
        if len(pid_arr) == 0: continue
        pid = pid_arr[0]

        # --- 提取拦截点数据 (使用 0-based 索引) ---
        # ICFinal 索引对照: 1:te, 2:tp, 5:idxE, 6:idxP (MATLAB) 
        # -> 0:te, 1:tp, 4:idxE, 5:idxP (Python)
        tp_time_idx = int(ICFinal[pid, 1])
        te_time_idx = int(ICFinal[pid, 0])
        iso_idx_p = int(ICFinal[pid, 5])
        iso_idx_e = int(ICFinal[pid, 4])

        # 追捕者预测拦截点
        s_p = IsoMapPTP2Iso_i_tt_timeShift[pid][tp_time_idx]
        intercept_p_pos = s_p.IsoPos[:, iso_idx_p]
        intercept_p_theta = s_p.IsoVtheta[0, iso_idx_p]

        # 逃避者预测拦截点
        s_e = IsoMapETP2Val_i_tt_timeShift[pid][te_time_idx]
        intercept_e_pos = s_e.IsoPos[:, iso_idx_e]
        intercept_e_theta = s_e.IsoVtheta[0, iso_idx_e]

        # --- 绘制拦截点与方向 ---
        ax.plot(intercept_p_pos[0], intercept_p_pos[1], '^', color=color_p, markersize=6)
        ax.quiver(intercept_p_pos[0], intercept_p_pos[1], 200*np.cos(intercept_p_theta), 200*np.sin(intercept_p_theta), 
                color='r', angles='xy', scale_units='xy', scale=1, width=0.004)

        ax.plot(intercept_e_pos[0], intercept_e_pos[1], 's', color=color_e, markersize=6)
        ax.quiver(intercept_e_pos[0], intercept_e_pos[1], 200*np.cos(intercept_e_theta), 200*np.sin(intercept_e_theta), 
                color='r', angles='xy', scale_units='xy', scale=1, width=0.004)

        # --- 绘制轨迹线条 ---
        # 预测轨迹 (PathP, PathE)
        ax.plot(PathP[i][0, :], PathP[i][1, :], '-', color=color_p, linewidth=1.5, alpha=0.7)
        ax.plot(PathE[i][0, :], PathE[i][1, :], '-', color=color_e, linewidth=1.5, alpha=0.7)

        # 真实历史轨迹
        # 逃避者: PathE2Val_true[i] 是 N x 2
        len_e_true = int(t_all * v_E)
        ax.plot(PathE2Val_true[i][:len_e_true, 0], PathE2Val_true[i][:len_e_true, 1], 
                '.-', color=[0.3, 0.3, 0.3], linewidth=2, markersize=2)
        
        # 追捕者: PathPtrue[i] 是 2 x N
        len_p_true = int(t_all * v_P)
        ax.plot(PathPtrue[i][0, :len_p_true], PathPtrue[i][1, :len_p_true], 
                '-', color=[0.3, 0.3, 0.3], linewidth=2)

        # DWA 最优判断轨迹 (蓝色虚线)
        ax.plot(BestPaths[i][0, :], BestPaths[i][1, :], '--', color='blue', linewidth=2)

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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from datetime import datetime
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
from intercept.Prediction.predictLikelyTarget import predictLikelyTarget,predictLikelyTargetNew
from intercept.IsoPair.obtainNearTPall import obtainNearETP,obtainNearTP
from intercept.IsoPair.obtainPE2TP import obtainPE2TP
from intercept.IsoPair.obtainIsoPath import insertIsoMapP2TP
# parser = argparse.ArgumentParser()
# parser.add_argument('--nodraw', action='store_true', help='如果指定则不进行绘图，直接返回结果')
# args = parser.parse_args()

# === 1. 视频保存配置 (在循环开始前) ===
# 使用系统当前时间生成文件名（格式：MMDD_HHMM）
current_time = datetime.now().strftime('%m%d_%H%M')

# 创建output文件夹（如果不存在）
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_video_name = os.path.join(output_dir, f'uav_interception_{current_time}.mp4')
TimeMap='_0305_1800'
TimeIso='_0305_1800'


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

length_E_max=0
for i in range(len(pathFinalE2ValIn)):
    for j in range(len(pathFinalE2ValIn[i])):
        length_E_max=max(length_E_max,len(pathFinalE2ValIn[i][j][0]))

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
CapDist = 200  # m
CapRef={}
CapRef['CapDist'] = CapDist
CapRef['CapAngle'] = 87 #度
CapRef['Stepsize'] = Stepsize
CapRef['CapDistRef'] = 350 # camera_depth=350,
CapRef['v_P'] = v_P
CapRef['v_E'] = v_E
CapRef['timeIsoRes'] = timeIsoRes
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

CapRef['CapDistTime'] = np.array([CapDist] * num_e)
CapRef['CapAngleTime'] = np.array([CapRef['CapAngle']] * num_e)
# --- 2. 创建索引网格 (替代 ndgrid) ---
# indexing='ij' 确保与 MATLAB ndgrid 的行为一致
eid_idx, pid_idx, te_idx, tp_idx = np.meshgrid(
    np.arange(num_e), np.arange(num_p), np.arange(num_t1), np.arange(num_t2), 
    indexing='ij'
)

# --- 3. 筛选有效组合 (向量化逻辑) ---
# te 从 tp 开始，剔除 te < tp 的情况
valid_idx = te_idx >= tp_idx

pid_flat = pid_idx.ravel()
eid_flat = eid_idx.ravel()
tp_flat = tp_idx.ravel()
te_flat = te_idx.ravel()

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
            [te,tp], CapRef, pid, ValuePos, 
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
InterceptCandidates = obtainTask(IsoPairs_time_Eid_Pid_Posid, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt,CapRef)
IC = InterceptCandidates.copy()

# 排序并取前 100 个用于绘图
IC_Plot = IC[np.argsort(IC[:, 8])][:100]

# ax1=plt.figure(figsize=(10, 8))
# draw_candidates(IC_Plot, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt, pathFinalE2ValIn, pathFinalP2TP)
# Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)
# plt.show()

###########################################################转化从这开始

# --- 1. 预筛选：每个 (Pid, Eid) 仅保留 cost 最小的一行 ---
# lexsort 按从后往前的主次键排序：Pid(12) -> Eid(11) -> cost(8)
# 这样排序后，相同的 (Pid, Eid) 对会排列在一起，且 cost 最小的排在最前面
best_idx = np.lexsort((InterceptCandidates[:, 8], InterceptCandidates[:, 2], InterceptCandidates[:, 3]))
IC_sorted = InterceptCandidates[best_idx]

# 针对 (Pid, Eid) 进行去重，return_index=True 得到每个组合 cost 最小的行索引
_, unique_indices = np.unique(IC_sorted[:, [2, 3]], axis=0, return_index=True)
IC_candidates = IC_sorted[unique_indices]

# --- 2. 构建代价矩阵 ---
# 将原始 ID 映射为 0~N 的矩阵索引
E_set, e_inv = np.unique(IC_candidates[:, 2], return_inverse=True)
P_set, p_inv = np.unique(IC_candidates[:, 3], return_inverse=True)

# 构建矩阵：行对应 Pursuer，列对应 Evader
CostMat = np.full((len(P_set), len(E_set)), 1e9)
# 建立一个相同维度的“行号记录矩阵”，用于最后快速反查 IC_candidates 的行
# 这里存储的是 IC_candidates 的相对索引
CandidateIdxMat = np.full((len(P_set), len(E_set)), -1, dtype=int)

# 向量化填充
CostMat[p_inv, e_inv] = IC_candidates[:, 8]
CandidateIdxMat[p_inv, e_inv] = np.arange(len(IC_candidates))

# --- 3. 执行指派算法 ---
p_indices, e_indices = linear_sum_assignment(CostMat)

# --- 4. 提取结果与过滤 ---
# 过滤掉代价超过阈值的无效分配
valid_mask = CostMat[p_indices, e_indices] < 1e8
p_final = p_indices[valid_mask]
e_final = e_indices[valid_mask]

# 利用 CandidateIdxMat 一次性取出对应的原始 IC 数据行
final_rows = CandidateIdxMat[p_final, e_final]

ICFinal = IC_candidates[final_rows]
AssignedIntercepts = ICFinal.copy()  # 保持变量名一致
# 提取 [Eid, Pid] 配对
pairs_realE2P = AssignedIntercepts[:, [2, 3]] 

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

num_E = Evader.shape[0]
num_P = PStart_Point.shape[0]

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
TimeRes = 1 # s真实时间
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
PathP = [None] * num_P
for pid, tpid, isoid in zip(PidAll, PTPidAll, PIsoidAll):
    PathP[pid] = pathFinalP2TP[pid][tpid][:, :isoid]


# --- 4. 提取 Evader 路径信息 ---
EfromTPid = np.zeros(num_E)
# 11: Eid, 9: ValuePosid, 13: EIsoid
EidAll = ICFinal[:, 11].astype(int)
ValuePosid = ICFinal[:, 9].astype(int)
EIsoidAll = ICFinal[:, 13].astype(int)

# 提取 PathE: pathFinalE2ValIn{Eid, VPid}(:, 1:Isoid)
PathE = [None] * num_E
for eid, vpid, isoid in zip(EidAll, ValuePosid, EIsoidAll):
    PathE[eid] = pathFinalE2ValIn[eid][vpid][:, :isoid]


IsoMapPTP2Iso_i_tt_timeShift = IsoMapP2TP_i_tt.copy()  # 初始化为原始全量 Map 的副本
IsoMapETP2Val_i_tt_timeShift = IsoMapE2ValIn_i_tt.copy()
# --- 5. 准备绘图与循环环境 ---



# === 【视频保存配置】 ===
# ... (前面的数据加载和初始化代码保持不变) ...


metadata = dict(title='UAV Swarm Interception', artist='Python_Simulation')
# fps 建议设置为 20-30，根据你 t_all % 200 的跳帧频率调整
import matplotlib.pyplot as plt
import matplotlib as mpl

# 请将下方的路径替换为你电脑上 ffmpeg.exe 的实际路径
plt.rcParams['animation.ffmpeg_path'] = r'D:\JianyingPro\7.2.0.12475\ffmpeg.exe'

# 然后再创建 writer
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
# 初始化绘图
plt.ioff() # 关闭交互模式，提高保存效率
fig, ax = plt.subplots(figsize=(12, 10))

# 注意：writer.setup 必须在所有初始化之后调用
with writer.saving(fig, output_video_name, dpi=100):
    # 辅助变量初始化
    t = 0
    t_all = 0
    # 假设 steps_size, v_P, v_E, timeIsoRes 等变量已定义
    UnCapPid=np.array([i for i in range(num_P)])
    UnCapEid=np.array([i for i in range(num_E)])
    while (not np.all(distances < CapDist)) and (t_all < length_E_max /v_E):

        # --- 1. 时间更新 ---
        t += TimeRes / Stepsize
        t_all += TimeRes / Stepsize

        # --- 2. 确定未拦截的 E 和 P ---
        Capflag = distances < CapDist
        # pairs_realE2P: [Eid, Pid] (注意保持 ID 对应关系)
        if pairs_realE2P.shape[0]==Capflag.shape[0]:
            UnCapPidNew = pairs_realE2P[~Capflag, 1].astype(int)
            UnCapEidNew = pairs_realE2P[~Capflag, 0].astype(int)
            pairs_realE2P=pairs_realE2P[~Capflag] # 保持未捕获 Evader 的 ID 顺序
        # 假设在进入循环前已初始化这些变量为 None
        # IsoMapPTP2Iso_i_tt_timeShift = None 
        # IsoMapETP2Val_i_tt_timeShift = None

        flagPlot=1 if len(UnCapPidNew)!=len(UnCapPid) else 0


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
        PosP = np.array([PathP[np.where(pid==UnCapPid)[0][0]][:,min(curr_idx_p, PathP[np.where(pid==UnCapPid)[0][0]].shape[1] - 1)]
        for pid in UnCapPidNew])

        for id,pid in enumerate(UnCapPidNew):
        # 提取并排序 (替代最后一个 cellfun + cell2mat + sort)
            PathPtrue[pid] = np.hstack((PathPtrue[pid], PathP[np.where(pid==UnCapPid)[0][0]][:, max(0, seg_start) : seg_end]))



        curr_idx_e = int(round(t_all * v_E))
            # 提取并排序 (替代最后一个 cellfun + cell2mat + sort)
        PosE = np.array([PathE2Val_true[eid][min(curr_idx_e, PathE2Val_true[eid].shape[0] - 1), :]
            for eid in UnCapEidNew])

        PathEpre=[None]*len(UnCapEidNew)
        for id,eid in enumerate(UnCapEidNew):
        # 提取并排序 (替代最后一个 cellfun + cell2mat + sort)
            PathEpre[id] = PathE[np.where(eid==UnCapEid)[0][0]]


    #########################################################
        # 注：如果 PathE2Val_true 的结构是 (2 x Time)，则改为：
        # PathE2Val_true[i][:, min(...)]
        # --- 4. 攻击意图判断 (DWA) ---
        # 调用 DWA 子函数
        min_dists, BestPaths = obtainDWAprePath(PosE, PathEpre, E_PreRef)

        flagIn = 1 if np.all(min_dists <= CapDist) else 0

        # --- 5. 重新规划逻辑 ---
        if not flagIn:
            # 轨迹预测
            traj = [PathE2Val_true[i][:int(t_all * v_E), :] for i in range(num_E)]
            targets = ValuePos[:, :2]
            # res = predictLikelyTarget(traj, targets, UseWindow=3*v_E, Method='poly')
            res = predictLikelyTargetNew(traj, targets)
            Validnew = res['rank_idx'][:, 0]
            pairsE2Val_ = np.column_stack((np.arange(len(Validnew)), Validnew))
            pairsE2Val=pairsE2Val_[UnCapEidNew, :]       

            # 检查是否需要触发重规划
            NearTPpos_E, NearTPid_E = obtainNearETP(Trans_Point[:, :2], PosE, ValuePos, pairsE2Val)
            
            
            # if not np.array_equal(NearTPid_E, EfromTPid) or not np.array_equal(Validnew, Valid):
            if True:
                UnCapEid=UnCapEidNew
                UnCapPid=UnCapPidNew
                flagPlot=1
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
                    UnCapEid, UnCapPid, np.arange(TimeL), np.arange(TimeL), 
                    indexing='ij'
                )

                idxE, idxP, te_idx, tp_idx = np.meshgrid(
                    np.arange(num_e),np.arange(num_p),  np.arange(TimeL), np.arange(TimeL),
                    indexing='ij'
                )

                # 筛选 te >= tp 的有效组合 (向量化掩码)
                valid_mask = te_idx >= tp_idx
                eid_flat = eid_idx.ravel()
                pid_flat = pid_idx.ravel()
                te_flat = te_idx.ravel() # 转为 1-based 供子函数使用
                tp_flat = tp_idx.ravel()
                idxE_flat=idxE.ravel()
                idxP_flat=idxP.ravel()

                # --- 6. 执行拦截对搜索 (替代 arrayfun) ---
                # 使用 zip 并行迭代，这是 Python 中替代多参数 arrayfun 的最高性能方案
                # 定义参数
                x1, y1 = CapDist, CapDist
                x2, y2 = 2 * CapRef['CapDistRef'], CapRef['CapDistRef']

                # 单行嵌套公式实现
                CapDistTime = np.where(
                    distances < x1,
                    np.inf,  # 0 ~ CapDist 区间返回无穷大
                    np.where(
                        distances >= x2,
                        y2,  # 2*CapRef ~ inf 区间返回 CapRef
                        y1 + (distances - x1) * (y2 - y1) / (x2 - x1)  # 中间线性段 (CapDist ~ 2*CapRef)
                    )
                )

                CapRef['CapDistTime']=CapDistTime

                # 定义关键节点                
                # 定义线性段的端点
                x1, y1 = CapDist, CapRef['CapAngle']         # 起点 (CapDist, CapAngle)
                x2, y2 = 2 * CapRef['CapDistRef'], 180        # 终点 (2*CapRef, 180)

                # 单行嵌套公式实现
                CapAngleTime = np.where(
                    distances < x1,
                    360,  # 0 ~ CapDist 区间返回 360 度（全向允许）
                    np.where(
                        distances >= x2,
                        y2,   # 2*CapRef ~ inf 区间返回 180 度
                        y1 + (distances - x1) * (y2 - y1) / (x2 - x1)  # 中间线性段 (CapAngle ~ 180)
                    )
                )
                
                CapRef['CapAngleTime']=CapAngleTime

                results = [
                    [
                        te, tp, E2TP_TPIdx[ide], P2TP_TPIdx[idp],
                        obtainPTP2TP_IsoPos_timeShift2(
                            idp, ide, tp, te, CapRef, v_E, CapDistTime, 
                            IsoMap_i_tt_P2Iso, IsoMap_i_tt_E2Iso, ValuePos, pairsE2Val
                        ),
                        eid, pid,ide,idp
                    ]
                    for eid, pid, te, tp , ide,idp in zip(eid_flat, pid_flat, te_flat, tp_flat,idxE_flat,idxP_flat)
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
                    IsoPairs_time_ETPid_PTPid_Posid, IsoMap_i_tt_P2Iso, IsoMap_i_tt_E2Iso,CapRef
                )

                IC = InterceptCandidates.copy()
                
                # --- 任务指派 (完全向量化替代 accumarray/matchpairs) ---

                # --- 1. 预筛选：每个 (Pid, Eid) 仅保留 cost 最小的一行 ---
                # lexsort 按从后往前的主次键排序：Pid(12) -> Eid(11) -> cost(8)
                # 这样排序后，相同的 (Pid, Eid) 对会排列在一起，且 cost 最小的排在最前面
                best_idx = np.lexsort((IC[:, 8], IC[:, 11], IC[:, 12]))
                IC_sorted = IC[best_idx]
                
                # 针对 (Pid, Eid) 进行去重，return_index=True 得到每个组合 cost 最小的行索引
                _, unique_indices = np.unique(IC_sorted[:, [11, 12]], axis=0, return_index=True)
                IC_candidates = IC_sorted[unique_indices]

                # --- 2. 构建代价矩阵 ---
                # 将原始 ID 映射为 0~N 的矩阵索引
                E_set, e_inv = np.unique(IC_candidates[:, 11], return_inverse=True)
                P_set, p_inv = np.unique(IC_candidates[:, 12], return_inverse=True)
                
                # 构建矩阵：行对应 Pursuer，列对应 Evader
                CostMat = np.full((len(P_set), len(E_set)), 1e9)
                # 建立一个相同维度的“行号记录矩阵”，用于最后快速反查 IC_candidates 的行
                # 这里存储的是 IC_candidates 的相对索引
                CandidateIdxMat = np.full((len(P_set), len(E_set)), -1, dtype=int)
                
                # 向量化填充
                CostMat[p_inv, e_inv] = IC_candidates[:, 8]
                CandidateIdxMat[p_inv, e_inv] = np.arange(len(IC_candidates))

                # --- 3. 执行指派算法 ---
                p_indices, e_indices = linear_sum_assignment(CostMat)

                # --- 4. 提取结果与过滤 ---
                # 过滤掉代价超过阈值的无效分配
                valid_mask = CostMat[p_indices, e_indices] < 1e8
                p_final = p_indices[valid_mask]
                e_final = e_indices[valid_mask]

                # 利用 CandidateIdxMat 一次性取出对应的原始 IC 数据行
                final_rows = CandidateIdxMat[p_final, e_final]
                AssignedIntercepts = IC_candidates[final_rows]
                
                # 提取 [Eid, Pid] 配对
                pairs_realE2P_ = AssignedIntercepts[:, [11, 12]]

                ICFinal = AssignedIntercepts.copy()  # 保持变量名一致
                pairs_realE2P=np.array([[UnCapEid[pairs_realE2P_[i,0].astype(int)],UnCapPid[pairs_realE2P_[i,1].astype(int)] ]
                                for i in range(len(pairs_realE2P_))])

                # 更新 TimeShift 缓存列表 (List Comprehension 处理对象)
                IsoMapPTP2Iso_i_tt_timeShift = [IsoMap_i_tt_P2Iso[i] for i in p_indices]
                IsoMapETP2Val_i_tt_timeShift = [IsoMap_i_tt_E2Iso[i] for i in e_indices]

                # --- 6. 结果拼接 (优化：无需 ismember，数据已天然对齐) ---
                # MATLAB: extractedData = AssignedIntercepts(Locb(Lia), 16:18)
                # Python: 提取第 16-18 列 (索引 15:18)
                extractedData = AssignedIntercepts[:, 15:18]

                # 最终拼接结果 [Eid, Pid, data1, data2, data3]
                pairs_CostE2P_ = np.column_stack((pairs_realE2P_, extractedData))

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
                        PathE2TP[idx], 
                        pathFinalTP2Val[ETP_idx[i]][VP_idx[i]][:, :int(Iso_idx[i] - PathE2TP[idx].shape[1])]
                    )) if VP_idx[i] >= 0 else PathE2TP[idx][:, :int(Iso_idx[i])]
                    for i, idx in enumerate(E_idx)
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
                        PathP2TP[idx], 
                        pathFinalMapPTP2Iso[TP_from_idx[i]][TP_to_idx[i]][:, :int(Iso_p_idx[i] - PathP2TP[idx].shape[1])]
                    )) if TP_to_idx[i] >= 0 else PathP2TP[idx][:, :int(Iso_p_idx[i])]
                    for i, idx in enumerate(P_idx)
                ]

                # 全局重排 (替代 PathP_(PidAns)=PathP_)
                PathP = [PathP_active[i] for i in np.argsort(P_idx)]

                # 重置时间
                t = 0
        # --- 6. 绘图 (交互式更新) ---#

        # --- 1. 清理与基础地图绘制 ---
    # --- 1. 清理与基础地图绘制 ---
        if t_all%10==0 or flagPlot==1:
        # if True:
            ax.cla()
            
            Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, 
                     obs_no_circle, obs_no_circle_in, ax=ax)

            # --- 2. 准备颜色集 (使用 tab20 提供更多对比度) ---
            numIntercepts = len(UnCapEidNew)
            # tab20 每两个颜色为一组同色系，适合表示 Pursuer-Evader 对
            color_map = mpl.colormaps['tab20'] 

            # --- 3. 向量化绘制全局位置点 (作为背景) ---
            ax.scatter(Evader[:, 0], Evader[:, 1], c='gray', marker='d', s=20, alpha=0.5, label='Evader Start')
            
            # --- 4. 遍历每个拦截任务绘图 ---
            for idx, eid in enumerate(UnCapEidNew):
                # 为每组任务分配 tab20 中的一对颜色
                # color_p (追捕者): 深色； color_e (逃避者): 浅色
                color_p = color_map((idx * 2) % 20)
                color_e = color_map((idx * 2 + 1) % 20)
                
                # 查找分配表索引
                IcId = np.where(ICFinal[:, 11] == idx)
                pid = int(ICFinal[IcId, 12])  # 获取对应的 Pid

                # --- 提取拦截点数据 ---
                tp_time_idx = int(ICFinal[IcId, 1])
                te_time_idx = int(ICFinal[IcId, 0])
                iso_idx_p = int(ICFinal[IcId, 5])
                iso_idx_e = int(ICFinal[IcId, 4])

                # # 追捕者与逃避者预测拦截点
                # s_p = IsoMapPTP2Iso_i_tt_timeShift[pid][tp_time_idx]
                # intercept_p_pos = s_p.IsoPos[:, iso_idx_p]
                # intercept_p_theta = s_p.IsoVtheta[iso_idx_p]

                # s_e = IsoMapETP2Val_i_tt_timeShift[idx][te_time_idx]
                # intercept_e_pos = s_e.IsoPos[:, iso_idx_e]
                # intercept_e_theta = s_e.IsoVtheta[iso_idx_e]

                # --- 绘制当前实时点与方向 (向量颜色跟随对象) ---
                ax.plot(PosP[idx, 0], PosP[idx, 1], 'o', color=color_p, markersize=8, markeredgecolor='w',alpha=1)
                ax.quiver(PosP[idx, 0], PosP[idx, 1], 150*np.cos(PosP[idx, 2]), 150*np.sin(PosP[idx, 2]), 
                        color=color_p, angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.8)

                ax.plot(PosE[idx, 0], PosE[idx, 1], 'd', color=color_e, markersize=8, markeredgecolor='w',alpha=1)
                ax.quiver(PosE[idx, 0], PosE[idx, 1], 150*np.cos(PosE[idx, 2]), 150*np.sin(PosE[idx, 2]), 
                        color=color_e, angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.8)

                # # --- 绘制拦截预测点 (增加白色边框突出显示) ---
                # ax.plot(intercept_p_pos[0], intercept_p_pos[1], '^', color=color_p, markersize=5, markeredgecolor='k', label=f'P-{pid} Intercept')
                # ax.plot(intercept_e_pos[0], intercept_e_pos[1], 's', color=color_e, markersize=5, markeredgecolor='k')

                # --- 绘制预测轨迹 (虚线表示预测) ---
                ax.plot(PathP[idx][0, :], PathP[idx][1, :], '-', color=color_p, linewidth=2, alpha=1)
                ax.plot(PathE[idx][0, :], PathE[idx][1, :], '--', color=color_e, linewidth=2, alpha=1)

                # ax.plot(PathP2TP[pid][0, :], PathP2TP[pid][1, :], '-', color='black', linewidth=1, alpha=0.6)
                # ax.plot(PathE2TP[eid][0, :], PathE2TP[eid][1, :], '--', color='black', linewidth=1, alpha=0.6)

                # ax.plot(PathP[i][0, :], PathP[i][1, :], '--', color=color_p, linewidth=1, alpha=0.6)
                # ax.plot(PathE[i][0, :], PathE[i][1, :], '--', color=color_e, linewidth=1, alpha=0.6)
                


                # --- DWA 最优路径 (使用高亮的金黄色或对应颜色的实线) ---
                ax.plot(BestPaths[idx][0, :], BestPaths[idx][1, :], '-', color=color_p, linewidth=2.5, zorder=3)

        # 设置图例避免遮挡
        # ax.legend(loc='upper right', fontsize='x-small', ncol=2)
                        # --- 绘制真实历史轨迹 (加粗并稍微调淡灰色) ---
            for iPlot in range(PathE2Val_true):
                len_e_true = int(t_all * v_E)
                ax.plot(PathE2Val_true[iPlot][:len_e_true, 0], PathE2Val_true[iPlot][:len_e_true, 1], 
                        '-', color='gray', linewidth=2.5, alpha=0.3, zorder=1)
                
                len_p_true = int(t_all * v_P)
                ax.plot(PathPtrue[iPlot][0, :len_p_true], PathPtrue[iPlot][1, :len_p_true], 
                        '-', color='gray', linewidth=2.5, alpha=0.3, zorder=1)
        # --- 5. 刷新画布 ---
            ax.set_title(f"Simulation Time: {t_all:.2f}")
            plt.grid(True, linestyle='--', alpha=0.5)
            # plt.show()
            
            # 重要：捕获当前帧
            writer.grab_frame()

        # === 【捕获当前帧】 ===
        # writer.grab_frame()
        # ======================

        # 更新距离用于循环判断
        distances = np.linalg.norm(
            PosP[:, :2] - 
            PosE[:, :2], 
            axis=1
        )

        if np.all(distances < CapDist):
            print(f"All evaders captured at time {t_all:.2f} seconds.")


plt.figure(figsize=(10, 8))
Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in, ax)
len_e_true = int(t_all * v_E)
len_p_true = int(t_all * v_P)

for eid in range(num_E):
    PosEfinal=PathE2Val_true[eid][min(len_e_true-1, PathE2Val_true[eid].shape[0]-1), :]

    plt.plot(PathE2Val_true[eid][:len_e_true, 0], PathE2Val_true[eid][:len_e_true, 1], 
            '-', color='gray', linewidth=2.5, alpha=0.3, zorder=1)

    plt.plot(PosEfinal[0], PosEfinal[1], 'd', color='black', markersize=8, markeredgecolor='w',alpha=1)
    plt.quiver(PosEfinal[0], PosEfinal[1], 150*np.cos(PosEfinal[2]), 150*np.sin(PosEfinal[2]), 
            color='black', angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.8)




for pid in range(num_P):
    PosPfinal=PathPtrue[pid][:, min(len_p_true-1, PathPtrue[pid].shape[1]-1)]

    plt.plot(PathPtrue[pid][0, :len_p_true], PathPtrue[pid][1, :len_p_true], 
            '-', color='gray', linewidth=2.5, alpha=0.3, zorder=1)
    plt.plot(PosPfinal[0], PosPfinal[1], 'o', color='black', markersize=8, markeredgecolor='w',alpha=1)
    plt.quiver(PosPfinal[0], PosPfinal[1], 150*np.cos(PosPfinal[2]), 150*np.sin(PosPfinal[2]), 
            color=color_p, angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.8)



print("Simulation completed and video saved as:", output_video_name)

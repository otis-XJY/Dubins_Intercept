import os
import sys
from datetime import datetime

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调用转换后的函数
from intercept.IsoPair.obtainIsoPairsAll import obtainPTP2TP_IsoPos_timeShift2
from intercept.IsoPair.obtainTaskAll import obtainTask_timeShift2
from Draw.Draw_map import Draw_map
from intercept.Prediction.obtainDWAprePath import obtainDWAprePath
from intercept.Prediction.predictLikelyTarget import predictLikelyTargetNew
from intercept.IsoPair.obtainNearTPall import obtainNearETP,obtainNearTP
from intercept.IsoPair.obtainPE2TP import obtainPE2TP
from intercept.IsoPair.obtainIsoPath import insertIsoMapP2TP
from intercept.IsoPair.obtainRLOutput import obtain_output_iso,obtainNeighbour

import torch

from marl0327.config import MARLConfig
from marl0327.intercept_select import build_rl_model, refine_icfinal_with_model

TimeMap='0320_0920'
TimeIso='0320_0920'

# === 1. 视频保存配置 (在循环开始前) ===
# 使用系统当前时间生成文件名（格式：MMDD_HHMM）
current_time = datetime.now().strftime('%m%d_%H%M')

# 创建output文件夹（如果不存在）
output_dir = 'output/'+TimeMap+'/'+TimeIso
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_video_name = os.path.join(output_dir, f'uav_interception_{current_time}.mp4')



Map = joblib.load('./map/'+TimeMap+'/Map.jbl')	

IsoMapPTP2Iso_i_tt = joblib.load('./map/'+TimeMap+'/IsoMapPTP2Iso_i_tt.jbl')	  
IsoMapPIso2TP_i_tt=joblib.load('./map/'+TimeMap+'/IsoMapPIso2TP_i_tt.jbl')
pathFinalMapPTP2Iso = joblib.load('./map/'+TimeMap+'/pathFinalMapPTP2Iso.jbl')

pathFinalE2ValIn = joblib.load('./map/'+TimeIso+'/pathFinalE2ValIn.jbl')	
PathE2Val_true=joblib.load('./map/'+TimeIso+'/PathE2Val_true.jbl')

IsoMapTP2Val_i_tt = joblib.load('./map/'+TimeMap+'/IsoMapTP2Val_i_tt.jbl')
pathFinalTP2Val = joblib.load('./map/'+TimeMap+'/pathFinalTP2Val.jbl')

length_E_max=0
for i in range(len(pathFinalE2ValIn)):
    for j in range(len(pathFinalE2ValIn[i])):
        length_E_max=max(length_E_max,len(pathFinalE2ValIn[i][j][0]))

IsoMapETP2Iso_i_tt=joblib.load('./map/'+TimeMap+'/IsoMapETP2Iso_i_tt.jbl')
IsoMapEIso2TP_i_tt=joblib.load('./map/'+TimeMap+'/IsoMapEIso2TP_i_tt.jbl')
pathFinalMapETP2Iso=joblib.load('./map/'+TimeMap+'/pathFinalMapETP2Iso.jbl')

# 1. 图形窗口清理
# MATLAB: clf, close all
# plt.close('all')

# 2. 从 Map 中提取变量 (假设 Map 是一个字典)
# 严格保留变量名
obs = Map['obs']
sure = Map['sure']
obs_no_circle = Map['obs_no_circle']
obs_no_circle_in = Map['obs_no_circle_in']
Stepsize = Map['Stepsize']
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


obs_polygons = [
    Polygon(np.asarray(poly, dtype=float)[:, :2])
    for poly in (obs_no_circle)
    if poly is not None
]
CapRef['obs_polygons'] = obs_polygons

# RL：任务分配仍由匈牙利完成；拦截点由策略网络在 (E,P) 候选子集中选择
rl_cfg = MARLConfig()
_rl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_rl_model = build_rl_model(rl_cfg, _rl_device)
_rl_model.eval()
# 训练后可取消注释加载权重：_rl_model.load_state_dict(torch.load("path.pt", map_location=_rl_device))

# RL输出邻域半径配置
distance_P = float(CapRef['CapDistRef'])
distance_E = float(CapRef['CapDistRef'])
distance_V = float(CapRef['CapDistRef'])

num_E = Evader.shape[0]
num_P = PStart_Point.shape[0]

CapRef['CapDistTime'] = np.array([CapDist] * num_E)
CapRef['CapAngleTime'] = np.array([CapRef['CapAngle']] * num_E)

PosE = Evader.copy()
PosP = PStart_Point.copy()
distances = np.linalg.norm(
    PosP[:, :2] - 
    PosE[:, :2], 
    axis=1
)
# 计算当前配对的距离
# pairs_realE2P: [Eid, Pid] (1-based IDs)
# 转换为 0-based 索引进行切片




# --- 2. 转换 PathPtrue (替代 mat2cell) ---
# MATLAB: PathPtrue = mat2cell(PosP', size(PosP,2), ones(1,size(PosP,1)))'
# 意图：将 PosP 的每一行提取出来作为一个独立的数组存入列表，且保持为列向量(2x1 或 3x1)
PathPtrue = [p.reshape(-1, 1) for p in PosP]

# 时间参数
TimeRes = 1 # s真实时间
t = 0
t_all = 0



# 提取 PathP: pathFinalP2TP{Pid, TPid}(:, 1:Isoid)
# 使用列表推导式配合 zip 替代 arrayfun，处理非规整切片

PathP = [None] * num_P
for pid in range(num_P):
    PathP[pid] = np.tile(PosP[pid].reshape(-1, 1), (1, length_E_max))

PathE = [None] * num_E
for eid in range(num_E):
    PathE[eid] =  PathE2Val_true[eid][:int(TimeRes / Stepsize*v_E),:].T




Capflag=np.array([False]*num_E) # 初始化捕获标志
pairs_realE2P=None
# --- 5. 准备绘图与循环环境 ---



# === 【视频保存配置】 ===
# ... (前面的数据加载和初始化代码保持不变) ...


# 请将下方的路径替换为你电脑上 ffmpeg.exe 的实际路径
plt.rcParams['animation.ffmpeg_path'] = r'D:\JianyingPro\7.2.0.12475\ffmpeg.exe'

# 然后再创建 writer
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
    UnCapPidNew=UnCapPid
    UnCapEid=np.array([i for i in range(num_E)])
    UnCapEidNew=UnCapEid
    step = 0  # 决策步: 仅在触发重规划时递增
    decision_outputs = []
    while (not np.all(Capflag)) and (t_all < length_E_max /v_E):
##############################——————————————————————————————————env.update for cursor 
#  RL: _phase_update
        # --- 1. 时间更新 ---
        t += TimeRes / Stepsize
        t_all += TimeRes / Stepsize

        # --- 2. 确定未拦截的 E 和 P ---
        # Capflag = distances < CapDist
        # pairs_realE2P: [Eid, Pid] (注意保持 ID 对应关系)
        if pairs_realE2P is not None and pairs_realE2P.shape[0]==Capflag.shape[0]:
            UnCapPidNew = pairs_realE2P[~Capflag, 1].astype(int)
            UnCapEidNew = pairs_realE2P[~Capflag, 0].astype(int)
            pairs_realE2P=pairs_realE2P[~Capflag] # 保持未捕获 Evader 的 ID 顺序
        # 假设在进入循环前已初始化这些变量为 None
        # IsoMapPTP2Iso_i_tt_timeShift = None 
        # IsoMapETP2Val_i_tt_timeShift = None

        flagPlot=1 if len(UnCapPidNew)!=len(UnCapPid) else 0

#  RL: _advance_from_paths//_phase_geometry_step
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

#  RL: _phase_check_decision
    #########################################################
        # 注：如果 PathE2Val_true 的结构是 (2 x Time)，则改为：
        # PathE2Val_true[i][:, min(...)]
        # --- 4. 攻击意图判断 (DWA) ---
        # 调用 DWA 子函数
        min_dists, BestPaths = obtainDWAprePath(PosE, PathEpre, E_PreRef)

        flagIn = 1 if np.all(min_dists <= 10) else 0

        # --- 5. 重新规划逻辑 ---
##############################——————————————————————————————————env.step begin for cursor
        if not flagIn:
# RL: _compute_isomap_intercept_candidates
            step += 1
            # 轨迹预测
            traj = [PathE2Val_true[i][:int(t_all * v_E), :] for i in range(num_E)]
            targets = ValuePos[:, :2]
            # res = predictLikelyTarget(traj, targets, UseWindow=3*v_E, Method='poly')
            res = predictLikelyTargetNew(traj, targets)
            Validnew = res['rank_idx'][:, 0]
            pairsE2Val_ = np.column_stack((np.arange(len(Validnew)), Validnew))
            pairsE2Val=pairsE2Val_[UnCapEidNew, :]       

            # 检查是否需要触发重规划
            _, NearTPid_E = obtainNearETP(Trans_Point[:, :2], PosE, ValuePos, pairsE2Val)
            
            
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
                # 提取 TP 索引：MATLAB 逻辑是 x(:,2) 后取 x{1}，在 Python 结果列表中对应索引 1
                E2TP_TPIdx = np.array([res[1] for res in E2TP_results])

                # --- 2. 获得 P2TP (追捕者到转移点路径) ---
                # 计算最近 TP
                _, NearTPid_P = obtainNearTP(Trans_Point[:, :2], PosP, PosE)
                # 计算路径
                PathP2TP, IsoMapP2TP, P2TP_results = obtainPE2TP(
                    IsoMapPIso2TP_i_tt, NearTPid_P, PosP, Trans_Point, pathFinalMapPTP2Iso, Map, Map['v_P']
                )

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

                # 仅在决策时刻输出，并以Pid作为主键对齐。
                output_P_state = {int(pid): PosP[idx].copy() for idx, pid in enumerate(UnCapPid)}
                output_P_alley_state = obtainNeighbour(
                    PosP,
                    PosP,
                    distance_P,
                    query_ids=UnCapPid,
                    target_ids=UnCapPid,
                    return_mode="dict",
                )
                output_enemy_state = obtainNeighbour(
                    PosP,
                    PosE,
                    distance_E,
                    query_ids=UnCapPid,
                    target_ids=UnCapEid,
                    return_mode="dict",
                )
                output_IsoP = obtain_output_iso(IC, IsoMapP2TP_i_tt, None)
                output_E_state = {int(eid): PosE[idx].copy() for idx, eid in enumerate(UnCapEid)}
                output_V_state = obtainNeighbour(
                    PosP,
                    ValuePos,
                    distance_V,
                    query_ids=UnCapPid,
                    target_ids=np.arange(ValuePos.shape[0]),
                    return_mode="dict",
                )

                decision_outputs.append(
                    {
                        "step": int(step),
                        "t_all": float(t_all),
                        "pids": UnCapPid.copy(),
                        "eids": UnCapEid.copy(),
                        "output_P_state": output_P_state,
                        "output_P_alley_state": output_P_alley_state,
                        "output_enemy_state": output_enemy_state,
                        "output_IsoP": output_IsoP,
                        "output_E_state": output_E_state,
                        "output_V_state": output_V_state,
                    }
                )


    # plt.figure(figsize=(10, 8))
    # draw_candidates(IC_candidates, IsoMap_i_tt_P2Iso, IsoMap_i_tt_E2Iso, pathFinalE2ValIn, pathFinalP2TP)
    # Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in)
    # plt.show()
                # --- 任务指派 (完全向量化替代 accumarray/matchpairs) ---

                # 1. 识别所有唯一的 (Pid_ref, Eid_ref) 组合
                unique_pairs = np.unique(IC[:, [11, 12]], axis=0)
                best_candidates_list = []

                for pair in unique_pairs:
                    eid_val, pid_val = pair
                    # 提取当前 (Eid_ref, Pid_ref) 对的所有候选行
                    mask = (IC[:, 11] == eid_val) & (IC[:, 12] == pid_val)
                    group = IC[mask]

                    # 按优先级排序：索引15 Delt_t降序，索引8 cost 升序，索引16 Delat_d 升序
                    # np.lexsort 的最后一个键为主键
                    sort_idx = np.lexsort((group[:, 16], group[:, 8], -group[:, 15]))
                    best_candidates_list.append(group[sort_idx[0]])

                # 转换为 numpy 数组
                IC_candidates = np.array(best_candidates_list)
# RL:_apply_hungarian_and_paths
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
# RL:
                # 提取 [Eid, Pid] 配对
                pairs_realE2P_ = AssignedIntercepts[:, [11, 12]]

                pairs_realE2P=np.array([[UnCapEid[pairs_realE2P_[i,0].astype(int)],UnCapPid[pairs_realE2P_[i,1].astype(int)] ]
                                for i in range(len(pairs_realE2P_))])

                # 匈牙利仅决定 E-P 配对；具体拦截候选行由模型在 IC 的该 (E_ref,P_ref) 子集中选取
                decision_entry = {
                    "output_P_state": output_P_state,
                    "output_P_alley_state": output_P_alley_state,
                    "output_enemy_state": output_enemy_state,
                    "output_V_state": output_V_state,
                }
                ICFinal = refine_icfinal_with_model(
                    IC,
                    AssignedIntercepts,
                    output_IsoP,
                    decision_entry,
                    UnCapPid,
                    _rl_model,
                    _rl_device,
                    rl_cfg,
                )

# RL: _apply_paths_from_assigned_rows
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
##############################——————————————————————————————————env.step end for cursor
        # --- 6. 绘图 (交互式更新) ---#

        # --- 1. 清理与基础地图绘制 ---
    # --- 1. 清理与基础地图绘制 ---
        if t_all%10==0 or flagPlot==1:
        # if True:
            ax.cla()
            
            Draw_map(PStart_Point, Trans_Point, ValuePos, obs, sure, 
                     obs_no_circle, obs_no_circle_in, ax=ax)

            # --- 2. 准备颜色集 (使用 tab20 提供更多对比度) ---
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
            for iPlot in range(len(PathE2Val_true)):
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
##############################——————————————————————————————————env.check for cursor
# RL: _update_capflag_from_geometry
        # 更新距离用于循环判断
        distances = np.linalg.norm(
            PosP[:, :2] - 
            PosE[:, :2], 
            axis=1
        )

        # --- Step 2. 角度差与代价 (向量化广播) ---
        # dx, dy 均为 (nE, nP) 矩阵
        dx = PosE[:, 0] - PosP[:, 0]
        dy = PosE[:, 1] - PosP[:, 1]
        
        # 计算 P 到 E 的方位角 (转换为度)
        angPE = np.degrees(np.arctan2(dy, dx))
        
        # 计算朝向偏差: angPE - Pori (Pori 转换为度)
        # 利用广播: (nE, nP) - (nP,) -> (nE, nP)
        angleDiff_ = angPE - np.degrees(PosP[:,2])
        
        # 角度归一化到 [-180, 180]
        angleDiff = (angleDiff_ + 180) % 360 - 180

        Capflag = (distances <= CapDist) & (np.abs(angleDiff) <= CapRef['CapAngle']/2)        

        if np.all(Capflag):
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

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import joblib
import time  # 添加时间测试模块
from collections import OrderedDict

# ============================================
# 性能分析器类定义
# ============================================
class PerformanceProfiler:
    def __init__(self):
        self.timings = OrderedDict()
        self.current_timer = None
        self.start_time = None

    def start(self, name):
        """开始计时某个代码块"""
        self.current_timer = name
        self.start_time = time.perf_counter()

    def stop(self):
        """停止当前计时"""
        if self.current_timer and self.start_time:
            elapsed = time.perf_counter() - self.start_time
            if self.current_timer in self.timings:
                self.timings[self.current_timer] += elapsed
            else:
                self.timings[self.current_timer] = elapsed
            self.current_timer = None
            self.start_time = None

    def report(self):
        """生成性能报告"""
        print("\n" + "="*60)
        print("性能分析报告 - 各模块耗时")
        print("="*60)

        total_time = sum(self.timings.values())
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        for name, elapsed in sorted_timings:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{name:<40} {elapsed:>8.4f}s ({percentage:>5.1f}%) {bar}")

        print("-"*60)
        print(f"{'总计':<40} {total_time:>8.4f}s (100.0%)")
        print("="*60)

        # 找出最耗时的部分
        if sorted_timings:
            slowest = sorted_timings[0]
            print(f"\n最耗时模块: [{slowest[0]}] 占用 {slowest[1]:.4f}s ({(slowest[1]/total_time)*100:.1f}%)")

# 创建全局性能分析器
profiler = PerformanceProfiler()

# ============================================
# 主程序开始
# ============================================
profiler.start("0. 模块导入与初始化")

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
from intercept.IsoPair.obtainIsoPairsAll import obtainIsoPairs
from intercept.IsoPair.obtainTaskAll import obtainTask
from Draw.DrawIntercept import draw_candidates
from Draw.Draw_map import Draw_map
from scipy.optimize import linear_sum_assignment

profiler.stop()

# ============================================
# 1. 数据加载阶段
# ============================================
profiler.start("1. 数据加载 (joblib)")

TimeMap='_0206_2010'
TimeIso='_0206_2010'

Map = joblib.load('Map'+TimeMap+'.jbl')	
IsoMapP2TP_i_tt = joblib.load('IsoMapP_i_tt'+TimeMap+'.jbl')	    
IsoMapPTP2Iso_i_tt = joblib.load('IsoMapPTP2Iso_i_tt'+TimeMap+'.jbl')	  
pathFinalP2TP = joblib.load('pathFinalP'+TimeMap+'.jbl')	  
IsoMapE2ValIn_i_tt = joblib.load('IsoMapE2ValIn_i_tt'+TimeIso+'.jbl')	
pathFinalE2ValIn = joblib.load('pathFinalE2ValIn'+TimeIso+'.jbl')	

profiler.stop()

# ============================================
# 2. 数据提取与初始化
# ============================================
profiler.start("2. Map数据提取与初始化")

obs = Map['obs']
sure = Map['sure']
r = Map['r']
obs_no_circle = Map['obs_no_circle']
obs_no_circle_in = Map['obs_no_circle_in']
outline_all = Map['outline_all']
Stepsize = Map['Stepsize']
resolution = Map['resolution']
v_P = Map['v_P']
v_E = Map['v_E']
Trans_Point = Map['Trans_Point']
ValuePos = Map['ValuePos']
PStart_Point=Map['PStart_Point']

# 定义 Evader 矩阵
Evader = np.array([
    [200, 1950, -np.pi/2],
    [1000, 1950, -np.pi/2],
    [1800, 1950, -np.pi/2]
])

# 初始化 E_PreRef 结构体
E_PreRef = {
    'num_v': 10,
    'num_w': 10,
    'v_range': [100, 150],
    'w_range': [-np.pi/6, np.pi/6],
    'Stepsize': Map['Stepsize'],
    'T_pred': 1
}

profiler.stop()

# ============================================
# 3. 基础参数与数据提取
# ============================================
profiler.start("3. 基础参数设置与数据提取")

CapDist = 100
PosE = Evader
PosP = PStart_Point

# 使用列表推导式处理尺寸不一的结构体
ETP2Val_IsoPos = [[item.IsoPos for item in row] for row in IsoMapE2ValIn_i_tt]
ETP2Val_pathid = [[item.pathid for item in row] for row in IsoMapE2ValIn_i_tt]
PTP2TP_IsoPos = [[item.IsoPos for item in row] for row in IsoMapP2TP_i_tt]

num_e = len(ETP2Val_IsoPos)
num_t1 = len(ETP2Val_IsoPos[0])
num_p = len(PTP2TP_IsoPos)
num_t2 = len(PTP2TP_IsoPos[0])

profiler.stop()

# ============================================
# 4. 创建索引网格 (向量化操作)
# ============================================
profiler.start("4. 创建索引网格 (np.meshgrid)")

eid_idx, pid_idx, te_idx, tp_idx = np.meshgrid(
    np.arange(num_e), np.arange(num_p), np.arange(num_t1), np.arange(num_t2), 
    indexing='ij'
)

valid_idx = te_idx >= tp_idx

pid_flat = pid_idx[valid_idx]
eid_flat = eid_idx[valid_idx]
tp_flat = tp_idx[valid_idx]
te_flat = te_idx[valid_idx]

profiler.stop()

# ============================================
# 5. 意图分析
# ============================================
profiler.start("5. 意图分析 (analyzeEvaderIntent)")

Valid, threatMatrix = analyzeEvaderIntent(ValuePos, Evader)
pairsE2Val = np.column_stack((np.arange(len(Valid)), Valid))

profiler.stop()

# ============================================
# 6. 获取 IsoPairs (核心计算瓶颈)
# ============================================
profiler.start("6. 获取 IsoPairs (obtainIsoPairs循环)")

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

profiler.stop()

# ============================================
# 7. 展平、过滤与排序
# ============================================
profiler.start("7. 结果展平、过滤与排序")

flat_results = [r for r in results if r[4] is not None and len(r[4]) > 0]
IsoPairs_time_Eid_Pid_Posid_ = np.array(flat_results, dtype=object)
sort_idx = np.argsort(IsoPairs_time_Eid_Pid_Posid_[:, 0].astype(float))
IsoPairs_time_Eid_Pid_Posid = IsoPairs_time_Eid_Pid_Posid_[sort_idx]

profiler.stop()

# ============================================
# 8. 任务获取与初步筛选
# ============================================
profiler.start("8. 任务获取 (obtainTask)")

InterceptCandidates = obtainTask(IsoPairs_time_Eid_Pid_Posid, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt)
IC = InterceptCandidates
IC_Plot = IC[np.argsort(IC[:, 8])][:100]

profiler.stop()

# ============================================
# 9. 构建代价矩阵与任务分配 (核心优化点)
# ============================================
profiler.start("9. 构建代价矩阵与任务分配")

EidAll = IC[:, 2]
PidAll = IC[:, 3]
cost = IC[:, 8]

E_set = np.unique(EidAll)
P_set = np.unique(PidAll)
nE, nP = len(E_set), len(P_set)

# 快速映射 ID 到索引
E_map = {id_val: i for i, id_val in enumerate(E_set)}
P_map = {id_val: i for i, id_val in enumerate(P_set)}

# 为每个 (Eid, Pid) 对寻找最小 cost 的索引
sort_keys = np.lexsort((cost, PidAll, EidAll))
IC_sorted = IC[sort_keys]

_, unique_indices = np.unique(IC_sorted[:, [2, 3]].astype(float), axis=0, return_index=True)
best_IC_per_EP = IC_sorted[unique_indices]

# 构建 Cost 矩阵
CostMat = np.full((nE, nP), 1e9)
row_indices = np.array([E_map[e] for e in best_IC_per_EP[:, 2]])
col_indices = np.array([P_map[p] for p in best_IC_per_EP[:, 3]])
CostMat[row_indices, col_indices] = best_IC_per_EP[:, 8]

# 线性指派算法
row_ind, col_ind = linear_sum_assignment(CostMat)

mask = CostMat[row_ind, col_ind] < 1e9
row_ind = row_ind[mask]
col_ind = col_ind[mask]

pairs_realE2P = np.column_stack((E_set[row_ind], P_set[col_ind]))

profiler.stop()

# ============================================
# 10. 提取最终分配结果
# ============================================
profiler.start("10. 提取最终分配结果")

final_results_mask = np.array([
    np.where((best_IC_per_EP[:, 2] == e) & (best_IC_per_EP[:, 3] == p))[0][0]
    for e, p in pairs_realE2P
])
AssignedIntercepts = best_IC_per_EP[final_results_mask]
ICFinal = AssignedIntercepts

extractedData = AssignedIntercepts[:, 15:18] 
pairs_CostE2P = np.column_stack((pairs_realE2P, extractedData))

profiler.stop()

# ============================================
# 11. 后续路径处理
# ============================================
profiler.start("11. 路径提取与处理")

PosE = Evader.copy()
PosP = PStart_Point.copy()

idx_E_dist = pairs_realE2P[:, 0].astype(int) - 1
idx_P_dist = pairs_realE2P[:, 1].astype(int) - 1

distances = np.sqrt(np.sum((PosP[idx_P_dist, :2] - PosE[idx_E_dist, :2])**2, axis=1))

PathPtrue = [p.reshape(-1, 1) for p in PosP]

TimeRes = 0.01
t = 0
t_all = 0

PidAll = ICFinal[:, 12].astype(int)
PTPidAll = ICFinal[:, 10].astype(int)
PIsoidAll = ICFinal[:, 14].astype(int)

PathP = [
    pathFinalP2TP[pid][tpid][:, :isoid] 
    for pid, tpid, isoid in zip(PidAll, PTPidAll, PIsoidAll)
]

EfromTPid = np.zeros(PosE.shape[0])
EidAll = ICFinal[:, 11].astype(int)
ValuePosid = ICFinal[:, 9].astype(int)
EIsoidAll = ICFinal[:, 13].astype(int)

PathE = [
    pathFinalE2ValIn[eid][vpid][:, :isoid]
    for eid, vpid, isoid in zip(EidAll, ValuePosid, EIsoidAll)
]

profiler.stop()

# ============================================
# 生成最终报告
# ============================================
profiler.report()
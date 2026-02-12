import sys
import os
import numpy as np
import time
import joblib  # 必须先安装 pip install joblib
import gc
# 统一保存函数
def fast_save(obj, name_, TimeIso_):
    # 使用 .jbl 后缀区分普通 pickle
    filename = f"{name_}{TimeIso_}.jbl"
    print(f"正在压缩保存 {filename} ...")
    joblib.dump(obj, filename, compress=3) 

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调用转换后的函数
import pickle
import argparse
from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainMapP,WH_main_obtainMapRef,WH_main_obtainIso


# parser = argparse.ArgumentParser()
# parser.add_argument('--nodraw', action='store_true', help='如果指定则不进行绘图，直接返回结果')
# args = parser.parse_args()
TimeMap='_0211_2245'
TimeIso=TimeMap

Map = joblib.load('Map'+TimeMap+'.jbl')	

Map, IsoMapP_i_tt, pathFinalP = WH_main_obtainMapP(Map, draw=False, draw_interactive=False)
print("IsoMapP_i_tt")

fast_save(IsoMapP_i_tt, 'IsoMapP_i_tt',TimeMap)
fast_save(pathFinalP, 'pathFinalP',TimeMap)
del IsoMapP_i_tt, pathFinalP
gc.collect()

# 严格保留变量名
Trans_Point_verse = Map['Trans_Point'].copy()

# 计算 angles
# MATLAB: Trans_Point_verse(:,2) 是第2列 -> Python: [:, 1]
# MATLAB: Trans_Point_verse(:,1) 是第1列 -> Python: [:, 0]
angles = np.arctan2(0 - Trans_Point_verse[:, 1], Map['obs_mapsize_x'][1] / 2 - Trans_Point_verse[:, 0])

# 将角度赋值给第3列 (索引为 2)
if Trans_Point_verse.shape[1] < 3:
    # 如果原始只有2列，通过 column_stack 扩展到3列
    Trans_Point_verse = np.column_stack((Trans_Point_verse, angles))
else:
    # 如果已有3列或更多，直接覆盖第3列
    Trans_Point_verse[:, 2] = angles

# 第一组计算：从 Trans_Point_verse 到 Map.ValuePos (flagAll=1)
Map, final_pathTP2Val, vthetaAllTP2Val = WH_main_obtainMapRef(
    Map, Trans_Point_verse, Map['ValuePos'], 1
)

pathFinalTP2Val, IsoMapTP2Val_i_tt,_ = WH_main_obtainIso(
    Map, Map['v_E'], final_pathTP2Val, vthetaAllTP2Val, 0
)
print('IsoMapTP2Val_i_tt')

fast_save(IsoMapTP2Val_i_tt, 'IsoMapTP2Val_i_tt',TimeMap)
fast_save(pathFinalTP2Val, 'pathFinalTP2Val',TimeMap)
del IsoMapTP2Val_i_tt, pathFinalTP2Val,final_pathTP2Val
gc.collect()
# 第二组计算：从 Map.Trans_Point 到 Map.Trans_Point (flagAll=0, 对应 i!=j 逻辑)
Map, final_pathPTP2Iso, vthetaAllPTP2Iso = WH_main_obtainMapRef(
    Map, Map['Trans_Point'], Map['Trans_Point'], 0
)

pathFinalMapPTP2Iso, IsoMapPTP2Iso_i_tt,IsoMapPIso2TP_i_tt = WH_main_obtainIso(
    Map, Map['v_E'], final_pathPTP2Iso, vthetaAllPTP2Iso, 1
)
print('IsoMapPTP2Iso_i_tt')

fast_save(IsoMapPTP2Iso_i_tt, 'IsoMapPTP2Iso_i_tt',TimeMap)
fast_save(IsoMapPIso2TP_i_tt, 'IsoMapPIso2TP_i_tt',TimeMap)
fast_save(pathFinalMapPTP2Iso, 'pathFinalMapPTP2Iso',TimeMap)
fast_save(pathFinalMapPTP2Iso, 'pathFinalMapPTP2Iso',TimeMap)
del IsoMapPTP2Iso_i_tt,IsoMapPIso2TP_i_tt, pathFinalMapPTP2Iso,final_pathPTP2Iso
gc.collect()

# 第三组计算：从 Trans_Point_verse 到 Trans_Point_verse (flagAll=0, 对应 i!=j 逻辑)
Map, final_pathETP2Iso, vthetaAllETP2Iso = WH_main_obtainMapRef(
    Map, Trans_Point_verse, Trans_Point_verse, 0
)

pathFinalMapETP2Iso, IsoMapETP2Iso_i_tt,IsoMapEIso2TP_i_tt = WH_main_obtainIso(
    Map, Map['v_E'], final_pathETP2Iso, vthetaAllETP2Iso, 1
)
print('IsoMapETP2Iso_i_tt')

fast_save(IsoMapEIso2TP_i_tt, 'IsoMapEIso2TP_i_tt',TimeMap)
fast_save(IsoMapETP2Iso_i_tt, 'IsoMapETP2Iso_i_tt',TimeMap)
fast_save(pathFinalMapETP2Iso, 'pathFinalMapETP2Iso',TimeMap)
del IsoMapETP2Iso_i_tt,IsoMapEIso2TP_i_tt, pathFinalMapETP2Iso,final_pathETP2Iso
gc.collect()
fast_save(Map, 'Map',TimeMap)
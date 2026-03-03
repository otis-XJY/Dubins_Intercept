import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调用转换后的函数
import pickle
import argparse
from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainMapP,WH_main_obtainMapRef,WH_main_obtainIso
from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm
from Draw.Draw_pathA_no_circle_final_path import Draw_pathA_no_circle_final_path
from Draw.Draw_map import Draw_map
from intercept.IsoMap.obtainPath import obtainPath
from intercept.IsoMap.DrawIso import draw_iso
# parser = argparse.ArgumentParser()
# parser.add_argument('--nodraw', action='store_true', help='如果指定则不进行绘图，直接返回结果')
# args = parser.parse_args()
import joblib 
TimeMap='_0302_1940'
TimeIso=TimeMap
Map = joblib.load('Map'+TimeMap+'.jbl')	
      

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

print(Trans_Point_verse[8, :])
print(Trans_Point_verse[9, :])
path_segments, _, _,_= A_dubins_nocircle_swarm(
    Map['Trans_Point'][23, :], 
    Map['Trans_Point'][24,:], 
    Map['obs'], 
    Map['sure'], 
    Map['r'], 
    Map['obs_no_circle'], 
    Map['outline_all'], 
    Map['Stepsize'], 
    1, 
    Map['resolution'],
    depth=0,max_depth=1
)

vthetaAll_=[]
# 提取路径中的速度信息（vtheta_all 表示路径中每个点的速度角度）
for tt in range(len(path_segments) - 1):
    for k in range(len(path_segments[tt + 1]['vtheta_all'])):
        vthetaAll_.extend(path_segments[tt + 1]['vtheta_all'][k])
vthetaAllP = vthetaAll_

pathFinalP = obtainPath(path_segments, vthetaAllP, 0) 


plt.figure(figsize=(10, 8))
Draw_map(Map['PStart_Point'], Map['Trans_Point'], Map['ValuePos'], Map['obs'], Map['sure'], Map['obs_no_circle'], Map['obs_no_circle_in'])
plt.plot(pathFinalP[0,:], pathFinalP[1,:], 'r-', markersize=10)
# draw_iso(Map, path_segments, pathFinalP, vthetaAllP, interactive=1)

# Draw_pathA_no_circle_final_path(path_segments, 0,ax=None)
plt.show()
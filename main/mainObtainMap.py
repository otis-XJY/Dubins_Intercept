import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('TkAgg')  # 强制使用TkAgg后端
import numpy as np
import matplotlib.pyplot as plt
from intercept.Map.obtainMap import obtainMap

import joblib  # 必须先安装 pip install joblib
import gc
# 统一保存函数
def fast_save(obj, name_, TimeIso_):
    # 使用 .jbl 后缀区分普通 pickle
    filename = f"{name_}{TimeIso_}.jbl"
    print(f"正在压缩保存 {filename} ...")
    joblib.dump(obj, filename, compress=3) 


import time

# 初始化 Map 字典
Map = {}


Map['numTime'] = 50  # 时间分割数量

# 设置路径规划相关参数
Map['r'] = 10  # 最小转弯半径
Map['Stepsize'] = 0.01  # 路径生成间隔
Map['sure'] = 20  # 障碍物的安全预留距离

# 设置障碍物的区域大小
Map['obs_mapsize_x'] = [0,1000 * 2]
Map['obs_mapsize_y'] = [250,1000 * 2]

Map['mapsize_x'] = 1000 * 2
Map['mapsize_y'] = 1000 * 2

# 障碍物参数
Map['R'] = [100, 120]  # 障碍物半径范围
Map['num_obs_nocircle'] = 10  # 多边形随机障碍物数量
Map['num_steps'] = 100  # 边界上几个点

# 设施点位置
numValue = 3
ValuePos_x = np.linspace(10 * Map['sure'], Map['mapsize_x'] - 10 * Map['sure'], numValue)
ValuePos = np.column_stack([ValuePos_x, 2 * Map['sure'] * np.ones(numValue), -np.ones(numValue) * np.pi / 2])
Map['ValuePos'] = ValuePos

# UAV 起始位置
numUav = 3
Start_Point_x = np.linspace(5 * Map['sure'], Map['mapsize_x'] - 5 * Map['sure'], numUav)
PStart_Point = np.column_stack([Start_Point_x, 2.5 * Map['sure'] * np.ones(numUav), np.ones(numUav) * np.pi / 2])
Map['PStart_Point'] = PStart_Point

# 设置地图分辨率
Map['resolution_map_pos'] = [5, 5]
Trans_Point_x = np.linspace(Map['mapsize_x'] / Map['resolution_map_pos'][0] / 2,
                             Map['mapsize_x'] - Map['mapsize_x'] / Map['resolution_map_pos'][0] / 2,
                             Map['resolution_map_pos'][0])
Trans_Point_y = np.linspace(Map['mapsize_y'] / Map['resolution_map_pos'][1] / 2,
                             Map['mapsize_y'] - Map['mapsize_y'] / Map['resolution_map_pos'][1] / 2,
                             Map['resolution_map_pos'][1])

# 生成网格点
End_Point_X, End_Point_Y = np.meshgrid(Trans_Point_x, Trans_Point_y)

# 转换为 N×2 的矩阵（每行是一个点的 [x, y]）
angles = np.arctan2(Map['mapsize_y'] - End_Point_Y.ravel(), Map['mapsize_x'] / 2 - End_Point_X.ravel())
Trans_Point = np.column_stack([End_Point_X.ravel(), End_Point_Y.ravel(), angles])
Map['Trans_Point'] = Trans_Point

# 设置终点位置
# resolution_map_pos = 10
# E_TranPoint_x = np.linspace(0, Map['mapsize_x'], resolution_map_pos)
# E_TranPoint_y = np.ones(resolution_map_pos) * Map['mapsize_y'] - 2 * Map['r']
# E_TranPoint = np.column_stack([E_TranPoint_x, E_TranPoint_y])
# Map['E_TranPoint'] = E_TranPoint

# 调用 obtainMap 函数
Map = obtainMap(Map, n_topo=15, n_blank=15, safety=20.0)
TimeMap='_0305_1800'
SaveName = 'Map'+TimeMap

fast_save(Map, 'Map',TimeMap)

print('地图保存完成')

plt.savefig(SaveName, dpi=300)
# 显示所有绘制的图形
plt.show()
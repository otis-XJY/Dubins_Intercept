import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
# 设置matplotlib后端
matplotlib.use('QtAgg') 

# 导入必要的模块
import numpy as np
import matplotlib.pyplot as plt
from Draw.Draw_pathA_no_circle_final_path import Draw_pathA_no_circle_final_path
from Draw.Draw_map import Draw_map


import pickle

# 尝试从项目根目录加载 Map 和 pathFinalP（由 main 脚本保存为 pkl）
Map = None
pathFinalP = None
map_pkl = os.path.join(project_root, 'Map.pkl')
path_pkl = os.path.join(project_root, 'pathFinalP.pkl')

if os.path.exists(map_pkl):
    with open(map_pkl, 'rb') as f:
        Map = pickle.load(f)

if os.path.exists(path_pkl):
    with open(path_pkl, 'rb') as f:
        pathFinalP = pickle.load(f)

# 如果没有找到 Map，尝试从 main.mainObtainMap 导入已初始化的 Map（若该模块已构建）
if Map is None:
    try:
        from main.mainObtainMap import Map as _Map_from_main
        Map = _Map_from_main
    except Exception:
        pass

plt.figure(figsize=(10, 8))

# 检查 Map 是否有效
if Map is None:
    raise RuntimeError("Map is not available. Run 'main/mainObtainMap.py' or save 'Map.pkl' to the project root before running this debug script.")

# 检查必需键
required_keys = ['PStart_Point', 'Trans_Point', 'ValuePos', 'sure', 'obs_no_circle', 'obs_no_circle_in']
missing = [k for k in required_keys if k not in Map]
if missing:
    raise KeyError(f"Map is missing required keys: {missing}")

# 绘制地图
Draw_map(Map['PStart_Point'], Map['Trans_Point'], Map['ValuePos'], Map.get('obs', None), Map['sure'], Map['obs_no_circle'], Map['obs_no_circle_in'])

# 正确调用绘图函数（如果有 pathFinalP）
if pathFinalP is None:
    print("Warning: 'pathFinalP' not found. Skipping path drawing. Create 'pathFinalP.pkl' by running 'main/main0901.py' to save the result.")
else:
    Draw_pathA_no_circle_final_path(pathFinalP[0,0], 0)

plt.show()

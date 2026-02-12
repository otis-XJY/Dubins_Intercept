import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import joblib
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调用转换后的函数
import pickle
import argparse
from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainMapP,WH_main_obtainMapRef,WH_main_obtainIso
from Draw.Draw_map import Draw_map


def fast_save(obj, name_, TimeIso_):
    # 使用 .jbl 后缀区分普通 pickle
    filename = f"{name_}{TimeIso_}.jbl"
    print(f"正在压缩保存 {filename} ...")
    joblib.dump(obj, filename, compress=3) 
	


# parser = argparse.ArgumentParser()
# parser.add_argument('--nodraw', action='store_true', help='如果指定则不进行绘图，直接返回结果')
# args = parser.parse_args()
TimeMap='_0211_2245'
TimeIso='_0212_1200'

Map = joblib.load('Map'+TimeMap+'.jbl')	
# with open('Map'+TimeMap+'.pkl', 'rb') as f:
# 	Map = pickle.load(f)


# 1. 变量提取（假设 Map 是一个字典）
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
ValuePos=Map['ValuePos']
PStart_Point=Map['PStart_Point']

# 2. 定义 Evader 和 目标 id
Evader = np.array([
    [200, 1950, -np.pi/2],
    [1000, 1950, -np.pi/2],
    [1800, 1950, -np.pi/2]
])

# MATLAB id = [3, 2, 1]，对应 Python 索引为 [2, 1, 0]
id_list = np.array([3, 2, 1])
ValuePosOb = ValuePos[id_list - 1, :] # 注意：ValuePos 需要预先定义

# 3. 绘图准备
fig, ax = plt.subplots(figsize=(10, 8))
# 假设 Draw_map 已经转化为了 Python 函数
Draw_map(PStart_Point, Trans_Point, ValuePosOb, obs, sure, obs_no_circle, obs_no_circle_in)

ax.plot(Evader[:, 0], Evader[:, 1], 'kd', linewidth=2, label='Evader')

# 设置起点 S 和 终点 E
S = Evader[:, 0:2]
E = ValuePosOb[:, 0:2]

PathE2Val_true = {} # 存储路径的字典

# 4. 路径选取与插值循环
for i in range(S.shape[0]):
    print(f'请为第 {i+1} 条路径选取中间点（左键选择，右键或回车结束）')
    
    # 绘制当前起终点
    ax.plot(S[i, 0], S[i, 1], 'gx', markersize=8, markeredgewidth=1.5)
    ax.plot(E[i, 0], E[i, 1], 'gx', markersize=8, markeredgewidth=1.5)
    plt.draw()

    # 鼠标点选中间点 (matplotlib 的 ginput)
    # n=-1 表示不限点数，直到按回车或右键
    # 注意：在某些 IDE（如 PyCharm）中 ginput 可能需要开启弹窗模式
    pts = plt.ginput(n=-1, timeout=-1, show_clicks=True)
    MidPts = np.array(pts) if pts else np.empty((0, 2))

    # 构建完整路径点序列：S -> MidPts -> E
    if MidPts.size > 0:
        PathPts = np.vstack([S[i, :], MidPts, E[i, :]])
    else:
        PathPts = np.vstack([S[i, :], E[i, :]])

    SmoothPath_list = []

    # 5. 线性插值生成 SmoothPath
    for j in range(PathPts.shape[0] - 1):
        p1 = PathPts[j, :]
        p2 = PathPts[j+1, :]
        
        # 计算距离和需要的点数
        dist = np.linalg.norm(p2 - p1)
        N = int(max(2, np.ceil(dist / Stepsize)))
        
        # 生成插值点
        x = np.linspace(p1[0], p2[0], N)
        y = np.linspace(p1[1], p2[1], N)
        
        # 拼接点（避免重复添加分段连接处点）
        segment = np.column_stack((x, y))
        if j == 0:
            SmoothPath_list.append(segment)
        else:
            SmoothPath_list.append(segment[1:]) # 跳过首点，防止重复

    # 合并为最终矩阵
    SmoothPath = np.vstack(SmoothPath_list)

    # 6. 计算路径方向 (朝向角)
    delta = np.diff(SmoothPath, axis=0)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    
    # 保持数量一致，首位补上 Evader 的初始角度 (i, 2)
    # MATLAB: thetaAll = [Evader(i,3); theta]
    # 如果 SmoothPath 有 N 个点，diff 后 theta 有 N-1 个点，补一个后回到 N 个点
    thetaAll = np.concatenate(([Evader[i, 2]], theta))

    # 存储结果：[x, y, theta]
    # 注意 thetaAll 需要转置为列向量
    PathE2Val_true[i] = np.column_stack((SmoothPath, thetaAll))

    # 可视化生成的路径
    ax.plot(SmoothPath[:, 0], SmoothPath[:, 1], 'b.-', linewidth=1.5)
    plt.draw()

print("路径采集完成。")

# import pickle
# with open('PathE2Val_true'+TimeIso+'.pkl', 'wb') as f:
# 	pickle.dump(PathE2Val_true, f)

fast_save(PathE2Val_true, 'PathE2Val_true',TimeIso)
print('路径采集完成')

# 5. 调用路径获取函数
# 计算从 Evader 起点到 ValuePos 的所有路径 (flagAll=1)
# 假设 ValuePos 已经在当前作用域中定义
Map, final_pathE2Val, vthetaAllE2Val = WH_main_obtainMapRef(
    Map, Evader, ValuePos, 1
)

# 6. 调用等时线生成函数
# 注意：最后一个参数 verse = 0
pathFinalE2ValIn, IsoMapE2ValIn_i_tt,_ = WH_main_obtainIso(
    Map, v_E, final_pathE2Val, vthetaAllE2Val, 0
)

# 打印或后续处理

# with open('IsoMapE2ValIn_i_tt'+TimeIso+'.pkl', 'wb') as f:
# 	pickle.dump(IsoMapE2ValIn_i_tt, f)

fast_save(IsoMapE2ValIn_i_tt, 'IsoMapE2ValIn_i_tt',TimeIso)

# with open('pathFinalE2ValIn'+TimeIso+'.pkl', 'wb') as f:
# 	pickle.dump(pathFinalE2ValIn, f)
     
fast_save(pathFinalE2ValIn, 'pathFinalE2ValIn',TimeIso)
print("IsoMapE2ValIn_i_tt 计算完成")
plt.show()
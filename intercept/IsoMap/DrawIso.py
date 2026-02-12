import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Draw.Draw_map import Draw_map
from Draw.Draw_pathA_no_circle_final_path import Draw_pathA_no_circle_final_path


def draw_iso(Map, final_pathUAV, pathFinalP, vthetaAllP, interactive=False, pause_time=0.001):
    """绘制 Iso 图：
    Map: 地图字典，包含 PStart_Point, Trans_Point, ValuePos, sure, obs_no_circle, obs_no_circle_in 等
    final_pathUAV: 由规划器返回的原始路径字典，键为 (i,j)
    pathFinalP: 处理后可直接绘制的路径字典，键为 (i,j)
    vthetaAllP: 每条路径的速度角数据字典
    interactive: 是否以交互方式逐条绘制（模拟原脚本的实时绘图），默认为 False
    pause_time: 交互绘制时每条路径绘制后的暂停时间
    """

    PStart_Point = Map['PStart_Point']
    Trans_Point = Map['Trans_Point']
    ValuePos = Map['ValuePos']
    sure = Map['sure']
    obs_no_circle = Map['obs_no_circle']
    obs_no_circle_in = Map['obs_no_circle_in']

    N = len(PStart_Point[:, 1])
    colors = cm.viridis(np.linspace(0, 1, N))

    # 交互式（或逐条）绘制所有路径
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    Draw_map(PStart_Point, Trans_Point, ValuePos, [], sure, obs_no_circle, obs_no_circle_in, ax=ax1)

    if interactive:
        plt.ion()

    for i in range(len(PStart_Point[:, 1])):
        for j in range(len(Trans_Point[:, 1])):
            key = (i, j)
            if key in pathFinalP:
                try:
                    ax1.plot(pathFinalP[i, j][0, :], pathFinalP[i, j][1, :], color=colors[i], linewidth=2)
                except Exception:
                    # 忽略不能绘制的路径
                    continue
                if interactive:
                    plt.draw()
                    plt.pause(pause_time)

    if interactive:
        plt.ioff()

    # 最终绘图（使用已有绘图函数绘制最终路径形态）
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    Draw_map(PStart_Point, Trans_Point, ValuePos, [], sure, obs_no_circle, obs_no_circle_in, ax=ax2)

    for i in range(len(PStart_Point[:, 1])):
        for j in range(len(Trans_Point[:, 1])):
            key = (i, j)
            if key in final_pathUAV:
                try:
                    Draw_pathA_no_circle_final_path(final_pathUAV[i, j], 0, ax=ax2)
                except Exception:
                    continue

    plt.show()


if __name__ == '__main__':
    # 作为独立脚本运行时，可简单地从 Map.pkl 读取 Map 并绘图（如果存在）
    import os
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='Draw Iso maps from saved Map and path data (if available).')
    parser.add_argument('--map', default='Map.pkl', help='Path to Map pickle file containing the Map dict')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive incremental drawing')
    args = parser.parse_args()

    if os.path.exists(args.map):
        with open(args.map, 'rb') as f:
            Map = pickle.load(f)
        # 如果外部保存了路径数据，可以扩展以加载这些文件；否则仅绘制地图底图
        # 尝试读取可能的路径文件
        final_pathUAV = {}
        pathFinalP = {}
        vthetaAllP = {}
        draw_iso(Map, final_pathUAV, pathFinalP, vthetaAllP, interactive=args.interactive)
    else:
        print('没有找到 Map 文件：', args.map)

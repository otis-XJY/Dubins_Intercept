import numpy as np
import matplotlib.pyplot as plt


def Draw_pathA_no_circle_final_path(close, id_flag,ax=None):
    """
    绘制无圆障碍物情况下的最终路径 (对应 MATLAB 版本 Draw_pathA_no_circle_final_path)

    参数:
        close: list(dict)，存储路径信息，每个元素类似 MATLAB 的结构体
               需要包含:
               - point: ndarray (2, N)
               - path: list of ndarray (2, M)
               - center: ndarray (2, K)
               - r: ndarray 半径数组
               - R: float 半径
               - vtheta: float
               - vtheta_all: list of list/array
        id_flag: int，控制是否绘制圆心和圆
    """

    if ax is None:
        ax = plt.gca()
        
    num = len(close)

    # 遍历路径段
    for i in range(1, num - 1):
        if id_flag == 1:
            for j in range(1, close[i]["point"].shape[1]):
                ax.plot(close[i]["point"][0, j],
                         close[i]["point"][1, j],
                         '.',
                         markersize=15,
                         color='r')

                # 可选：绘制速度方向
                # vtheta = close[i]["vtheta_all"][j-1][-1]
                # ax.quiver(close[i]["point"][0, j],
                #            close[i]["point"][1, j],
                #            15*np.cos(vtheta),
                #            15*np.sin(vtheta),
                #            color='r', linewidth=1)

        # 绘制路径
        for lee in range(len(close[i]["path"])):
            ax.plot(close[i]["path"][lee][0], close[i]["path"][lee][1], linewidth=2)

        # 绘制圆心和圆
        if id_flag == 1:
            for k in range(0, close[i]["center"].shape[1]-1, 2):
                theta = np.linspace(0, 2*np.pi, 200)

                edge1x = close[i]["center"][0, k] + close[i]["r"][0] * np.cos(theta)
                edge1y = close[i]["center"][1, k] + close[i]["r"][0] * np.sin(theta)
                ax.plot(close[i]["center"][0, k], close[i]["center"][1, k], 'o')
                ax.plot(edge1x, edge1y, linestyle='--', color='r')

                if k+1 < close[i]["center"].shape[1]:
                    edge2x = close[i]["center"][0, k+1] + close[i]["r"][1] * np.cos(theta)
                    edge2y = close[i]["center"][1, k+1] + close[i]["r"][1] * np.sin(theta)

                    ax.plot(close[i]["center"][0, k+1], close[i]["center"][1, k+1], '^')
                    ax.plot(edge2x, edge2y, linestyle='--', color=(0, 0.5, 1), marker='*')

    # 最后一段
    if id_flag == 1:
        for j in range(1, close[num-1]["point"].shape[1]):
            ax.plot(close[num-1]["point"][0, j],
                     close[num-1]["point"][1, j],
                     '.',
                     markersize=15,
                     color='r')

    for lee in range(len(close[num-1]["path"])):
        ax.plot(close[num-1]["path"][lee][0], close[num-1]["path"][lee][1], linewidth=2)

    if id_flag == 1:
        theta = np.linspace(0, 2*np.pi, 200)
        edge1x = close[num-1]["center"][0, 0] + close[num-1]["r"][0] * np.cos(theta)
        edge1y = close[num-1]["center"][1, 0] + close[num-1]["r"][0] * np.sin(theta)
        edge2x = close[num-1]["center"][0, 1] + close[num-1]["r"][1] * np.cos(theta)
        edge2y = close[num-1]["center"][1, 1] + close[num-1]["r"][1] * np.sin(theta)

        ax.plot(close[num-1]["center"][0, 0], close[num-1]["center"][1, 0], 'o')
        ax.plot(close[num-1]["center"][0, 1], close[num-1]["center"][1, 1], 'o')
        ax.plot(edge1x, edge1y, linestyle='--', color='r')
        ax.plot(edge2x, edge2y, linestyle='--', color='b')

    if id_flag == 1:
        ax.plot(close[0]["point"][0, 0],
                 close[0]["point"][1, 0],
                 '.',
                 markersize=15,
                 color='r')

        ax.quiver(close[0]["point"][0, 0],
                   close[0]["point"][1, 0],
                   100*np.cos(close[0]["vtheta"]),
                   100*np.sin(close[0]["vtheta"]),
                   angles='xy', scale_units='xy', scale=1,
                   color='r', linewidth=2)

    ax.grid(True)
    ax.set_frame_on(True)
    # ax.show()

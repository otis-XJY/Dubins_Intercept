import numpy as np


def obtain_obs_to_avoid(Start_Point, End_Point, outline_all):
    x0, y0 = Start_Point[0], Start_Point[1]  # 起点坐标
    m = (End_Point[1] - Start_Point[1]) / (End_Point[0] - Start_Point[0])  # 斜率

    a = m
    b = -1
    c = y0 - m * x0

    k_vertical = b / a
    a1 = k_vertical
    b1 = -1
    c1 = Start_Point[1] - k_vertical * Start_Point[0]

    x = np.arange(Start_Point[0], End_Point[0], 0.5)

    # 计算每个障碍物到直线的距离
    distances = (a1 * outline_all[:, 0] + b1 * outline_all[:, 1] + c1) / np.sqrt(a1 ** 2 + b1 ** 2)

    singal = (a1 * End_Point[0] + b1 * End_Point[1] + c1) / np.sqrt(a1 ** 2 + b1 ** 2)

    if singal > 0:
        obs_id_all = outline_all[distances >= 0, 2]
    else:
        obs_id_all = outline_all[distances < 0, 2]

    obs_to_avoid = np.unique(obs_id_all)

    return obs_to_avoid

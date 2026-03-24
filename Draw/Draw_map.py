import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt

def Draw_map(startpoint, endpoint, ValuePos, obs, sure, obs_no_circle, obs_no_circle_in, ax=None):
    """
    绘制地图，标注起点、终点、设施点和障碍物。

    startpoint: 起点坐标 [x, y, theta]
    endpoint: 终点坐标 [x, y]
    ValuePos: 设施点坐标
    obs: 障碍物信息 [x, y, radius]
    sure: 确保边界
    obs_no_circle: 没有圆形障碍物的区域
    obs_no_circle_in: 具有圆形障碍物的区域
    ax: matplotlib axes对象，如果为None则使用当前axes
    """
    if ax is None:
        ax = plt.gca()

    # 绘制起点、终点和设施点
    ax.scatter(startpoint[:, 0], startpoint[:, 1], marker='*', color='k', linewidth=2, label="Start Point")
    ax.scatter(endpoint[:, 0], endpoint[:, 1], marker='*', color='r', linewidth=2, label="End Point")
    ax.plot(ValuePos[:, 0], ValuePos[:, 1], 'rp', linewidth=2, label="Facility Points")

    # 绘制起点朝向的箭头
    for sp in startpoint:
        ax.quiver(sp[0], sp[1], 200 * np.cos(sp[2]), 200 * np.sin(sp[2]), angles='xy', scale_units='xy', scale=1,
                   color='r', linewidth=2)

    # 标注起点和终点编号
    for i, sp in enumerate(startpoint):
        ax.text(sp[0], sp[1], f'UAV{i}', fontsize=12)

    for i, ep in enumerate(endpoint):
        ax.text(ep[0], ep[1], f'{i}', fontsize=12)

    # 绘制圆形障碍物
    if obs is not None:
        for i,o in enumerate(obs):
            theta = np.linspace(0, 2 * np.pi, 100)
            x1 = o[0] + (o[2] - sure) * np.cos(theta)
            y1 = o[1] + (o[2] - sure) * np.sin(theta)
            ax.plot(x1, y1, linestyle='--', color='k')

            x2 = o[0] + o[2] * np.cos(theta)
            y2 = o[1] + o[2] * np.sin(theta)
            ax.plot(x2, y2, linestyle='--', color='c')

            ax.plot(o[0], o[1], 'o')
            ax.text(o[0], o[1], f'{i}', fontsize=12)

    # 绘制没有圆形障碍物的区域
    # k = 0
    for i,no_circle in enumerate(obs_no_circle):
        # 添加空值检查，如果no_circle为空则跳过
        if no_circle is None:
            # k += 1
            continue
        x = no_circle[:, 0]
        y = no_circle[:, 1]
        ax.plot(x, y, 'b-', linewidth=2)
        ax.text(np.mean(x), np.mean(y), str(i), fontsize=12)
        # k += 1

    # 绘制具有圆形障碍物的区域
    for no_circle_in in obs_no_circle_in:
        # 添加空值检查，如果no_circle_in为空则跳过
        if no_circle_in is None:
            continue
        x = no_circle_in[:, 0]
        y = no_circle_in[:, 1]
        ax.plot(x, y, 'r-', linewidth=2)

    # 设置图像属性
    ax.grid(True)
    ax.set_frame_on(True)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('MAP', fontsize=14)
    ax.legend()
    ax.axis('equal')
    return ax
    # 不在这里调用plt.show()，让调用者决定何时显示
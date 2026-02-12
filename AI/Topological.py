import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. 核心拓扑函数：计算绕数 (Winding Number) ---
def calculate_winding_number(path, obstacle):
    """
    计算一段路径相对于某个障碍物的绕数
    path: [N, 2] 的坐标阵
    obstacle: [x, y] 障碍物坐标
    """
    if len(path) < 2:
        return 0
    
    # 转化为相对坐标
    rel_path = path - obstacle
    # 计算每个时刻的极角
    angles = np.arctan2(rel_path[:, 1], rel_path[:, 0])
    # 计算角度差，并处理 [-pi, pi] 的跳变
    diffs = np.diff(angles)
    diffs[diffs > np.pi] -= 2 * np.pi
    diffs[diffs < -np.pi] += 2 * np.pi
    
    return np.sum(diffs) / (2 * np.pi)

# --- 2. 仿真参数设置 ---
dt = 0.1
steps = 200
obstacles = np.array([[0, 0.5], [-0.5, -0.5], [0.5, -0.5]]) # 三个障碍物

# 初始位置
pos_p = np.array([-1.5, 0.0]) # 拦截者 Pursuer
pos_e = np.array([1.5, 0.0])  # 入侵者 Evader
v_max_p = 0.15
v_max_e = 0.12

# 轨迹存储
path_p = [pos_p.copy()]
path_e = [pos_e.copy()]
winding_history = []

# --- 3. 仿真循环 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def update(frame):
    global pos_p, pos_e
    
    # A. 入侵者逻辑：试图向左逃逸，同时避开障碍物
    goal_e = np.array([-2.0, 0.0])
    dir_e = goal_e - pos_e
    # 简单的势场避障
    for obs in obstacles:
        dist = np.linalg.norm(pos_e - obs)
        if dist < 0.5:
            dir_e += (pos_e - obs) / (dist**3 + 0.01)
    
    pos_e += (dir_e / np.linalg.norm(dir_e)) * v_max_e * dt
    path_e.append(pos_e.copy())

    # B. 拦截者逻辑：拓扑偏置控制 (Topological-Biased Control)
    # 不仅仅追逐目标，还要根据目标相对于中心障碍物的绕数，选择绕行方向
    pure_pursuit_dir = pos_e - pos_p
    
    # 计算入侵者相对于中心障碍物的瞬时趋势
    # 核心 Idea：如果发现目标正在顺时针绕行，拦截者尝试逆时针包抄，形成“拓扑对撞”
    center_obs = obstacles[0]
    vec_e = pos_e - center_obs
    vec_p = pos_p - center_obs
    
    # 拓扑引导力：强制产生一个与目标相反的环绕速度
    topo_force = np.array([-vec_p[1], vec_p[0]]) # 法向力
    
    # 混合策略：80% 追踪 + 20% 拓扑占位
    dir_p = 0.8 * pure_pursuit_dir / np.linalg.norm(pure_pursuit_dir) + \
            0.2 * topo_force / np.linalg.norm(topo_force)
    
    pos_p += (dir_p / np.linalg.norm(dir_p)) * v_max_p * dt
    path_p.append(pos_p.copy())

    # C. 计算当前的拓扑状态（绕数）
    curr_path_e = np.array(path_e)
    wn = calculate_winding_number(curr_path_e, center_obs)
    winding_history.append(wn)

    # --- 绘图更新 ---
    ax1.clear()
    # 画障碍物
    ax1.scatter(obstacles[:,0], obstacles[:,1], c='black', s=200, label='Obstacles')
    # 画轨迹
    p_arr = np.array(path_p)
    e_arr = np.array(path_e)
    ax1.plot(p_arr[:,0], p_arr[:,1], 'blue', label='Interceptor (P)')
    ax1.plot(e_arr[:,0], e_arr[:,1], 'red', label='Evader (E)')
    # 画当前位置
    ax1.scatter(pos_p[0], pos_p[1], c='blue', s=100)
    ax1.scatter(pos_e[0], pos_e[1], c='red', s=100)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title(f"Space View: Topological Braiding Interception\nStep: {frame}")
    ax1.legend()
    ax1.grid(True)

    ax2.clear()
    ax2.plot(winding_history, color='green', lw=2)
    ax2.set_title("Topological Invariant: Winding Number")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Winding Number around Center Obstacle")
    ax2.grid(True)

# 运行仿真
ani = FuncAnimation(fig, update, frames=steps, interval=50, repeat=False)
plt.tight_layout()
plt.show()
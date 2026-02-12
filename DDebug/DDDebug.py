import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from Draw.Draw_map import Draw_map

# 配置参数
param_all_plot = param_all_start[0] # 或者 param_all_1/param_all_2，确保数据格式正确
plot_type = 1
save_name = "debug_plot_fixed.png"

# 定义公共点
Point = np.stack((Start_Point, End_Point))

if plot_type == 2:
    # --- 情况 1：单条路径 ---
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    Draw_map(Point, Point, Point, [], 3, obs_no_circle, obs_no_circle, ax=ax)
    
    # 强制设置等比例缩放（单图不受限）
    ax.set_aspect('equal')
    
    segments = param_all_plot['path']
    N = len(segments)
    colors = cm.plasma(np.linspace(0, 1, N))

    for i in range(N):
        x = segments[i][0]
        y = segments[i][1]
        plt.plot(x, y, '-', color=colors[i], linewidth=2, alpha=0.8)
    
    plt.plot(param_all_plot['point'][0,:], param_all_plot['point'][1,:], 'ko', markersize=6)
    plt.title("Single Path Multi-Segments")

else:
    # --- 情况 2：集群路径 (处理 RuntimeError) ---
    num_paths = len(param_all_plot)
    cols = 3
    rows = int(np.ceil(num_paths / cols))
    
    # 关键点 1: 共享轴时，后面需要配合 adjustable='box'
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), 
                             sharex=True, sharey=True)
    axes = axes.flatten()

    colors = cm.viridis(np.linspace(0, 1, num_paths))

    for i in range(num_paths):
        ax = axes[i]
        
        # 绘制地图
        Draw_map(Point, Point, Point, [], 3, obs_no_circle, obs_no_circle, ax=ax)
        
        # 关键点 2: 解决 RuntimeError 的核心代码
        # 强制设置 aspect 为 'equal'，但必须指定 adjustable='box' 以支持共享轴
        ax.set_aspect('equal', adjustable='box')
        
        # 关键点 3: 安全地提取数据，防止 KeyError
        # 如果 param_all_plot 是列表，使用索引；如果是字典且以数字为键，也支持
        try:
            path_struct = param_all_plot[i]
        except (KeyError, IndexError):
            # 处理可能的字典键名不是连续整数的情况
            keys = list(param_all_plot.keys())
            path_struct = param_all_plot[keys[i]]

        # 绘图逻辑
        if 'path' in path_struct:
            for k in range(len(path_struct['path'])):
                px = path_struct['path'][k][0]
                py = path_struct['path'][k][1]
                ax.plot(px, py, '-', color=colors[i], linewidth=1.5)
            
            if 'point' in path_struct:
                ax.plot(path_struct['point'][0,:], path_struct['point'][1,:], 'k.', markersize=4)
        
        ax.set_title(f"Interception Path {i}")

    # 隐藏多余的空子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Swarm Paths Individual View (Fixed Aspect Ratio)", fontsize=16)
    
    # 关键点 4: tight_layout 在处理共享轴 + aspect('equal') 时容易报错
    # 建议使用 rect 留出顶部标题空间
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(save_name, dpi=300)
plt.show()
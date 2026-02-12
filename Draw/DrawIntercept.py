import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# 3. 绘图逻辑 (对应 MATLAB 注释部分)
def draw_candidates(IC_Plot, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt, pathFinalE2ValIn, pathFinalP2TP,ax=None):
    """
    可视化前 100 个拦截方案
    """
    if ax is None:
        ax = plt.gca()
    
    if IC_Plot.size == 0:
        print("没有可用的拦截候选方案进行绘图")
        return

    
    # 此处假设你已经有了 Draw_map 函数的 Python 实现
    # Draw_map(...) 
    
    num_plots = len(IC_Plot)
    for i in range(num_plots):
        # 获取颜色 (对应 MATLAB hsv2rgb)
        color_val = i / num_plots
        color_rgb = mcolors.hsv_to_rgb([color_val, 0.7, 0.9])
        
        # 提取索引并转为整数
        te = int(IC_Plot[i, 0])
        tp = int(IC_Plot[i, 1])
        Eid = int(IC_Plot[i, 2])
        Pid = int(IC_Plot[i, 3])
        IsoIdxE = int(IC_Plot[i, 4])
        IsoIdxP = int(IC_Plot[i, 5])
        ValPosId = int(IC_Plot[i, 9])  # Eid 对应的目标 ID
        TPId = int(IC_Plot[i, 10])     # Pid 对应的目标 ID

        # 绘制 Pursuer 等时线点 (^)
        # 假设 IsoMapP2TP_i_tt 是之前转化的字典
        iso_p_pos = IsoMapP2TP_i_tt[Pid][tp].IsoPos
        ax.plot(iso_p_pos[0, IsoIdxP], iso_p_pos[1, IsoIdxP], '^', color=color_rgb, markersize=5)
        
        # 绘制 Evader 等时线点 (d)
        iso_e_pos = IsoMapE2ValIn_i_tt[Eid][te].IsoPos
        ax.plot(iso_e_pos[0, IsoIdxE], iso_e_pos[1, IsoIdxE], 'd', color=color_rgb, markersize=5)
        
        # 绘制路径 (假设 pathFinal... 是字典或 2D 列表)
        path_e = pathFinalE2ValIn[Eid][ValPosId]
        if path_e is not None:
            ax.plot(path_e[0, :], path_e[1, :], '-', alpha=0.3)
            
        path_p = pathFinalP2TP[Pid][TPId]
        if path_p is not None:
            ax.plot(path_p[0, :], path_p[1, :], '-', alpha=0.3)

    ax.axis('equal')
    ax.grid(True)
    ax.set_title('Top 100 Intercept Candidates')
    # ax.show()

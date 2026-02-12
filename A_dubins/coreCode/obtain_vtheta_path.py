import numpy as np

def obtain_vtheta(center, pos_now, vtheta_old):
    """
    计算圆弧切线角。pos_now 是嵌套列表 [[x], [y]]。
    目标：计算 (-pi, pi] 范围内的连续切线方向。
    """
    # 安全检查：如果数据为空，直接返回空数组
    if pos_now is None or len(pos_now) == 0 or len(pos_now[0]) == 0:
        return np.array([])

    # 将嵌套列表转为 NumPy 数组 (2, N)
    pos = np.array(pos_now)
    x = pos[0, :] - center[0]
    y = pos[1, :] - center[1]

    # 计算两个候选切线角
    vtheta1 = np.arctan2(x, -y)
    vtheta2 = np.arctan2(-x, y)

    # 映射到 [0, 2*pi] 寻找起始点最接近的分支 (复刻 MATLAB 逻辑)
    v1_0 = vtheta1[0] % (2 * np.pi)
    v2_0 = vtheta2[0] % (2 * np.pi)
    old_0 = vtheta_old % (2 * np.pi)

    cha1 = np.abs(v1_0 - old_0)
    cha2 = np.abs(v2_0 - old_0)

    # 原有临界跳转判断逻辑
    if (old_0 < 0.01 and 2 * np.pi - max(v1_0, v2_0) < 0.01) or \
       (2 * np.pi - old_0 < 0.01 and min(v1_0, v2_0) < 0.01):
        idx = 0 if cha1 > cha2 else 1
    else:
        idx = 0 if cha1 < cha2 else 1

    selected = vtheta1 if idx == 0 else vtheta2
    
    # 保证弧段内角度连续变化
    res = np.unwrap(selected)
    
    # 严格映射回 (-pi, pi] 范围
    return (res + np.pi) % (2 * np.pi) - np.pi

def obtain_vtheta_path(param):
    """
    向量化处理所有 agent 路径，支持 path 中存在空段的情况。
    """
    # 定义角度映射匿名函数
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

    def process_agent(p):
        # 检查字段是否存在且 path 不为空
        if 'path' not in p or p['path'] is None:
            return p
        
        path = p['path']
        center = p['center']
        v_plot = p['vtheta_plot']
        
        # 预分配容器 (对应 MATLAB 的 cell 数组)
        # 初始化为 numpy 空数组，防止后续处理空值时报错
        v_all = [np.array([])] * len(path)

        # 辅助函数：判断某段 path[j] 是否有效
        is_val = lambda j: j < len(path) and path[j] is not None and len(path[j]) > 0 and len(path[j][0]) > 0

        # --- 针对 Dubins 的 5 段路径逻辑 (j=0..4) ---
        # j=1 (弧线):
        if is_val(0):
            v_all[0] = obtain_vtheta(center[:, 0], path[0], v_plot[0])
        
        # j=2 (直线):
        if is_val(1):
            v_all[1] = wrap(np.full(len(path[1][0]), v_plot[1]))
            
        # j=3 (弧线):
        if is_val(2):
            v_all[2] = obtain_vtheta(center[:, 1], path[2], v_plot[1])
            
        # j=4 (直线):
        if is_val(3):
            v_all[3] = wrap(np.full(len(path[3][0]), v_plot[2]))
            
        # j=5 (弧线):
        if is_val(4):
            v_all[4] = obtain_vtheta(center[:, 2], path[4], v_plot[2])

        p['vtheta_all'] = v_all
        return p

    # 使用列表推导式处理所有 Agent，消除显式 i 循环
    return [process_agent(p) for p in param]
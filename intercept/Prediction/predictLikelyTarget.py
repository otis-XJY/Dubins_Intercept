import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import numpy as np

def predictLikelyTargetNew(trajs, targets,lookback=50, k_angle=2.5, k_dist=0.001):
    """
    基于回溯采样的意图预测函数。
    
    参数:
    trajs       : list of arrays, 敌方轨迹列表 [x, y]
    targets     : (M, 2) array, 目标点位置
    lookback    : 回溯的点数。采样越密，此值应越大（例如 50-100）
    k_angle     : 朝向惩罚权重。值越大，对偏离航向的意图判定越苛刻
    k_dist      : 距离惩罚权重。根据地图尺度调整
    """
    num_drones = len(trajs)
    num_targets = targets.shape[0]
    
    if num_drones == 0:
        return {'rank_idx': np.empty((0, num_targets)), 'probs': np.empty((0, num_targets))}

    # --- 1. 提取当前位置与回溯位置 (消除显式循环，利用列表推导式) ---
    # pos_now: (N, 2) 当前时刻点
    pos_now = np.array([t[-1, :2] for t in trajs])
    
    # pos_old: (N, 2) 回溯 lookback 个位置的点
    # 如果轨迹长度不足 lookback，则取起始点 t[0]
    pos_old = np.array([t[-lookback, :2] if t.shape[0] > lookback else t[0, :2] for t in trajs])

    # --- 2. 计算平滑航向角 (N,) ---
    # 通过当前点与回溯点的位移矢量确定方向
    delta = pos_now - pos_old
    headings = np.array([t[-1, 2] for t in trajs])

    # --- 3. 向量化几何计算 (Broadcasting) ---
    # 计算所有无人机到所有目标的相对矢量: (N, M, 2)
    # targets[None, :, :] -> (1, M, 2)
    # pos_now[:, None, :] -> (N, 1, 2)
    vec_to_targets = targets[None, :, :] - pos_now[:, None, :]
    
    # 计算距离矩阵: (N, M)
    dists = np.linalg.norm(vec_to_targets, axis=2)
    
    # 计算目标的方位角: (N, M)
    target_phis = np.arctan2(vec_to_targets[..., 1], vec_to_targets[..., 0])
    
    # 计算朝向偏差 angle_errors: (N, M)
    # 使用最短角位移公式：abs(atan2(sin(a-b), cos(a-b)))
    angle_diffs = target_phis - headings[:, None]
    angle_errors = np.abs(np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs)))

    # --- 4. 计算意图得分与概率 ---
    # 公式：越正对目标且距离越近，得分越高
    # scores: (N, M)
    scores = np.exp(-k_angle * angle_errors) * np.exp(-k_dist * dists)
    
    # 归一化概率
    probs = scores / (np.sum(scores, axis=1, keepdims=True) + 1e-12)

    # --- 5. 排序并返回结果 ---
    # 获取最可能目标的索引 (1-based)
    rank_idx = np.argsort(-probs, axis=1)

    return {
        'rank_idx': rank_idx,
        'probs': probs
    }
def predictLikelyTarget(trajs, targets, **kwargs):
    """
    根据多条二维轨迹预测最可能的目标点。
    变量名与逻辑与 MATLAB 版本保持一致。
    """
    # --- 1. 参数解析 (替代 inputParser) ---
    opt = {
        'Method': kwargs.get('Method', 'heading'),
        'UseWindow': kwargs.get('UseWindow', 3),
        'KAngle': kwargs.get('KAngle', 0.15),
        'KDist': kwargs.get('KDist', 0.01),
        'Verbose': kwargs.get('Verbose', False)
    }

    K = len(trajs)
    m = targets.shape[0]
    eps = np.finfo(float).eps

    # --- 2. 获取当前位置与方向 (Step 1) ---
    pos = np.zeros((K, 2))
    dir_deg = np.zeros(K)

    for k in range(K):
        T = np.atleast_2d(trajs[k])
        pos[k, :] = T[-1, :2]
        
        if opt['Method'] == 'heading' and T.shape[1] >= 3:
            # 假设第三列是弧度 (MATLAB T(end,3)/pi*180)
            dir_deg[k] = np.degrees(T[-1, 2])
        else:
            # 估计方向向量
            direction_vec = _estimate_direction(T, opt)
            dir_deg[k] = np.degrees(np.arctan2(direction_vec[1], direction_vec[0]))

    # --- 3. 计算角度差与距离 (Step 2) ---
    # 利用广播机制: vec 形状 (K, m, 2)
    # targets: (m, 2) -> (1, m, 2)
    # pos: (K, 2) -> (K, 1, 2)
    vec = targets[np.newaxis, :, :] - pos[:, np.newaxis, :]
    
    # distance: (K, m)
    distance = np.sqrt(np.sum(vec**2, axis=2))
    
    # ang_t: 目标点相对于当前位置的角度 (K, m)
    ang_t = np.degrees(np.arctan2(vec[:, :, 1], vec[:, :, 0]))
    
    # angle_error: 计算最小角度差 (K, m)
    # 逻辑: abs(mod(ang_t - dir_deg + 180, 360) - 180)
    angle_diff = ang_t - dir_deg[:, np.newaxis]
    angle_error = np.abs((angle_diff + 180) % 360 - 180)

    # --- 4. 计算得分 (Step 3) ---
    scores_raw = np.exp(-opt['KAngle'] * angle_error) * np.exp(-opt['KDist'] * distance)
    # 归一化
    row_sums = scores_raw.sum(axis=1, keepdims=True)
    scores_norm = scores_raw / (row_sums + eps)
    
    # 排序 (降序)
    rank_idx = np.argsort(-scores_norm, axis=1)

    # --- 5. 构造结果 (Step 4) ---
    result = {
        'scores_norm': scores_norm,
        'angle_error': angle_error,
        'distance': distance,
        'rank_idx': rank_idx, # 注意：Python 索引从 0 开始
        'pos': pos,
        'direction_deg': dir_deg
    }

    # --- 6. 可视化 (Step 5) ---
    if opt['Verbose']:
        _plot_prediction(trajs, targets, pos, dir_deg, m, K)

    return result

def _estimate_direction(T, opt):
    """子函数：估计轨迹末端方向向量"""
    n = T.shape[0]
    eps = np.finfo(float).eps
    if n < 2:
        return np.array([np.nan, np.nan])

    if opt['Method'] == 'diff':
        w = min(opt['UseWindow'], n - 1)
        # 差分均值
        v = np.diff(T[-w-1:, :2], axis=0)
        direction = np.mean(v, axis=0)
    
    elif opt['Method'] == 'poly':
        w = min(opt['UseWindow'] + 2, n)
        recent = T[-w:, :2]
        t = np.arange(recent.shape[0])
        # 拟合二次多项式
        px = np.polyfit(t, recent[:, 0], 2)
        py = np.polyfit(t, recent[:, 1], 2)
        # 求导数并取最后一个点的值
        dx = np.polyval(np.polyder(px), t[-1])
        dy = np.polyval(np.polyder(py), t[-1])
        direction = np.array([dx, dy])
    
    else: # 默认 fallback 到简单差分
        direction = T[-1, :2] - T[-2, :2]

    # 归一化
    norm_val = np.linalg.norm(direction) + eps
    return direction / norm_val

def _plot_prediction(trajs, targets, pos, dir_deg, m, K):
    """子函数：处理可视化"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', m)
    
    for k in range(K):
        t_arr = np.array(trajs[k])
        plt.plot(t_arr[:, 0], t_arr[:, 1], '-o', markersize=4, alpha=0.6)
        # 绘制方向箭头
        rad = np.radians(dir_deg[k])
        plt.quiver(pos[k, 0], pos[k, 1], np.cos(rad), np.sin(rad), 
                   color='k', scale=20, width=0.005)

    for i in range(m):
        plt.plot(targets[i, 0], targets[i, 1], 'o', markersize=10, 
                 markeredgecolor='k', markerfacecolor=colors(i))
        plt.text(targets[i, 0] + 0.3, targets[i, 1], f'T{i}')
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('预测最可能目标方向 (Python)')
    plt.show()

# --- 使用示例 ---
# trajs = [np.random.rand(10, 2) for _ in range(3)]
# targets = np.array([[5, 5], [10, 0]])
# res = predictLikelyTarget(trajs, targets, Method='poly', Verbose=True)
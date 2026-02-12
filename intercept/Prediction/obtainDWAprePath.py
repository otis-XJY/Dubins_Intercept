import numpy as np

def obtainDWAprePath(PosE, PathE, E_PreRef):
    """
    使用全向量化方式实现的 DWA 路径预测
    变量名与 MATLAB 保持严格一致
    """
    # 参数提取
    dt = 0.1
    num_v = int(E_PreRef['num_v'])
    num_w = int(E_PreRef['num_w'])
    v_range = E_PreRef['v_range']
    w_range = E_PreRef['w_range']
    T_pred = E_PreRef['T_pred']
    Stepsize = E_PreRef['Stepsize']

    # 1. 基础计算
    v_avg = np.mean(v_range)
    K_p = int(np.ceil(v_avg * T_pred / Stepsize)) # 目标点数
    K_t = int(np.ceil(T_pred / dt))               # 模拟步数
    n = PosE.shape[0]                             # 目标数量

    # 2. 控制输入网格化 (N = num_v * num_w)
    v_vec = np.linspace(v_range[0], v_range[1], num_v)
    w_vec = np.linspace(w_range[0], w_range[1], num_w)
    VV, WW = np.meshgrid(v_vec, w_vec)
    V = VV.flatten() # (N,)
    W = WW.flatten() # (N,)
    N = V.size

    # 3. 全向量化轨迹模拟 (重点：消除 for t = 1:K_t 循环)
    # 创建时间序列张量 (K_t,)
    t_steps = np.arange(1, K_t + 1) * dt
    
    # 利用广播机制计算所有时刻的 theta: th = th0 + W * t
    # PosE[:, 2] 是 (n,1), W 是 (1, N), t_steps 是 (1, 1, K_t)
    # 结果 th_all 维度: (n, N, K_t)
    th0 = PosE[:, [2]] # (n, 1)
    th_all = th0[:, :, np.newaxis] + W[np.newaxis, :, np.newaxis] * t_steps[np.newaxis, np.newaxis, :]
    
    # 计算每一步的位移增量 dx, dy
    # 结果维度: (n, N, K_t)
    dV = V[np.newaxis, :, np.newaxis] * dt
    dx = dV * np.cos(th_all)
    dy = dV * np.sin(th_all)
    
    # 通过累加和(cumsum)实现数值积分，替代显式循环
    # 结果维度: (n, N, K_t)
    x_all = PosE[:, [0], np.newaxis] + np.cumsum(dx, axis=2)
    y_all = PosE[:, [1], np.newaxis] + np.cumsum(dy, axis=2)
    
    # 组合为全量轨迹张量: (n, N, 2, K_t)
    traj_full = np.stack([x_all, y_all], axis=2)
    
    # 均匀采样 K_p 个点 (对应 MATLAB 的 sample_times)
    sample_idx = np.round(np.linspace(0, K_t - 1, K_p)).astype(int)
    SimPaths = traj_full[:, :, :, sample_idx] # (n, N, 2, K_p)

    # 4. 处理 PathE (Truncate & Pad)
    # 由于 PathE 是 list of arrays，长度不等，这里使用列表推导式后转为 numpy 数组
    # 目标维度: (n, 2, K_p)
    target_list = []
    for P in PathE:
        P_sub = P[:2, :K_p]
        if P_sub.shape[1] < K_p:
            # 补齐长度
            P_padded = np.pad(P_sub, ((0, 0), (0, K_p - P_sub.shape[1])), mode='constant')
            target_list.append(P_padded)
        else:
            target_list.append(P_sub)
    TargetPaths = np.array(target_list) # (n, 2, K_p)

    # 5. 比较并选择最佳轨迹 (向量化计算误差)
    # SimPaths: (n, N, 2, K_p), TargetPaths: (n, 2, K_p)
    # 将 TargetPaths 扩展维度以进行广播减法: (n, 1, 2, K_p)
    diffs = SimPaths - TargetPaths[:, np.newaxis, :, :]
    
    # 计算欧氏距离 dists: (n, N, K_p)
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # 计算平均误差 mean_dists: (n, N)
    mean_dists = np.mean(dists, axis=2)
    
    # 找到每个目标(n)对应的最小误差索引 (axis=1)
    min_idx = np.argmin(mean_dists, axis=1) # (n,)
    min_dists = mean_dists[np.arange(n), min_idx] # (n,)
    
    # 提取最佳轨迹 BestPaths: list of (2, K_p)
    # 对应 MATLAB 的 n×1 cell
    BestPaths = [SimPaths[i, min_idx[i], :, :] for i in range(n)]

    return min_dists, BestPaths
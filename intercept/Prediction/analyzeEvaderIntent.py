import numpy as np

def analyzeEvaderIntent(ValuePos, EvaderPos):
    """
    Python 版意图分析函数
    :param ValuePos: np.array, n x 3 [x, y, theta]
    :param EvaderPos: np.array, m x 3 [x, y, theta]
    :return: targetIndices (m,), threatMatrix (m, n)
    """
    # 1. 提取坐标并构建差分矩阵 (利用 NumPy 广播机制)
    # EvaderPos[:, 0][:, np.newaxis] 变成 (m, 1), ValuePos[:, 0] 是 (n,)
    # 相减结果 dx 和 dy 均为 (m, n)
    dx = ValuePos[:, 0] - EvaderPos[:, 0][:, np.newaxis]
    dy = ValuePos[:, 1] - EvaderPos[:, 1][:, np.newaxis]
    
    # 2. 计算距离矩阵 (m x n)
    distMatrix = np.hypot(dx, dy)
    
    # 3. 计算方位角矩阵 (m x n)
    # 每一行代表一个 Evader 看向所有 ValuePos 的绝对角度
    targetAngles = np.arctan2(dy, dx)
    
    # 4. 计算航向偏差矩阵 (m x n)
    # evaderHeadings 形状为 (m, 1), 自动广播减去 targetAngles 的每一行
    evaderHeadings = EvaderPos[:, 2][:, np.newaxis]
    angleDiff = targetAngles - evaderHeadings
    
    # 5. 角度归一化到 [-pi, pi] (矩阵化处理)
    angleDiff = np.arctan2(np.sin(angleDiff), np.cos(angleDiff))
    
    # 6. 计算威胁评估得分 (意图函数)
    # 距离因子：使用 np.exp 进行向量化指数运算
    distScore = np.exp(-distMatrix / 1000)
    
    # 意图因子：使用 np.cos 映射，偏差 > 90度 (cos < 0) 的置为 0
    intentScore = np.cos(angleDiff)
    intentScore[intentScore < 0] = 0 
    
    # 7. 综合权重 (m x n)
    threatMatrix = 0.4 * distScore + 0.6 * intentScore
    
    # 8. 寻找每行的最大值索引 (即每个入侵者的意图目标)
    # 注意：Python 索引从 0 开始。如果后续逻辑依赖 MATLAB 的 1 索引，请加 1：np.argmax(...) + 1
    targetIndices = np.argmax(threatMatrix, axis=1)
    
    return targetIndices, threatMatrix

# --- 转换要点说明 ---
# 1. 变量名完全保持一致。
# 2. 核心加速：利用 np.newaxis 实现 (m,1) 与 (n,) 的广播，消除双重循环。
# 3. 性能：np.hypot, np.arctan2 均为底层 C 实现，处理大规模矩阵速度极快。
# 4. 索引：保持 Python 惯例使用 0-based 索引，这对后续 scipy 分配算法更友好。
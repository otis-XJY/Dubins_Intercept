import numpy as np

def check_points_order(PosP, Trans_PointFrom, Trans_PointTo, ValuePos):
    """
    判断在从 PosP 到 ValuePos 的运动方向上，From 和 To 点的先后顺序。
    
    返回:
        1 : 先经过 From, 后经过 To (From -> To)
       -1 : 先经过 To, 后经过 From (To -> From)
        0 : 两者在运动方向上的位置重合
    """
    # 转换为 numpy 数组确保计算安全
    P = np.array(PosP)
    V = np.array(ValuePos)
    Pf = np.array(Trans_PointFrom)
    Pt = np.array(Trans_PointTo)

    # 1. 建立主轴向量（运动方向）
    # Direction Vector: 从起点指向终点
    dir_vec = V - P
    
    # 如果起点和终点重合，无法判断方向
    if np.linalg.norm(dir_vec) < 1e-6:
        return 0

    # 2. 计算两个待测点相对于起点 P 的向量
    vec_from = Pf - P
    vec_to = Pt - P

    # 3. 计算点在主轴向量上的投影得分 (点积)
    # Score = |vec| * |dir_vec| * cos(theta)
    # 这个得分反映了点在运动射线上的“进度”
    score_from = np.dot(vec_from, dir_vec)
    score_to = np.dot(vec_to, dir_vec)

    # 4. 比较得分
    if score_from < score_to:
        # Score 越小，距离起点 PosP 越近，越先经过
        return 1
    elif score_from > score_to:
        return -1
    else:
        return 0

# --- 测试示例 ---
PosP = (100, 100)
ValuePos = (0, 0)
Trans_PointFrom = (90, 100)
Trans_PointTo = (10, 0)

result = check_points_order(PosP, Trans_PointFrom, Trans_PointTo, ValuePos)

if result == 1:
    print("顺序为：先经过 Trans_PointFrom，后经过 Trans_PointTo")
elif result == -1:
    print("顺序为：先经过 Trans_PointTo，后经过 Trans_PointFrom")
else:
    print("两点在运动方向上并列")
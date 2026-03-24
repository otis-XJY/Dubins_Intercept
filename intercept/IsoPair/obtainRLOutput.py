import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection


def _extract_iso_points(iso_map, agent_ids, time_ids, iso_indices):
    get_iso = np.frompyfunc(lambda agent_id, time_id: iso_map[int(agent_id)][int(time_id)].IsoPos, 2, 1)
    get_x = np.frompyfunc(lambda iso_pos, iso_idx: iso_pos[0, int(iso_idx)], 2, 1)
    get_y = np.frompyfunc(lambda iso_pos, iso_idx: iso_pos[1, int(iso_idx)], 2, 1)
    get_vtheta = np.frompyfunc(lambda iso_pos, iso_idx: iso_pos[2, int(iso_idx)], 2, 1)

    iso_pos_arr = get_iso(agent_ids, time_ids)
    x = np.asarray(get_x(iso_pos_arr, iso_indices), dtype=float)
    y = np.asarray(get_y(iso_pos_arr, iso_indices), dtype=float)
    vtheta = np.asarray(get_vtheta(iso_pos_arr, iso_indices), dtype=float)
    return np.column_stack((x, y,vtheta))

def obtain_output_iso(InterceptCandidates, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt):
    """
    从 InterceptCandidates 中提取 E 和 P 的状态，并从 IsoMap 中获取对应的 IsoPos。

    InterceptCandidates: 包含拦截候选方案的数组，假设其结构与 MATLAB 代码中的相似。
    IsoMapP2TP_i_tt: P 的 IsoMap，二维列表或字典结构。
    IsoMapE2ValIn_i_tt: E 的 IsoMap，二维列表或字典结构。

    返回:
        output_P_state: P 的状态数组
        output_IsoP: P 的 IsoPos 数组
        output_E_state: E 的状态数组
        output_IsoE: E 的 IsoPos 数组
    """
    if InterceptCandidates.size == 0:
        print("没有可用的拦截候选方案进行提取")
        return None, None, None, None
    
    te = InterceptCandidates[:, 0].astype(int)
    tp = InterceptCandidates[:, 1].astype(int)
    Eid = InterceptCandidates[:, 11].astype(int)
    Pid = InterceptCandidates[:, 12].astype(int) #根据这个判断不同的PosP对象
    IsoIdxE = InterceptCandidates[:, 4].astype(int)
    IsoIdxP = InterceptCandidates[:, 5].astype(int)
    ValPosId = InterceptCandidates[:, 9].astype(int)
    TPId = InterceptCandidates[:, 10].astype(int)

    color_p = color_map((plot_ids * 2) % 20)
    color_e = color_map((plot_ids * 2 + 1) % 20)

    iso_p_points = _extract_iso_points(IsoMapP2TP_i_tt, Pid, tp, IsoIdxP)
    outputIsoP=np.hstack((iso_p_points,InterceptCandidates[:,15:]))
    return outputIsoP

def obtainNeighbour(
    PosP,
    ValuePos,
    distance,
    query_ids=None,
    target_ids=None,
    return_mode="list",
):
    """
    获得 PosP 中每个 pos 在范围 distance 内的 ValuePos 邻居。
    
    参数:
    - PosP: (N, D) 数组，N 个查询位置点，D 为维度（通常为 2 或 3）
    - ValuePos: (M, D) 数组，M 个候选位置点
    - distance: float，距离阈值
    
    返回:
    - return_mode="list":
        长度为 N 的列表，每个元素是该 PosP 点在 distance 范围内的 ValuePos 坐标数组。
    - return_mode="dict":
        以 query_ids 为键的字典，每个值为:
        {
            "target_ids": 邻居ID数组,
            "states": 邻居状态数组,
        }
    
    示例:
    >>> PosP = np.array([[0, 0], [10, 10]])
    >>> ValuePos = np.array([[1, 1], [5, 5], [100, 100]])
    >>> distance = 5.0
    >>> result = obtainNeighbour(PosP, ValuePos, distance)
    >>> len(result[0])  # PosP[0] 的邻居数量
    2
    >>> len(result[1])  # PosP[1] 的邻居数量
    0
    """
    # 输入验证
    if PosP.size == 0:
        return {} if return_mode == "dict" else []

    if ValuePos.size == 0:
        if return_mode == "dict":
            n_pos = PosP.shape[0] if PosP.ndim > 1 else 0
            if query_ids is None:
                query_ids = np.arange(n_pos)
            return {
                int(qid): {"target_ids": np.array([], dtype=int), "states": np.empty((0, 0), dtype=float)}
                for qid in np.asarray(query_ids).astype(int)
            }
        return [[] for _ in range(len(PosP))]
    
    if distance < 0:
        raise ValueError("距离阈值必须为非负数")
    
    # 提取前两个维度进行距离计算（假设至少包含 x, y 坐标）
    PosP_2d = PosP[:, :2] if PosP.ndim > 1 and PosP.shape[1] > 2 else PosP.reshape(-1, 2)
    ValuePos_2d = ValuePos[:, :2] if ValuePos.ndim > 1 and ValuePos.shape[1] > 2 else ValuePos.reshape(-1, 2)
    
    n_pos = PosP_2d.shape[0]
    n_value = ValuePos_2d.shape[0]
    
    # 使用广播机制计算所有点对之间的距离 (N, M)
    # PosP_2d: (N, 1, 2), ValuePos_2d: (1, M, 2) -> 差值：(N, M, 2)
    diff = PosP_2d[:, np.newaxis, :] - ValuePos_2d[np.newaxis, :, :]
    
    # 计算欧几里得距离矩阵 (N, M)
    distances_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # 创建掩码矩阵，标记哪些点在距离范围内 (N, M)
    mask_matrix = distances_matrix <= distance
    
    if return_mode not in ("list", "dict"):
        raise ValueError("return_mode 必须是 'list' 或 'dict'")

    if return_mode == "list":
        neighbour_list = []
        for i in range(n_pos):
            neighbours = ValuePos[mask_matrix[i]]
            neighbour_list.append(neighbours)
        return neighbour_list

    if query_ids is None:
        query_ids = np.arange(n_pos)
    if target_ids is None:
        target_ids = np.arange(n_value)

    query_ids = np.asarray(query_ids).astype(int)
    target_ids = np.asarray(target_ids).astype(int)

    if query_ids.shape[0] != n_pos:
        raise ValueError("query_ids 长度必须与 PosP 行数一致")
    if target_ids.shape[0] != n_value:
        raise ValueError("target_ids 长度必须与 ValuePos 行数一致")

    neighbour_dict = {}
    for i in range(n_pos):
        mask = mask_matrix[i]
        neighbour_dict[int(query_ids[i])] = {
            "target_ids": target_ids[mask],
            "states": ValuePos[mask],
        }

    return neighbour_dict

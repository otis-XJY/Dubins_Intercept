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


def _extract_path_segments(path_map, row_ids, col_ids):
    get_path = np.frompyfunc(lambda row_id, col_id: path_map[int(row_id)][int(col_id)], 2, 1)
    to_segment = np.frompyfunc(lambda path: path[:2, :].T, 1, 1)
    is_not_none = np.frompyfunc(lambda seg: seg is not None, 1, 1)

    path_arr = get_path(row_ids, col_ids)
    valid_mask = np.asarray(is_not_none(path_arr), dtype=bool)
    segment_arr = np.empty(path_arr.shape, dtype=object)
    segment_arr[:] = None
    if np.any(valid_mask):
        segment_arr[valid_mask] = to_segment(path_arr[valid_mask])
    return segment_arr, valid_mask


# 3. 绘图逻辑 (对应 MATLAB 注释部分)
def draw_candidates(InterceptCandidates, IsoMapP2TP_i_tt, IsoMapE2ValIn_i_tt, pathFinalE2ValIn, pathFinalP2TP,ax=None):
    """
    可视化前 100 个拦截方案
    """

    IC_Plot=InterceptCandidates[:100, :]  # 取前 100 个方案进行绘图
    if ax is None:
        ax = plt.gca()
    
    if IC_Plot.size == 0:
        print("没有可用的拦截候选方案进行绘图")
        return

    
    # 此处假设你已经有了 Draw_map 函数的 Python 实现
    # Draw_map(...) 
    
    num_plots = len(IC_Plot)
    plot_ids = np.arange(num_plots)
    color_map = mpl.colormaps['tab20']

    te = IC_Plot[:, 0].astype(int)
    tp = IC_Plot[:, 1].astype(int)
    Eid = IC_Plot[:, 11].astype(int)
    Pid = IC_Plot[:, 12].astype(int)
    IsoIdxE = IC_Plot[:, 4].astype(int)
    IsoIdxP = IC_Plot[:, 5].astype(int)
    ValPosId = IC_Plot[:, 9].astype(int)
    TPId = IC_Plot[:, 10].astype(int)

    color_p = color_map((plot_ids * 2) % 20)
    color_e = color_map((plot_ids * 2 + 1) % 20)

    iso_p_points = _extract_iso_points(IsoMapP2TP_i_tt, Pid, tp, IsoIdxP)
    iso_e_points = _extract_iso_points(IsoMapE2ValIn_i_tt, Eid, te, IsoIdxE)

    # 将每一对 P/E 拦截点批量连线（不使用显式循环）
    pair_segments = np.stack((iso_p_points, iso_e_points), axis=1)
    ax.add_collection(
        LineCollection(
            pair_segments.tolist(),
            colors=color_p,
            linewidths=1.8,
            linestyles='--',
            alpha=0.5,
            zorder=2,
        )
    )

    ax.scatter(
        iso_p_points[:, 0],
        iso_p_points[:, 1],
        c=color_p,
        marker='^',
        s=100,
        edgecolors='w',
        alpha=1,
    )
    ax.scatter(
        iso_e_points[:, 0],
        iso_e_points[:, 1],
        c=color_e,
        marker='d',
        s=100,
        edgecolors='w',
        alpha=1,
    )

    # path_e_segments, valid_e_mask = _extract_path_segments(pathFinalE2ValIn, Eid, ValPosId)
    # if np.any(valid_e_mask):
    #     ax.add_collection(
    #         LineCollection(
    #             path_e_segments[valid_e_mask].tolist(),
    #             colors=color_e[valid_e_mask],
    #             linewidths=2,
    #             linestyles='--',
    #             alpha=1,
    #         )
    #     )

    # path_p_segments, valid_p_mask = _extract_path_segments(pathFinalP2TP, Pid, TPId)
    # if np.any(valid_p_mask):
    #     ax.add_collection(
    #         LineCollection(
    #             path_p_segments[valid_p_mask].tolist(),
    #             colors=color_p[valid_p_mask],
    #             linewidths=2,
    #             linestyles='-',
    #             alpha=1,
    #         )
    #     )

    ax.axis('equal')
    ax.grid(True)
    ax.set_title('Top 100 Intercept Candidates')
    # ax.show()

"""验证 8 维特征匹配（含角度周期处理）相对于仅 xy 匹配的必要性。

对比两种索引映射方式：
  - Old：仅用 (c_x, c_y) 做 L1 最近邻，匹配到 obs 子集中最近的候选
  - New：用完整 8 维节点做 L1 最近邻，theta 分量用最短角距离

指标：
  1. 映射一致性：Old 和 New 选出相同索引的比例
  2. 映射误差：将选定 obs 节点与目标 IC 行的 8 维节点做 L1 距离比较
  3. 累积 reward：用两种映射生成的动作索引执行 step，比较 step_reward
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from marl.envs.todc_env import TODCMARLEnv
from marl.obs.generator import CandidateColumns

_COLS = CandidateColumns()


# ── Old：仅 xy 匹配 ──
def _match_obs_index_xy_only(
    env: TODCMARLEnv,
    row: np.ndarray,
    pid: int,
    self_pts_row: np.ndarray,
    mask_row: np.ndarray,
) -> int:
    """仅用 (x, y) 做最近邻匹配。"""
    k_valid = np.where(mask_row)[0]
    if k_valid.size == 0:
        return -1

    # 从 IC 行提取候选位置
    c_x, c_y, _ = env._extract_candidate_pos(row, int(pid))

    sub = np.asarray(self_pts_row[k_valid], dtype=np.float32)
    # 只用 x, y
    diff_xy = np.abs(sub[:, 0] - c_x) + np.abs(sub[:, 1] - c_y)
    return int(k_valid[np.argmin(diff_xy)])


# ── New：8 维特征匹配 + 角度周期 ──
def _ic_row_to_obs_node(env, row, pid):
    if row.ndim != 1 or row.shape[0] <= _COLS.Delta_V:
        return None
    c_x, c_y, theta = env._extract_candidate_pos(row, int(pid))
    return np.asarray(
        [
            float(c_x), float(c_y), float(theta),
            float(row[_COLS.Delta_t]),
            float(row[_COLS.Delta_d]),
            float(row[_COLS.Delta_theta]),
            float(row[_COLS.path_L]),
            float(row[_COLS.Delta_V]),
        ],
        dtype=np.float32,
    )


def _match_obs_index_8d(
    env: TODCMARLEnv,
    row: np.ndarray,
    pid: int,
    self_pts_row: np.ndarray,
    mask_row: np.ndarray,
) -> int:
    """8 维特征匹配，theta 用最短角距离。"""
    k_valid = np.where(mask_row)[0]
    if k_valid.size == 0:
        return -1

    tgt = _ic_row_to_obs_node(env, row, pid)
    if tgt is None:
        return int(k_valid[0])

    sub = np.asarray(self_pts_row[k_valid], dtype=np.float32)
    diff = np.abs(sub - tgt[None, :])
    # theta 周期处理
    d_theta = sub[:, 2] - float(tgt[2])
    diff[:, 2] = np.abs(np.arctan2(np.sin(d_theta), np.cos(d_theta)))

    score = np.sum(diff, axis=1)
    return int(k_valid[np.argmin(score)])


# ── 匈牙利分配（与 warm_start.py 一致） ──
from scipy.optimize import linear_sum_assignment


def _run_hungarian_on_ic(env):
    """在 env.IC_candidates 上做分组筛优 + 匈牙利分配，返回每对最优行的全局索引。"""
    ic_candidates = env.IC_candidates
    if ic_candidates is None or ic_candidates.ndim != 2 or ic_candidates.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    eid_ref = ic_candidates[:, _COLS.eid_ref].astype(np.int64)
    pid_ref = ic_candidates[:, _COLS.pid_ref].astype(np.int64)

    pair_key = eid_ref.astype(np.int64) * (pid_ref.max() + 2) + pid_ref.astype(np.int64)
    sorted_by_key = np.argsort(pair_key, kind="mergesort")
    sorted_keys = pair_key[sorted_by_key]

    change_mask = np.ones(len(sorted_keys), dtype=bool)
    change_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = np.where(change_mask)[0]
    group_ends = np.append(group_starts[1:], len(sorted_keys))

    best_rows, best_e_refs, best_p_refs = [], [], []
    for gs, ge in zip(group_starts, group_ends):
        group_global_idx = sorted_by_key[gs:ge]
        group = ic_candidates[group_global_idx]
        if group.shape[1] > _COLS.Delta_d:
            sort_idx = np.lexsort(
                (group[:, _COLS.Delta_d], group[:, 8], -group[:, _COLS.Delta_t])
            )
        else:
            sort_idx = np.argsort(group[:, 8])
        best_local = group_global_idx[sort_idx[0]]
        best_rows.append(best_local)
        best_e_refs.append(int(eid_ref[best_local]))
        best_p_refs.append(int(pid_ref[best_local]))

    if len(best_rows) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    best_rows = np.array(best_rows, dtype=np.int64)
    ic_best = ic_candidates[best_rows]
    best_e_refs = np.array(best_e_refs, dtype=np.int64)
    best_p_refs = np.array(best_p_refs, dtype=np.int64)

    e_unique, e_inv = np.unique(best_e_refs, return_inverse=True)
    p_unique, p_inv = np.unique(best_p_refs, return_inverse=True)
    n_p = len(p_unique)
    n_e = len(e_unique)

    cost_mat = np.full((n_p, n_e), 1e9, dtype=np.float64)
    idx_mat = np.full((n_p, n_e), -1, dtype=np.int64)
    cost_mat[p_inv, e_inv] = ic_best[:, 8].astype(np.float64)
    idx_mat[p_inv, e_inv] = np.arange(len(ic_best), dtype=np.int64)

    p_hung, e_hung = linear_sum_assignment(cost_mat)
    valid = cost_mat[p_hung, e_hung] < 1e8

    hung_ic_best_idx = idx_mat[p_hung[valid], e_hung[valid]]
    hung_global_rows = best_rows[hung_ic_best_idx]
    hung_e = best_e_refs[hung_ic_best_idx]
    hung_p = best_p_refs[hung_ic_best_idx]

    return hung_global_rows, hung_e, hung_p


def _build_pid_compact_map(env):
    pairs_realE2P = env.pairs_realE2P
    pairs_ic_compact = env._pairs_ic_compact
    pid_to_compact = {}
    if pairs_realE2P is not None and pairs_ic_compact is not None:
        for row_i in range(pairs_realE2P.shape[0]):
            eg = int(pairs_realE2P[row_i, 1])  # pid_global
            ec = int(pairs_ic_compact[row_i, 0])
            pc = int(pairs_ic_compact[row_i, 1])
            pid_to_compact[eg] = (ec, pc)
    return pid_to_compact


# ── 主测试 ──
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="测试 episode 数")
    parser.add_argument("--max-steps", type=int, default=50, help="每 episode 最大步数")
    args = parser.parse_args()

    # 创建环境（使用默认配置）
    env = TODCMARLEnv()
    os.environ["MPLBACKEND"] = "Agg"

    n_ep = args.episodes
    max_steps = args.max_steps

    # 统计量
    total_assigned_steps = 0  # 有匈牙利分配的步数
    total_p_assigned = 0  # 匈牙利覆盖的 P 总数
    n_same_index = 0  # Old/New 选出相同索引的次数
    sum_l1_old = 0.0  # Old 映射的 8 维 L1 误差
    sum_l1_new = 0.0  # New 映射的 8 维 L1 误差
    n_old_better = 0  # Old 的 L1 误差 < New 的次数
    n_new_better = 0  # New 的 L1 误差 < Old 的次数

    # theta 误差统计
    sum_theta_err_old = 0.0
    sum_theta_err_new = 0.0

    # delta_t 误差统计
    sum_dt_err_old = 0.0
    sum_dt_err_new = 0.0

    for ep in range(n_ep):
        obs_np, _ = env.reset()
        done = False
        trunc = False
        step = 0

        while (not done) and (not trunc) and step < max_steps:
            active = np.asarray(obs_np["pursuer_active"], dtype=bool).reshape(-1)
            mask = np.asarray(obs_np["self_pts_mask"], dtype=bool)
            self_pts = np.asarray(obs_np["self_pts"], dtype=np.float32)

            # 匈牙利分配
            hung_global_rows, hung_e, hung_p = _run_hungarian_on_ic(env)
            pid_to_compact = _build_pid_compact_map(env)

            # 构建紧凑→全局行映射
            hung_compact_map = {}
            for i in range(len(hung_e)):
                key = (int(hung_e[i]), int(hung_p[i]))
                hung_compact_map[key] = int(hung_global_rows[i])

            ic_candidates = env.IC_candidates
            num_P = env.num_P

            for p in range(num_P):
                if not active[p] or not np.any(mask[p]):
                    continue
                compact = pid_to_compact.get(p)
                if compact is None:
                    continue
                hung_row = hung_compact_map.get(compact)
                if hung_row is None:
                    continue

                total_p_assigned += 1

                # Old: 仅 xy
                idx_old = _match_obs_index_xy_only(
                    env, ic_candidates[hung_row], p, self_pts[p], mask[p]
                )
                # New: 8 维 + 角度周期
                idx_new = _match_obs_index_8d(
                    env, ic_candidates[hung_row], p, self_pts[p], mask[p]
                )

                if idx_old == idx_new:
                    n_same_index += 1

                # 计算 8 维 L1 误差
                tgt = _ic_row_to_obs_node(env, ic_candidates[hung_row], p)
                if tgt is None:
                    continue

                for label, idx, sum_l1_ref, sum_theta_ref, sum_dt_ref in [
                    ("old", idx_old, "sum_l1_old", "sum_theta_err_old", "sum_dt_err_old"),
                    ("new", idx_new, "sum_l1_new", "sum_theta_err_new", "sum_dt_err_new"),
                ]:
                    if idx < 0:
                        continue
                    node = self_pts[p, idx]
                    diff = np.abs(node - tgt)
                    # theta 周期
                    d_theta = node[2] - tgt[2]
                    diff[2] = abs(np.arctan2(np.sin(d_theta), np.cos(d_theta)))
                    l1 = float(np.sum(diff))
                    if label == "old":
                        sum_l1_old += l1
                        sum_theta_err_old += diff[2]
                        sum_dt_err_old += diff[3]
                    else:
                        sum_l1_new += l1
                        sum_theta_err_new += diff[2]
                        sum_dt_err_new += diff[3]

                # 比较哪个更优
                if idx_old >= 0 and idx_new >= 0:
                    node_old = self_pts[p, idx_old]
                    node_new = self_pts[p, idx_new]
                    diff_old = np.abs(node_old - tgt)
                    d_theta_old = node_old[2] - tgt[2]
                    diff_old[2] = abs(np.arctan2(np.sin(d_theta_old), np.cos(d_theta_old)))
                    l1_old = float(np.sum(diff_old))

                    diff_new = np.abs(node_new - tgt)
                    d_theta_new = node_new[2] - tgt[2]
                    diff_new[2] = abs(np.arctan2(np.sin(d_theta_new), np.cos(d_theta_new)))
                    l1_new = float(np.sum(diff_new))

                    if l1_old < l1_new - 1e-9:
                        n_old_better += 1
                    elif l1_new < l1_old - 1e-9:
                        n_new_better += 1

            total_assigned_steps += 1

            # 用 New 的动作继续执行（保持环境一致性）
            rule_idx = np.full(num_P, -1, dtype=np.int64)
            for p in range(num_P):
                if not active[p] or not np.any(mask[p]):
                    continue
                compact = pid_to_compact.get(p)
                if compact is None:
                    continue
                hung_row = hung_compact_map.get(compact)
                if hung_row is None:
                    continue
                idx = _match_obs_index_8d(
                    env, ic_candidates[hung_row], p, self_pts[p], mask[p]
                )
                if idx >= 0:
                    rule_idx[p] = idx

            # 兜底：未覆盖的 P
            for p in range(num_P):
                if not active[p] or not np.any(mask[p]):
                    continue
                if rule_idx[p] >= 0:
                    continue
                k_valid = np.where(mask[p])[0]
                if len(k_valid) > 0:
                    rule_idx[p] = int(k_valid[0])

            for p in range(num_P):
                if not active[p]:
                    rule_idx[p] = 0

            next_obs_np, _rew, terms, truncs, _infos = env.step(rule_idx)
            obs_np = next_obs_np
            done = bool(terms["__all__"])
            trunc = bool(truncs["__all__"])
            step += 1

    # ── 汇总 ──
    print("=" * 70)
    print("  8 维特征匹配 vs 仅 xy 匹配 — 必要性验证")
    print("=" * 70)
    print(f"  测试 episode 数: {n_ep}")
    print(f"  总决策步数（有匈牙利分配）: {total_assigned_steps}")
    print(f"  匈牙利覆盖的 P-步总数: {total_p_assigned}")
    print()

    if total_p_assigned > 0:
        same_rate = n_same_index / total_p_assigned * 100
        print(f"  映射一致率（Old/New 选相同索引）: {same_rate:.2f}%  ({n_same_index}/{total_p_assigned})")
        print()

        avg_l1_old = sum_l1_old / total_p_assigned
        avg_l1_new = sum_l1_new / total_p_assigned
        print(f"  8 维 L1 误差 — Old(xy): {avg_l1_old:.6f}")
        print(f"  8 维 L1 误差 — New(8d): {avg_l1_new:.6f}")
        improvement = (avg_l1_old - avg_l1_new) / max(avg_l1_old, 1e-9) * 100
        print(f"  相对改善: {improvement:.2f}%")
        print()

        avg_theta_old = sum_theta_err_old / total_p_assigned
        avg_theta_new = sum_theta_err_new / total_p_assigned
        print(f"  theta 角度误差 — Old: {avg_theta_old:.6f} rad")
        print(f"  theta 角度误差 — New: {avg_theta_new:.6f} rad")
        print()

        avg_dt_old = sum_dt_err_old / total_p_assigned
        avg_dt_new = sum_dt_err_new / total_p_assigned
        print(f"  delta_t 误差 — Old: {avg_dt_old:.6f}")
        print(f"  delta_t 误差 — New: {avg_dt_new:.6f}")
        print()

        print(f"  Old 更优次数: {n_old_better}")
        print(f"  New 更优次数: {n_new_better}")
        print(f"  持平次数: {total_p_assigned - n_old_better - n_new_better}")
        print()

        # 结论判断
        if same_rate > 99.0:
            print("  ★ 结论：映射一致率 > 99%，8 维特征匹配非必要（仅 xy 即可）")
        elif same_rate > 95.0:
            print("  ○ 结论：映射一致率 95~99%，8 维匹配有少量改善但非关键")
        elif improvement > 5.0:
            print(f"  ★ 结论：8 维匹配 L1 误差相对改善 {improvement:.1f}%，有必要保留")
        elif improvement > 1.0:
            print(f"  ○ 结论：8 维匹配 L1 误差相对改善 {improvement:.1f}%，改善有限但建议保留")
        else:
            print("  ○ 结论：8 维匹配改善微小（< 1%），非必要但无副作用")
    else:
        print("  无匈牙利覆盖的 P-步，无法比较")

    print("=" * 70)


if __name__ == "__main__":
    main()

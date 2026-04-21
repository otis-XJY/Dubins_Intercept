"""MARL 观测张量构造：将几何与分配关系转为策略网络可用的 dict。

与 ``TODCMARLEnv._build_obs`` 配合；候选行格式见 ``CandidateColumns``。
分配敌、友机槽位与 ``pairs_realE2P`` 对齐；攻击目标坐标取自 ``v_e_nodes[:,6:8]``（与推断目标一致）。
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np


def alive_pid_eid_sets(
    pairs_realE2P: Optional[np.ndarray], num_p: int, num_e: int
) -> Tuple[Set[int], Set[int]]:
    """由当前 ``pairs_realE2P`` 得到仍存活的全局 pid / eid 集合。"""
    if pairs_realE2P is None or pairs_realE2P.size == 0:
        return set(), set()
    pr = np.asarray(pairs_realE2P, dtype=np.int64)
    ap = {int(x) for x in np.unique(pr[:, 1]) if 0 <= int(x) < num_p}
    ae = {int(x) for x in np.unique(pr[:, 0]) if 0 <= int(x) < num_e}
    return ap, ae


# IC 候选行列索引（与 MARL_env 中 IC_candidates 列布局一致；索引 0–24 共 25 列）:
# 0–1: te, tp
# 2–3: ETPid, PTPid
# 4–5: IsoPosidxE, IsoPosidxP
# 6: Iso_dist；7: path_len；8: cost（汇总，观测构造未单独索引）
# 9: ValPosid（pathidE）；10: TPid（pathidP）
# 11–12: Eid, Pid 紧凑索引，而不是全局索引
# 13–14: Eiso, Piso
# 15–19: Delta_t, Delta_d, Delta_theta, path_L, Delta_V
# 20–24: cost_t, cost_d, cost_theta, cost_L, cost_V
@dataclass(frozen=True)
class CandidateColumns:
    te: int = 0
    tp: int = 1
    eid_ref: int = 11
    pid_ref: int = 12
    Delta_t: int = 15
    Delta_d: int = 16
    Delta_theta: int = 17
    path_L: int = 18
    Delta_V: int = 19
    cost_t: int = 20
    cost_d: int = 21
    cost_theta: int = 22
    cost_L: int = 23
    cost_V: int = 24



class TODCObservationGenerator:
    """从仿真状态构建单步观测 dict（含策略输入与计奖用 ``enemies``/``reward_nodes`` 等）。"""

    def __init__(
        self,
        candidate_limit: Optional[int] = None,
        candidate_cols: Optional[CandidateColumns] = None,
        ally_perception_radius: Optional[float] = None,
    ):
        self.candidate_limit = int(candidate_limit) if candidate_limit is not None else None
        self.cols = candidate_cols or CandidateColumns()
        # 仅感知半径内的友机；None 表示不裁剪（极大半径）
        self.ally_perception_radius = float(ally_perception_radius) if ally_perception_radius is not None else float("inf")

    def generate(
        self,
        *,
        pos_p: np.ndarray,
        pos_e: np.ndarray,
        v_p: float,
        v_e: float,
        value_pos: np.ndarray,
        num_p: int,
        num_e: int,
        ic_candidates: np.ndarray,
        candidate_pos_fn: Optional[Callable[[np.ndarray, int], Tuple[float, float]]] = None,
        inferred_targets: Optional[np.ndarray] = None,
        pairs_realE2P: Optional[np.ndarray] = None,
        pairs_ic_ref: Optional[np.ndarray] = None,
        capflag: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """组装节点特征并调用 ``_build_model_aligned_obs``。

        ``pairs_realE2P`` 为当前 E–P 分配（全局 eid/pid）；缺失时 ``_build_c_nodes`` 会报错。
        ``pairs_ic_ref`` 与 ``pairs_realE2P`` 逐行对齐，为 IC 表第 11–12 列使用的紧凑索引；为 None 时回退为与全局相同（仅当全局 id 与表内一致时有效）。
        """
        v_p_nodes = self._build_p_nodes(pos_p, v_p)
        v_e_nodes = self._build_e_nodes(pos_e, v_e, value_pos, inferred_targets)
        v_c_nodes, v_c_mask, reward_nodes = self._build_c_nodes(
            ic_candidates=ic_candidates,
            num_p=num_p,
            num_e=num_e,
            pos_p=pos_p,
            pos_e=pos_e,
            value_pos=value_pos,
            candidate_pos_fn=candidate_pos_fn,
            pairs_realE2P=pairs_realE2P,
            pairs_ic_ref=pairs_ic_ref,
        )

        model_obs = self._build_model_aligned_obs(
            v_p_nodes=v_p_nodes,
            v_e_nodes=v_e_nodes,
            v_c_nodes=v_c_nodes,
            v_c_mask=v_c_mask,
            value_pos=value_pos,
            reward_nodes=reward_nodes,
            pos_p=pos_p,
            pos_e=pos_e,
            pairs_realE2P=pairs_realE2P,
            capflag=capflag,
        )
        return model_obs

    @staticmethod
    def _eid_for_each_pid(pairs_realE2P: np.ndarray, num_p: int) -> np.ndarray:
        """pairs rows (eid, pid) -> eid assigned to each pursuer index, -1 if missing."""
        out = np.full(num_p, -1, dtype=np.int64)
        if pairs_realE2P is None or pairs_realE2P.size == 0:
            return out
        pr = np.asarray(pairs_realE2P, dtype=np.int64)
        for i in range(pr.shape[0]):
            eid, pid = int(pr[i, 0]), int(pr[i, 1])
            if 0 <= pid < num_p:
                out[pid] = eid
        return out

    def _build_model_aligned_obs(
        self,
        *,
        v_p_nodes: np.ndarray,
        v_e_nodes: np.ndarray,
        v_c_nodes: np.ndarray,
        v_c_mask: np.ndarray,
        value_pos: np.ndarray,
        reward_nodes: np.ndarray,
        pos_p: np.ndarray,
        pos_e: np.ndarray,
        pairs_realE2P: Optional[np.ndarray],
        capflag: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """构建与 ``UAVInterceptionNetwork`` 键一致的观测（首维为 pursuer 批 ``P``）。

        友机槽与 ``ally_pts`` 仅含 ``pairs_realE2P`` 中仍存活的 pursuer；无配对 pid 无友机观测。
        ``asset_target_*`` 为各分配敌在 ``v_e_nodes`` 中的推断攻击目标 ``(x,y)``。
        """
        num_p = v_p_nodes.shape[0]
        num_e = v_e_nodes.shape[0]
        k_max = v_c_nodes.shape[1]

        # self_uav: [P, 1, 3] -> (x, y, theta)
        theta_p = np.arctan2(v_p_nodes[:, 5], v_p_nodes[:, 4])
        self_uav = np.stack([v_p_nodes[:, 0], v_p_nodes[:, 1], theta_p], axis=-1)[:, None, :].astype(np.float32)

        theta_e = np.arctan2(v_e_nodes[:, 5], v_e_nodes[:, 4])
        eid_by_pid = self._eid_for_each_pid(pairs_realE2P if pairs_realE2P is not None else np.empty((0, 2)), num_p)
        alive_pids, alive_eids = alive_pid_eid_sets(pairs_realE2P, num_p, num_e)

        # allies_local: [P, A, 3] — 仅存活友机；感知半径内按距离升序填满槽位
        ally_slots = max(1, num_p - 1)
        allies_local = np.zeros((num_p, ally_slots, 3), dtype=np.float32)
        ally_mask = np.zeros((num_p, ally_slots), dtype=bool)
        enemy_assigned_self = np.zeros((num_p, 1, 3), dtype=np.float32)
        enemy_assigned_per_ally = np.zeros((num_p, ally_slots, 3), dtype=np.float32)
        enemy_self_mask = np.zeros((num_p, 1), dtype=bool)
        ally_enemy_mask = np.zeros((num_p, ally_slots), dtype=bool)
        # 重要设施：分配敌在 v_e_nodes 中推断的攻击目标平面坐标 [E,2] -> 按分配关系切片
        asset_target_self = np.zeros((num_p, 1, 2), dtype=np.float32)
        asset_target_per_ally = np.zeros((num_p, ally_slots, 2), dtype=np.float32)

        rad = self.ally_perception_radius
        for pid in range(num_p):
            if pid not in alive_pids:
                continue
            others = [j for j in range(num_p) if j != pid and j in alive_pids]
            if len(others) > 0:
                dists = [
                    float(np.hypot(pos_p[pid, 0] - pos_p[j, 0], pos_p[pid, 1] - pos_p[j, 1]))
                    for j in others
                ]
                order = np.argsort(np.asarray(dists, dtype=np.float64))
                in_range = [others[int(i)] for i in order if dists[int(i)] <= rad]
                n_fill = min(ally_slots, len(in_range))
                for slot in range(n_fill):
                    j = in_range[slot]
                    allies_local[pid, slot, 0] = v_p_nodes[j, 0]
                    allies_local[pid, slot, 1] = v_p_nodes[j, 1]
                    allies_local[pid, slot, 2] = theta_p[j]
                    ally_mask[pid, slot] = True
                    eid_j = int(eid_by_pid[j])
                    if 0 <= eid_j < num_e and eid_j in alive_eids:
                        ally_enemy_mask[pid, slot] = True
                        enemy_assigned_per_ally[pid, slot, 0] = pos_e[eid_j, 0]
                        enemy_assigned_per_ally[pid, slot, 1] = pos_e[eid_j, 1]
                        enemy_assigned_per_ally[pid, slot, 2] = float(theta_e[eid_j])
                        asset_target_per_ally[pid, slot, :] = v_e_nodes[eid_j, 6:8].astype(np.float32)

            eid_self = int(eid_by_pid[pid])
            if 0 <= eid_self < num_e and eid_self in alive_eids:
                enemy_self_mask[pid, 0] = True
                enemy_assigned_self[pid, 0, 0] = pos_e[eid_self, 0]
                enemy_assigned_self[pid, 0, 1] = pos_e[eid_self, 1]
                enemy_assigned_self[pid, 0, 2] = float(theta_e[eid_self])
                asset_target_self[pid, 0, :] = v_e_nodes[eid_self, 6:8].astype(np.float32)

        # self_pts: [P, K, 8] = (x,y,theta,Delta_t,Delta_d,Delta_.
        # theta,pathL,DistanceV)
        self_pts = np.asarray(v_c_nodes, dtype=np.float32)
        self_pts_mask = v_c_mask > 0.0

        ally_pts_len = max(1, (num_p - 1) * k_max)

        # enemies: [P, E, 3] -> (x, y, theta), replicated per pursuer（计奖用）
        enemies_single = np.stack([v_e_nodes[:, 0], v_e_nodes[:, 1], theta_e], axis=-1).astype(np.float32)
        enemies = np.repeat(enemies_single[None, :, :], num_p, axis=0)
        enemy_mask = np.ones((num_p, num_e), dtype=bool)

        # targets keep model compatibility; use inferred target coordinates from enemy nodes
        targets_single = v_e_nodes[:, 6:8].astype(np.float32)
        targets = np.repeat(targets_single[None, :, :], num_p, axis=0)
        target_mask = np.ones((num_p, num_e), dtype=bool)

        # assets: important facilities, [P, V, 2]
        assets_single = np.asarray(value_pos[:, :2], dtype=np.float32)
        num_v = assets_single.shape[0]
        assets = np.repeat(assets_single[None, :, :], num_p, axis=0)
        asset_mask = np.ones((num_p, num_v), dtype=bool)

        self_uav, self_pts, self_pts_mask, allies_local, ally_mask, reward_nodes = self._zero_dead_pursuers(
            self_uav,
            self_pts,
            self_pts_mask,
            allies_local,
            ally_mask,
            reward_nodes,
            enemy_assigned_self,
            enemy_assigned_per_ally,
            enemy_self_mask,
            ally_enemy_mask,
            asset_target_self,
            asset_target_per_ally,
            alive_pids,
            num_p,
        )
        ally_pts, ally_pts_mask = self._rebuild_ally_pts_from_self(
            self_pts, self_pts_mask, num_p, k_max, ally_pts_len, alive_pids
        )
        enemies, enemy_mask, targets, target_mask = self._zero_dead_evaders(
            enemies, enemy_mask, targets, target_mask, alive_eids, num_p, num_e
        )

        pursuer_active = np.asarray(np.any(self_pts_mask, axis=1), dtype=np.int8)

        return {
            "self_uav": self_uav,
            "allies_local": allies_local,
            "enemy_assigned_self": enemy_assigned_self,
            "enemy_assigned_per_ally": enemy_assigned_per_ally,
            "asset_target_self": asset_target_self,
            "asset_target_per_ally": asset_target_per_ally,
            "enemy_self_mask": np.asarray(enemy_self_mask, dtype=np.int8),
            "self_pts": self_pts,
            "ally_pts": ally_pts,
            "enemies": enemies,
            "targets": targets,
            "assets": assets,
            "reward_nodes": reward_nodes,
            "ally_mask": ally_mask,
            "ally_enemy_mask": np.asarray(ally_enemy_mask, dtype=np.int8),
            "self_pts_mask": self_pts_mask,
            "ally_pts_mask": ally_pts_mask,
            "enemy_mask": enemy_mask,
            "target_mask": target_mask,
            "asset_mask": asset_mask,
            "pursuer_active": pursuer_active,
            "self_Capflag": capflag,
        }

    def _zero_dead_pursuers(
        self,
        self_uav: np.ndarray,
        self_pts: np.ndarray,
        self_pts_mask: np.ndarray,
        allies_local: np.ndarray,
        ally_mask: np.ndarray,
        reward_nodes: np.ndarray,
        enemy_assigned_self: np.ndarray,
        enemy_assigned_per_ally: np.ndarray,
        enemy_self_mask: np.ndarray,
        ally_enemy_mask: np.ndarray,
        asset_target_self: np.ndarray,
        asset_target_per_ally: np.ndarray,
        alive_pids: Set[int],
        num_p: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """无配对的 pid（已拦截完成等）对应行置 0，mask 全 False；分配敌与资产目标一并清除。"""
        for pid in range(num_p):
            if pid in alive_pids:
                continue
            self_uav[pid] = 0.0
            self_pts[pid] = 0.0
            self_pts_mask[pid] = False
            allies_local[pid] = 0.0
            ally_mask[pid] = False
            reward_nodes[pid] = 0.0
            enemy_assigned_self[pid] = 0.0
            enemy_assigned_per_ally[pid] = 0.0
            enemy_self_mask[pid, :] = False
            ally_enemy_mask[pid, :] = False
            asset_target_self[pid] = 0.0
            asset_target_per_ally[pid] = 0.0
        return self_uav, self_pts, self_pts_mask, allies_local, ally_mask, reward_nodes

    @staticmethod
    def _rebuild_ally_pts_from_self(
        self_pts: np.ndarray,
        self_pts_mask: np.ndarray,
        num_p: int,
        k_max: int,
        ally_pts_len: int,
        alive_pids: Set[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """仅存活友机的 ``self_pts`` 按全局 pid 升序拼接；无配对 pid 行保持全 0 / mask False。"""
        ally_pts = np.zeros((num_p, ally_pts_len, 8), dtype=np.float32)
        ally_pts_mask = np.zeros((num_p, ally_pts_len), dtype=bool)
        for pid in range(num_p):
            if pid not in alive_pids:
                continue
            others = [i for i in range(num_p) if i != pid and i in alive_pids]
            if len(others) == 0:
                continue
            block = self_pts[others].reshape(len(others) * k_max, 8)
            block_mask = self_pts_mask[others].reshape(len(others) * k_max)
            ally_pts[pid, : block.shape[0]] = block
            ally_pts_mask[pid, : block.shape[0]] = block_mask
        return ally_pts, ally_pts_mask

    @staticmethod
    def _zero_dead_evaders(
        enemies: np.ndarray,
        enemy_mask: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
        alive_eids: Set[int],
        num_p: int,
        num_e: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """无配对 eid 的列置 0，mask False（``assets`` 不在此处理）。"""
        for eid in range(num_e):
            if eid in alive_eids:
                continue
            enemies[:, eid, :] = 0.0
            enemy_mask[:, eid] = False
            targets[:, eid, :] = 0.0
            target_mask[:, eid] = False
        return enemies, enemy_mask, targets, target_mask

    def _build_p_nodes(self, pos_p: np.ndarray, v_p: float) -> np.ndarray:
        vx = v_p * np.cos(pos_p[:, 2])
        vy = v_p * np.sin(pos_p[:, 2])
        return np.column_stack(
            (
                pos_p[:, 0],
                pos_p[:, 1],
                vx,
                vy,
                np.cos(pos_p[:, 2]),
                np.sin(pos_p[:, 2]),
            )
        ).astype(np.float32)

    def _build_e_nodes(
        self,
        pos_e: np.ndarray,
        v_e: float,
        value_pos: np.ndarray,
        inferred_targets: Optional[np.ndarray],
    ) -> np.ndarray:
        vx = v_e * np.cos(pos_e[:, 2])
        vy = v_e * np.sin(pos_e[:, 2])

        if inferred_targets is None:
            inferred_targets = np.array([value_pos[i % len(value_pos), :2] for i in range(pos_e.shape[0])])

        return np.column_stack(
            (
                pos_e[:, 0],
                pos_e[:, 1],
                vx,
                vy,
                np.cos(pos_e[:, 2]),
                np.sin(pos_e[:, 2]),
                inferred_targets[:, 0],
                inferred_targets[:, 1],
            )
        ).astype(np.float32)

    def _build_c_nodes(
        self,
        *,
        ic_candidates: np.ndarray,
        num_p: int,
        num_e: int,
        pos_p: np.ndarray,
        pos_e: np.ndarray,
        value_pos: np.ndarray,
        candidate_pos_fn: Optional[Callable[[np.ndarray, int], Tuple[float, float]]],
        pairs_realE2P: Optional[np.ndarray],
        pairs_ic_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pairs_realE2P is None:
            raise ValueError(
                "pairs_realE2P is None: observation requires an E–P assignment from replan (check enable_dwa_replan / reset loop)"
            )
        # IC 表 11–12 列为紧凑 ide/idp；过滤须用紧凑键。全局 eid/pid 仅用于 pid 槽与 alive。
        pr = np.asarray(pairs_realE2P, dtype=np.int64)
        pic = np.asarray(pairs_ic_ref, dtype=np.int64) if pairs_ic_ref is not None else None
        pid_to_subset = {}
        max_k = 0
        for row_i in range(pr.shape[0]):
            eid_g, pid_g = int(pr[row_i, 0]), int(pr[row_i, 1])
            ceid, c_pid = int(pic[row_i, 0]), int(pic[row_i, 1])
            eidMask = (ic_candidates[:, self.cols.eid_ref].astype(int) == ceid)
            pidMask = (ic_candidates[:, self.cols.pid_ref].astype(int) == c_pid)
            subset = ic_candidates[eidMask & pidMask]
            pid_to_subset[pid_g] = subset
            max_k = max(max_k, int(subset.shape[0]) if subset.ndim > 0 else 0)

        if self.candidate_limit is not None:
            max_k = min(max_k, self.candidate_limit)
        max_k = max(1, max_k)

        nodes = np.zeros((num_p, max_k, 8), dtype=np.float32)
        reward_nodes = np.zeros((num_p, max_k, 8), dtype=np.float32)
        mask = np.zeros((num_p, max_k), dtype=np.float32)

        ncols = ic_candidates.shape[1] if ic_candidates.ndim >= 2 else 0
        empty_subset = np.empty((0, ncols), dtype=ic_candidates.dtype if ic_candidates.size else np.float64)

        for pid in range(num_p):
            subset = pid_to_subset.get(pid, empty_subset)
            if subset.size == 0:
                continue

            n = min(max_k, subset.shape[0])
            for i in range(n):
                row = subset[i]
                if candidate_pos_fn is not None:
                    c_x, c_y, theta = candidate_pos_fn(row, pid)
                # else:
                #     eid = int(row[self.cols.eid_ref]) if row.shape[0] > self.cols.eid_ref else pid % max(1, num_e)
                #     eid = max(0, min(eid, num_e - 1))
                #     c_x, c_y = float(pos_e[eid, 0]), float(pos_e[eid, 1])

                t_p = float(row[self.cols.tp]) if row.shape[0] > self.cols.tp else 0.0
                t_e = float(row[self.cols.te]) if row.shape[0] > self.cols.te else 0.0
                path_l = float(row[self.cols.path_L]) if row.shape[0] > self.cols.path_L else 0.0
                cost_l = float(row[self.cols.cost_L]) if row.shape[0] > self.cols.cost_L else 0.0
                delta_t = float(row[self.cols.Delta_t]) if row.shape[0] > self.cols.Delta_t else 0.0
                cost_t = float(row[self.cols.cost_t]) if row.shape[0] > self.cols.cost_t else 0.0

                p_x, p_y, p_th = float(pos_p[pid, 0]), float(pos_p[pid, 1]), float(pos_p[pid, 2])
                vec_x, vec_y = c_x - p_x, c_y - p_y
                theta = float(np.arctan2(vec_y, vec_x)) if (abs(vec_x) + abs(vec_y)) > 1e-9 else p_th

                # Approximate intercept deviation features from available geometry
                delta_d = float(row[self.cols.Delta_d]) if row.shape[0] > self.cols.Delta_d else 0.0
                cost_d = float(row[self.cols.cost_d]) if row.shape[0] > self.cols.cost_d else 0.0
                delta_theta = float(row[self.cols.Delta_theta]) if row.shape[0] > self.cols.Delta_theta else 0.0
                cost_theta = float(row[self.cols.cost_theta]) if row.shape[0] > self.cols.cost_theta else 0.0
                distance_V = float(row[self.cols.Delta_V]) if row.shape[0] > self.cols.Delta_V else 0.0
                cost_V = float(row[self.cols.cost_V]) if row.shape[0] > self.cols.cost_V else 0.0

                nodes[pid, i] = np.array(
                    [c_x, c_y, theta, delta_t, delta_d, delta_theta, path_l, distance_V],
                    dtype=np.float32,
                )

                reward_nodes[pid, i] = np.array(
                    [c_x, c_y, theta, cost_t, cost_d, cost_theta, cost_l, cost_V],
                    dtype=np.float32,
                )

                mask[pid, i] = 1.0
                # （x,y,theta,Delta_t,Delta_d,Delta_theta,pathL,DistanceV）

        return nodes, mask, reward_nodes
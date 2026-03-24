from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CandidateColumns:
    te: int = 0
    tp: int = 1
    cost: int = 8
    inferred_value_id: int = 9
    eid_ref: int = 11
    pid_ref: int = 12


class TODCObservationGenerator:
    """Observation builder that outputs model-aligned tensors directly."""

    def __init__(self, candidate_limit: Optional[int] = None, candidate_cols: Optional[CandidateColumns] = None):
        self.candidate_limit = int(candidate_limit) if candidate_limit is not None else None
        self.cols = candidate_cols or CandidateColumns()

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
    ) -> Dict[str, np.ndarray]:
        v_p_nodes = self._build_p_nodes(pos_p, v_p)
        v_e_nodes = self._build_e_nodes(pos_e, v_e, value_pos, inferred_targets)
        v_c_nodes, v_c_mask = self._build_c_nodes(
            ic_candidates=ic_candidates,
            num_p=num_p,
            num_e=num_e,
            pos_p=pos_p,
            pos_e=pos_e,
            value_pos=value_pos,
            candidate_pos_fn=candidate_pos_fn,
        )

        model_obs = self._build_model_aligned_obs(
            v_p_nodes=v_p_nodes,
            v_e_nodes=v_e_nodes,
            v_c_nodes=v_c_nodes,
            v_c_mask=v_c_mask,
            value_pos=value_pos,
        )
        return model_obs

    def _build_model_aligned_obs(
        self,
        *,
        v_p_nodes: np.ndarray,
        v_e_nodes: np.ndarray,
        v_c_nodes: np.ndarray,
        v_c_mask: np.ndarray,
        value_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build obs tensors already aligned to UAVInterceptionNetwork.forward inputs."""
        num_p = v_p_nodes.shape[0]
        num_e = v_e_nodes.shape[0]
        k_max = v_c_nodes.shape[1]

        # self_uav: [P, 1, 3] -> (x, y, theta)
        theta_p = np.arctan2(v_p_nodes[:, 5], v_p_nodes[:, 4])
        self_uav = np.stack([v_p_nodes[:, 0], v_p_nodes[:, 1], theta_p], axis=-1)[:, None, :].astype(np.float32)

        # ally_uavs: [P, max(1, P-1), 3], ally_mask: [P, max(1, P-1)]
        ally_slots = max(1, num_p - 1)
        ally_uavs = np.zeros((num_p, ally_slots, 3), dtype=np.float32)
        ally_mask = np.zeros((num_p, ally_slots), dtype=bool)
        for pid in range(num_p):
            others = [i for i in range(num_p) if i != pid]
            if len(others) == 0:
                continue
            vals = np.stack([v_p_nodes[others, 0], v_p_nodes[others, 1], theta_p[others]], axis=-1).astype(np.float32)
            ally_uavs[pid, : len(others)] = vals
            ally_mask[pid, : len(others)] = True

        # self_pts: [P, K, 8] = (x,y,theta,Delta_t,Delta_d,Delta_theta,pathL,DistanceV)
        self_pts = np.asarray(v_c_nodes, dtype=np.float32)
        self_pts_mask = v_c_mask > 0.0

        # ally_pts: concat others' candidate points for each pursuer
        ally_pts_len = max(1, (num_p - 1) * k_max)
        ally_pts = np.zeros((num_p, ally_pts_len, 8), dtype=np.float32)
        ally_pts_mask = np.zeros((num_p, ally_pts_len), dtype=bool)
        for pid in range(num_p):
            others = [i for i in range(num_p) if i != pid]
            if len(others) == 0:
                continue
            block = self_pts[others].reshape(len(others) * k_max, 8)
            block_mask = self_pts_mask[others].reshape(len(others) * k_max)
            ally_pts[pid, : block.shape[0]] = block
            ally_pts_mask[pid, : block.shape[0]] = block_mask

        # enemies: [P, E, 3] -> (x, y, theta), replicated per pursuer
        theta_e = np.arctan2(v_e_nodes[:, 5], v_e_nodes[:, 4])
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

        return {
            "self_uav": self_uav,
            "ally_uavs": ally_uavs,
            "self_pts": self_pts,
            "ally_pts": ally_pts,
            "enemies": enemies,
            "targets": targets,
            "assets": assets,
            "ally_mask": ally_mask,
            "self_pts_mask": self_pts_mask,
            "ally_pts_mask": ally_pts_mask,
            "enemy_mask": enemy_mask,
            "target_mask": target_mask,
            "asset_mask": asset_mask,
        }

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        subsets = []
        max_k = 0
        for pid in range(num_p):
            subset = np.empty((0, 0), dtype=float)
            if ic_candidates is not None and ic_candidates.size > 0:
                subset = ic_candidates
                if ic_candidates.shape[1] > self.cols.pid_ref:
                    subset = ic_candidates[ic_candidates[:, self.cols.pid_ref].astype(int) == pid]
            subsets.append(subset)
            max_k = max(max_k, int(subset.shape[0]) if subset.ndim > 0 else 0)

        if self.candidate_limit is not None:
            max_k = min(max_k, self.candidate_limit)
        max_k = max(1, max_k)

        nodes = np.zeros((num_p, max_k, 8), dtype=np.float32)
        mask = np.zeros((num_p, max_k), dtype=np.float32)

        for pid in range(num_p):
            subset = subsets[pid]
            if subset.size == 0:
                continue

            n = min(max_k, subset.shape[0])
            for i in range(n):
                row = subset[i]
                if candidate_pos_fn is not None:
                    c_x, c_y = candidate_pos_fn(row, pid)
                else:
                    eid = int(row[self.cols.eid_ref]) if row.shape[0] > self.cols.eid_ref else pid % max(1, num_e)
                    eid = max(0, min(eid, num_e - 1))
                    c_x, c_y = float(pos_e[eid, 0]), float(pos_e[eid, 1])

                t_p = float(row[self.cols.tp]) if row.shape[0] > self.cols.tp else 0.0
                t_e = float(row[self.cols.te]) if row.shape[0] > self.cols.te else 0.0
                path_l = float(row[self.cols.cost]) if row.shape[0] > self.cols.cost else 0.0
                delta_t = t_e - t_p

                p_x, p_y, p_th = float(pos_p[pid, 0]), float(pos_p[pid, 1]), float(pos_p[pid, 2])
                vec_x, vec_y = c_x - p_x, c_y - p_y
                theta = float(np.arctan2(vec_y, vec_x)) if (abs(vec_x) + abs(vec_y)) > 1e-9 else p_th

                # Approximate intercept deviation features from available geometry
                delta_d = float(np.hypot(vec_x, vec_y))
                delta_theta = float(abs(np.arctan2(np.sin(theta - p_th), np.cos(theta - p_th))))
                distance_v = float(np.min(np.linalg.norm(value_pos[:, :2] - np.array([c_x, c_y]), axis=1)))

                nodes[pid, i] = np.array(
                    [c_x, c_y, theta, delta_t, delta_d, delta_theta, path_l, distance_v],
                    dtype=np.float32,
                )
                mask[pid, i] = 1.0

        return nodes, mask
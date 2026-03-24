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
    """Phase-2 observation builder: V_P, V_E, V_C + padding/mask."""

    def __init__(self, k_max: int, candidate_cols: Optional[CandidateColumns] = None):
        self.k_max = int(k_max)
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
            pos_e=pos_e,
            candidate_pos_fn=candidate_pos_fn,
        )

        # Keep both phase-2 names and existing env names for compatibility.
        return {
            "V_P": v_p_nodes,
            "V_E": v_e_nodes,
            "V_C": v_c_nodes,
            "V_C_mask": v_c_mask,
            "pursuers": v_p_nodes,
            "evaders": v_e_nodes,
            "candidates": v_c_nodes,
            "candidate_mask": v_c_mask,
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
        pos_e: np.ndarray,
        candidate_pos_fn: Optional[Callable[[np.ndarray, int], Tuple[float, float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        nodes = np.zeros((num_p, self.k_max, 6), dtype=np.float32)
        mask = np.zeros((num_p, self.k_max), dtype=np.float32)

        if ic_candidates is None or ic_candidates.size == 0:
            return nodes, mask

        for pid in range(num_p):
            subset = ic_candidates
            if ic_candidates.shape[1] > self.cols.pid_ref:
                subset = ic_candidates[ic_candidates[:, self.cols.pid_ref].astype(int) == pid]

            if subset.shape[0] == 0:
                continue

            n = min(self.k_max, subset.shape[0])
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
                c_cost = float(row[self.cols.cost]) if row.shape[0] > self.cols.cost else 0.0
                delta_t = t_e - t_p

                nodes[pid, i] = np.array([c_x, c_y, t_p, t_e, delta_t, c_cost], dtype=np.float32)
                mask[pid, i] = 1.0

        return nodes, mask
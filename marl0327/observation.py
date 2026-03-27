"""
将 main0319forRL 中 decision_outputs 条目编码为与 models.UAVInterceptionNetwork 一致的张量字典。

约定：
- self_uav / ally_uavs / enemies: (x, y, theta)，单位与仿真一致。
- targets: (x, y)。
- self_pts / ally_pts: 8 维特征（前 3 维为 Iso 点 x,y,theta，其余来自 InterceptCandidates 尾部列，不足则右侧零填充）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _pad2d(arr: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    out = np.zeros((target_rows, target_cols), dtype=np.float32)
    if arr.size == 0:
        return out
    r = min(arr.shape[0], target_rows)
    c = min(arr.shape[1], target_cols)
    out[:r, :c] = arr[:r, :c].astype(np.float32)
    return out


def _ensure_dim3(states: np.ndarray) -> np.ndarray:
    """邻居 states 统一为 (N, 3)。"""
    if states.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    s = np.atleast_2d(states.astype(np.float32))
    if s.shape[1] < 3:
        pad = np.zeros((s.shape[0], 3 - s.shape[1]), dtype=np.float32)
        s = np.hstack([s, pad])
    return s[:, :3]


def encode_single_pursuer_obs(
    pid: int,
    decision_entry: Dict[str, Any],
    intercept_candidates: np.ndarray,
    output_iso_p: np.ndarray,
    cfg_max: Tuple[int, int, int, int, int],
) -> Dict[str, np.ndarray]:
    """
    将单架 pursuer 的观测编码为定长 numpy 数组（再转 tensor）。

    Args:
        pid: pursuer ID（与 Map 中一致）。
        decision_entry: 含 output_P_state, output_P_alley_state, output_enemy_state,
            output_V_state 等的主循环字典项。
        intercept_candidates: 与 output_iso_p 行一一对应的 IC 矩阵（用于按 Pid 筛选）。
        output_iso_p: obtain_output_iso 返回的数组，shape (M, C)，C>=3。
        cfg_max: (max_allies, max_enemies, max_targets, max_self_pts, max_ally_pts)
    """
    max_allies, max_enemies, max_targets, max_self_pts, max_ally_pts = cfg_max

    out_p = decision_entry["output_P_state"]
    self_uav = np.asarray(out_p[int(pid)], dtype=np.float32).reshape(1, 3)

    ally_d = decision_entry["output_P_alley_state"].get(int(pid), {"states": np.empty((0, 0))})
    ally_raw = _ensure_dim3(np.asarray(ally_d["states"]))
    # 去掉与自身重合的点（若有）
    allies = np.array([r for r in ally_raw if np.linalg.norm(r[:2] - self_uav[0, :2]) > 1e-6], dtype=np.float32)
    ally_uavs = _pad2d(allies, max_allies, 3)

    en_d = decision_entry["output_enemy_state"].get(int(pid), {"states": np.empty((0, 0))})
    enemies = _ensure_dim3(np.asarray(en_d["states"]))
    enemies_arr = _pad2d(enemies, max_enemies, 3)

    v_d = decision_entry["output_V_state"].get(int(pid), {"states": np.empty((0, 0))})
    tv = np.asarray(v_d["states"], dtype=np.float32)
    if tv.size == 0:
        targets = np.zeros((max_targets, 2), dtype=np.float32)
    else:
        tv = np.atleast_2d(tv)
        targets = _pad2d(tv[:, :2], max_targets, 2)

    # 按 Pid 筛选候选拦截点
    if intercept_candidates.size == 0 or output_iso_p is None or output_iso_p.size == 0:
        self_pts = np.zeros((max_self_pts, 8), dtype=np.float32)
        m_self = np.zeros((max_self_pts,), dtype=bool)
    else:
        pcol = intercept_candidates[:, 12].astype(int)
        mask = pcol == int(pid)
        rows = output_iso_p[mask]
        rows = np.atleast_2d(rows.astype(np.float32))
        if rows.shape[1] < 8:
            rows = np.hstack([rows, np.zeros((rows.shape[0], 8 - rows.shape[1]), dtype=np.float32)])
        else:
            rows = rows[:, :8]
        self_pts = _pad2d(rows, max_self_pts, 8)
        n_valid = min(int(mask.sum()), max_self_pts)
        m_self = np.zeros((max_self_pts,), dtype=bool)
        m_self[:n_valid] = True

    # 友机候选点：可用其他 pid 的 Iso 行聚合；占位为零（由主循环后续补全）
    ally_pts = np.zeros((max_ally_pts, 8), dtype=np.float32)
    m_ally = np.zeros((max_ally_pts,), dtype=bool)

    ally_mask = np.zeros((max_allies,), dtype=bool)
    ally_mask[: min(len(allies), max_allies)] = True

    enemy_mask = np.zeros((max_enemies,), dtype=bool)
    enemy_mask[: min(enemies.shape[0], max_enemies)] = True

    n_tg = int(tv.shape[0]) if tv.size and tv.ndim >= 1 else 0
    target_mask = np.zeros((max_targets,), dtype=bool)
    target_mask[: min(n_tg, max_targets)] = True

    return {
        "self_uav": self_uav,
        "ally_uavs": ally_uavs,
        "self_pts": self_pts,
        "ally_pts": ally_pts,
        "enemies": enemies_arr,
        "targets": targets,
        "ally_mask": ally_mask,
        "self_pts_mask": m_self,
        "ally_pts_mask": m_ally,
        "enemy_mask": enemy_mask,
        "target_mask": target_mask,
    }


def encode_single_pursuer_obs_ep(
    pid: int,
    e_ref: int,
    p_ref: int,
    decision_entry: Dict[str, Any],
    intercept_candidates: np.ndarray,
    output_iso_p: np.ndarray,
    cfg_max: Tuple[int, int, int, int, int],
) -> Dict[str, np.ndarray]:
    """
    仅编码指定 (E_ref, P_ref) 对应的拦截候选（用于分配已固定后在该子集中选点）。
    """
    max_allies, max_enemies, max_targets, max_self_pts, max_ally_pts = cfg_max

    out_p = decision_entry["output_P_state"]
    self_uav = np.asarray(out_p[int(pid)], dtype=np.float32).reshape(1, 3)

    ally_d = decision_entry["output_P_alley_state"].get(int(pid), {"states": np.empty((0, 0))})
    ally_raw = _ensure_dim3(np.asarray(ally_d["states"]))
    allies = np.array(
        [r for r in ally_raw if np.linalg.norm(r[:2] - self_uav[0, :2]) > 1e-6],
        dtype=np.float32,
    )
    ally_uavs = _pad2d(allies, max_allies, 3)

    en_d = decision_entry["output_enemy_state"].get(int(pid), {"states": np.empty((0, 0))})
    enemies = _ensure_dim3(np.asarray(en_d["states"]))
    enemies_arr = _pad2d(enemies, max_enemies, 3)

    v_d = decision_entry["output_V_state"].get(int(pid), {"states": np.empty((0, 0))})
    tv = np.asarray(v_d["states"], dtype=np.float32)
    if tv.size == 0:
        targets = np.zeros((max_targets, 2), dtype=np.float32)
    else:
        tv = np.atleast_2d(tv)
        targets = _pad2d(tv[:, :2], max_targets, 2)

    ec = intercept_candidates[:, 11].astype(int)
    pc = intercept_candidates[:, 12].astype(int)
    mask = (ec == int(e_ref)) & (pc == int(p_ref))

    if intercept_candidates.size == 0 or output_iso_p is None or output_iso_p.size == 0 or not mask.any():
        self_pts = np.zeros((max_self_pts, 8), dtype=np.float32)
        m_self = np.zeros((max_self_pts,), dtype=bool)
    else:
        rows = output_iso_p[mask]
        rows = np.atleast_2d(rows.astype(np.float32))
        if rows.shape[1] < 8:
            rows = np.hstack([rows, np.zeros((rows.shape[0], 8 - rows.shape[1]), dtype=np.float32)])
        else:
            rows = rows[:, :8]
        self_pts = _pad2d(rows, max_self_pts, 8)
        n_valid = min(int(mask.sum()), max_self_pts)
        m_self = np.zeros((max_self_pts,), dtype=bool)
        m_self[:n_valid] = True

    ally_pts = np.zeros((max_ally_pts, 8), dtype=np.float32)
    m_ally = np.zeros((max_ally_pts,), dtype=bool)

    ally_mask = np.zeros((max_allies,), dtype=bool)
    ally_mask[: min(len(allies), max_allies)] = True

    enemy_mask = np.zeros((max_enemies,), dtype=bool)
    enemy_mask[: min(enemies.shape[0], max_enemies)] = True

    n_tg = int(tv.shape[0]) if tv.size and tv.ndim >= 1 else 0
    target_mask = np.zeros((max_targets,), dtype=bool)
    target_mask[: min(n_tg, max_targets)] = True

    return {
        "self_uav": self_uav,
        "ally_uavs": ally_uavs,
        "self_pts": self_pts,
        "ally_pts": ally_pts,
        "enemies": enemies_arr,
        "targets": targets,
        "ally_mask": ally_mask,
        "self_pts_mask": m_self,
        "ally_pts_mask": m_ally,
        "enemy_mask": enemy_mask,
        "target_mask": target_mask,
    }


def numpy_obs_to_torch(batch: List[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    """将一批 encode_single_pursuer_obs 结果堆叠为 Batched 张量。"""
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        arrs = [b[k] for b in batch]
        if k.endswith("_mask"):
            out[k] = torch.from_numpy(np.stack(arrs, axis=0)).bool().to(device)
        else:
            t = np.stack(arrs, axis=0)
            out[k] = torch.from_numpy(t).float().to(device)
    # self_uav 每样本已是 (1,3)，stack 后为 [B,1,3]
    return out

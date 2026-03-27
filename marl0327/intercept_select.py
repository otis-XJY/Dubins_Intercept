"""
在匈牙利任务分配确定后，对每个 (E_ref, P_ref) 从完整 InterceptCandidates 子集中用策略网络选择拦截行。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from marl0327.config import MARLConfig
from marl0327.models import UAVInterceptionNetwork
from marl0327.observation import encode_single_pursuer_obs_ep, numpy_obs_to_torch


def refine_icfinal_with_model(
    IC: np.ndarray,
    AssignedIntercepts: np.ndarray,
    output_iso_p: np.ndarray,
    decision_entry: Dict[str, Any],
    UnCapPid: np.ndarray,
    model: nn.Module,
    device: torch.device,
    cfg: MARLConfig,
) -> np.ndarray:
    """
    保持 AssignedIntercepts 所代表的「谁追谁」不变，仅将每行对应的拦截候选
    从「字典序启发式一行」替换为「网络在相同 (E,P) 候选子集上的选择」。

    Args:
        IC: 完整拦截候选表（与 output_iso_p 行对齐）。
        AssignedIntercepts: 匈牙利输出，每行对应一对分配。
        output_iso_p: obtain_output_iso(IC, ...) 的结果。
        decision_entry: 与 decision_outputs 单步条目同结构。
        UnCapPid: 当前未捕获 pursuer 的全局 ID 数组（与主脚本一致，用 p_ref 索引）。
    """
    if IC.size == 0 or len(AssignedIntercepts) == 0:
        return AssignedIntercepts.copy()

    cfg_max = (
        cfg.max_allies,
        cfg.max_enemies,
        cfg.max_targets,
        cfg.max_self_pts,
        cfg.max_ally_pts,
    )

    n = len(AssignedIntercepts)
    out_rows = np.empty_like(AssignedIntercepts)

    need_model: List[Tuple[int, np.ndarray, Dict[str, np.ndarray]]] = []

    for i in range(n):
        e_ref = int(AssignedIntercepts[i, 11])
        p_ref = int(AssignedIntercepts[i, 12])
        mask = (IC[:, 11].astype(int) == e_ref) & (IC[:, 12].astype(int) == p_ref)
        ic_sub = IC[mask]
        if ic_sub.shape[0] == 0:
            out_rows[i] = AssignedIntercepts[i]
            continue
        if ic_sub.shape[0] == 1:
            out_rows[i] = ic_sub[0]
            continue

        global_pid = int(UnCapPid[p_ref])
        obs = encode_single_pursuer_obs_ep(
            global_pid,
            e_ref,
            p_ref,
            decision_entry,
            IC,
            output_iso_p,
            cfg_max,
        )
        need_model.append((i, ic_sub, obs))

    if not need_model:
        return out_rows

    batch = numpy_obs_to_torch([t[2] for t in need_model], device)
    model.eval()
    with torch.no_grad():
        mod = model.module if hasattr(model, "module") else model
        pred = mod(batch)
        logits = pred["action_logits"]

    for j, (i, ic_sub, obs) in enumerate(need_model):
        n_valid = int(obs["self_pts_mask"].sum())
        if n_valid <= 0:
            out_rows[i] = ic_sub[0]
            continue
        local = int(logits[j, :n_valid].argmax().item())
        local = min(local, ic_sub.shape[0] - 1)
        out_rows[i] = ic_sub[local]

    return out_rows


def build_rl_model(cfg: MARLConfig, device: torch.device) -> UAVInterceptionNetwork:
    return UAVInterceptionNetwork(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        design_mode=cfg.design_mode,
    ).to(device)

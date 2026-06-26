"""离线 warm-start：规则标签采样 + 行为克隆预热。

标签生成流程与 main0319forRL.py 对齐：
1. 从 env.IC_candidates 按 (E_ref, P_ref) 分组，每组 lexsort 选最优行
2. 在最优行上构建代价矩阵，运行匈牙利算法分配 E-P 配对
3. 对每个 P，在其 obs 子集（self_pts / reward_nodes）中定位匈牙利选定行的索引
4. 未被匈牙利覆盖的 P，退而求其次取 obs 子集内 cost 最优行
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from marl.envs.todc_env import TODCMARLEnv
from marl.obs.generator import CandidateColumns


@dataclass
class WarmStartConfig:
    enable: bool = False
    episodes: int = 0
    rollout_steps: int = 0
    minibatch_size: int = 32
    temporal_window: int = 0
    hidden_dim: int = 128


class SelfCtxHistoryBuffer:
    """管理每步 self_ctx 的时序历史缓存，用于 Design D 多帧输入。"""

    def __init__(self, temporal_window: int, num_pursuers: int):
        self.temporal_window = max(0, int(temporal_window))
        self.num_pursuers = int(num_pursuers)
        self._buf: List[np.ndarray] = []
        self._mask_buf: List[np.ndarray] = []

    def append(self, self_ctx: np.ndarray) -> None:
        self._buf.append(self_ctx.copy())
        self._mask_buf.append(np.ones(self.num_pursuers, dtype=bool))
        max_hist = max(0, self.temporal_window - 1)
        while len(self._buf) > max_hist:
            self._buf.pop(0)
            self._mask_buf.pop(0)

    def get_history(self) -> Optional[np.ndarray]:
        if len(self._buf) == 0:
            return None
        return np.stack(self._buf, axis=0).transpose(1, 0, 2)

    def get_history_mask(self) -> Optional[np.ndarray]:
        if len(self._mask_buf) == 0:
            return None
        return np.stack(self._mask_buf, axis=0).T


# ──────────────────────────────────────────────
# 规则标签：分组筛优 + 匈牙利全局一致性
# ──────────────────────────────────────────────

_COLS = CandidateColumns()


def _rule_actions_from_hungarian(
    obs_np: Dict[str, np.ndarray],
    env: TODCMARLEnv,
) -> np.ndarray:
    """基于 env.IC_candidates 分组筛优 + 匈牙利分配，返回每 P 的动作索引 (num_P,) int64。

    流程与 main0319forRL.py / todc_env._apply_hungarian_and_paths 对齐：
    1. 按 (E_ref, P_ref) 分组，lexsort(-Delta_t, cost, Delta_d) 选每对最优行
    2. 在最优行上构建代价矩阵，linear_sum_assignment 做 E-P 分配
    3. 对每个已分配 P，在其 obs 子集中定位匈牙利选定行
    4. 未被匈牙利覆盖的 P，取 obs 子集内 lexsort 最优行
    """
    num_P = env.num_P
    active = np.asarray(obs_np["pursuer_active"], dtype=bool).reshape(-1)
    mask = np.asarray(obs_np["self_pts_mask"], dtype=bool)
    reward_nodes = np.asarray(obs_np["reward_nodes"], dtype=np.float32)
    self_pts = np.asarray(obs_np["self_pts"], dtype=np.float32)

    rule_idx = np.full(num_P, -1, dtype=np.int64)

    # ── 快速路径：无候选 ──
    ic_candidates = env.IC_candidates
    if ic_candidates is None or ic_candidates.ndim != 2 or ic_candidates.shape[0] == 0:
        # 无候选时，活跃 P 取第一个有效位
        for p in range(num_P):
            if active[p] and np.any(mask[p]):
                rule_idx[p] = int(np.argmax(mask[p]))
        return rule_idx

    # ── 第 1 步：按 (E_ref, P_ref) 分组，每组 lexsort 选最优 ──
    eid_ref = ic_candidates[:, _COLS.eid_ref].astype(np.int64)
    pid_ref = ic_candidates[:, _COLS.pid_ref].astype(np.int64)

    # 构造复合键用于分组
    pair_key = eid_ref.astype(np.int64) * (pid_ref.max() + 2) + pid_ref.astype(np.int64)
    sorted_by_key = np.argsort(pair_key, kind="mergesort")
    sorted_keys = pair_key[sorted_by_key]

    # 找到每个唯一键的边界
    change_mask = np.ones(len(sorted_keys), dtype=bool)
    change_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = np.where(change_mask)[0]
    group_ends = np.append(group_starts[1:], len(sorted_keys))

    best_rows = []
    best_e_refs = []
    best_p_refs = []
    for gs, ge in zip(group_starts, group_ends):
        group_global_idx = sorted_by_key[gs:ge]
        group = ic_candidates[group_global_idx]
        # lexsort: 末尾键为主键 → (-Delta_t, cost, Delta_d)
        # ic_candidates 列 15=Delta_t, 8=cost, 16=Delta_d
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
        # 回退：活跃 P 取 obs 子集内 lexsort 最优
        return _fallback_local_best(active, mask, self_pts, reward_nodes, rule_idx, num_P)

    best_rows = np.array(best_rows, dtype=np.int64)
    ic_best = ic_candidates[best_rows]
    best_e_refs = np.array(best_e_refs, dtype=np.int64)
    best_p_refs = np.array(best_p_refs, dtype=np.int64)

    # ── 第 2 步：匈牙利分配 ──
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

    # 匈牙利选中的 ic_best 行索引 → 全局 ic_candidates 行索引
    hung_ic_best_idx = idx_mat[p_hung[valid], e_hung[valid]]
    hung_global_rows = best_rows[hung_ic_best_idx]
    # 对应的 compact (E_ref, P_ref)
    hung_e = best_e_refs[hung_ic_best_idx]
    hung_p = best_p_refs[hung_ic_best_idx]

    # ── 第 3 步：映射到 obs 子集索引 ──
    # 对于被匈牙利选中的 P，在其 obs 子集 (self_pts / reward_nodes) 中
    # 找到匈牙利选定行对应的候选点索引。
    # obs 子集已按 (E_ref, P_ref) 过滤，需要匹配候选点特征。
    pairs_realE2P = env.pairs_realE2P
    pairs_ic_compact = env._pairs_ic_compact

    # 构建 pid_global → (e_compact, p_compact) 映射
    pid_to_compact: Dict[int, Tuple[int, int]] = {}
    if pairs_realE2P is not None and pairs_ic_compact is not None:
        for row_i in range(pairs_realE2P.shape[0]):
            eg = int(pairs_realE2P[row_i, 1])  # pid_global
            ec = int(pairs_ic_compact[row_i, 0])
            pc = int(pairs_ic_compact[row_i, 1])
            pid_to_compact[eg] = (ec, pc)

    # 构建 compact→hung_global_row 映射
    hung_compact_map: Dict[Tuple[int, int], int] = {}
    for i in range(len(hung_e)):
        key = (int(hung_e[i]), int(hung_p[i]))
        hung_compact_map[key] = int(hung_global_rows[i])

    for p in range(num_P):
        if not active[p] or not np.any(mask[p]):
            continue
        compact = pid_to_compact.get(p)
        if compact is None:
            continue
        hung_row = hung_compact_map.get(compact)
        if hung_row is None:
            continue
        # 在 obs 子集中找到匹配行：比较候选点坐标 (x,y) 与 theta
        # self_pts[p, k] = [c_x, c_y, theta, delta_t, delta_d, delta_theta, path_l, distance_V]
        target_x = float(ic_candidates[hung_row, 0])
        target_y = float(ic_candidates[hung_row, 1])
        # 匹配条件：距离最近
        k_valid = np.where(mask[p])[0]
        if len(k_valid) == 0:
            continue
        dists = np.sqrt(
            (self_pts[p, k_valid, 0] - target_x) ** 2
            + (self_pts[p, k_valid, 1] - target_y) ** 2
        )
        best_k = k_valid[np.argmin(dists)]
        rule_idx[p] = int(best_k)

    # ── 第 4 步：未被匈牙利覆盖的 P，取 obs 子集内 lexsort 最优 ──
    for p in range(num_P):
        if not active[p] or not np.any(mask[p]):
            continue
        if rule_idx[p] >= 0:
            continue
        k_valid = np.where(mask[p])[0]
        if len(k_valid) == 0:
            continue
        # lexsort: -delta_t (col 3), cost_t (reward_nodes col 3), delta_d (self_pts col 4)
        sub_self = self_pts[p, k_valid]
        sub_reward = reward_nodes[p, k_valid]
        sort_idx = np.lexsort((sub_self[:, 4], sub_reward[:, 3], -sub_self[:, 3]))
        rule_idx[p] = int(k_valid[sort_idx[0]])

    # 非活跃 P 保证 index=0（mask 全 False 时不会被使用）
    for p in range(num_P):
        if not active[p]:
            rule_idx[p] = 0

    return rule_idx


def _fallback_local_best(
    active: np.ndarray,
    mask: np.ndarray,
    self_pts: np.ndarray,
    reward_nodes: np.ndarray,
    rule_idx: np.ndarray,
    num_P: int,
) -> np.ndarray:
    """匈牙利不可用时的回退：每个活跃 P 在其 obs 子集内取 lexsort 最优。"""
    for p in range(num_P):
        if not active[p] or not np.any(mask[p]):
            continue
        k_valid = np.where(mask[p])[0]
        if len(k_valid) == 0:
            continue
        sub_self = self_pts[p, k_valid]
        sub_reward = reward_nodes[p, k_valid]
        sort_idx = np.lexsort((sub_self[:, 4], sub_reward[:, 3], -sub_self[:, 3]))
        rule_idx[p] = int(k_valid[sort_idx[0]])
    for p in range(num_P):
        if not active[p]:
            rule_idx[p] = 0
    return rule_idx


# ──────────────────────────────────────────────
# 轨迹采集与 BC 更新
# ──────────────────────────────────────────────


def _collect_rule_rollout(
    model: torch.nn.Module,
    env: TODCMARLEnv,
    device: torch.device,
    build_obs_fn: Callable[[Dict[str, np.ndarray], torch.device], Dict[str, torch.Tensor]],
    *,
    max_steps: int,
    temporal_window: int,
    hidden_dim: int,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """采集规则策略轨迹，用于 warm-start 行为克隆。"""
    if max_steps <= 0:
        return [], np.empty((0, 0), dtype=np.int64), None, None

    unwrap = model.module if hasattr(model, "module") else model
    obs_np, _ = env.reset()
    done = False
    trunc = False

    ro_obs: List[Dict[str, np.ndarray]] = []
    ro_rule_actions: List[np.ndarray] = []
    ro_self_ctx: List[np.ndarray] = []

    temporal_enabled = temporal_window > 1
    ctx_hist_buf: Optional[SelfCtxHistoryBuffer] = None
    if temporal_enabled:
        ctx_hist_buf = SelfCtxHistoryBuffer(temporal_window, env.num_P)

    while (not done) and (not trunc) and (len(ro_obs) < max_steps):
        ro_obs.append(obs_np)
        rule_idx = _rule_actions_from_hungarian(obs_np, env)
        ro_rule_actions.append(rule_idx)

        if temporal_enabled:
            obs_t = build_obs_fn(obs_np, device)
            hist = None
            hist_mask = None
            if ctx_hist_buf is not None:
                _hist = ctx_hist_buf.get_history()
                _hist_mask = ctx_hist_buf.get_history_mask()
                if _hist is not None:
                    hist = torch.as_tensor(_hist, dtype=torch.float32, device=device)
                    hist_mask = torch.as_tensor(_hist_mask, dtype=torch.bool, device=device)
            with torch.no_grad():
                ao = unwrap.actor_forward(obs_t, self_ctx_history=hist, self_ctx_history_mask=hist_mask)
                self_ctx_t = ao.get("self_ctx")
            if self_ctx_t is not None:
                self_ctx_np = self_ctx_t.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                self_ctx_np = np.zeros((env.num_P, hidden_dim), dtype=np.float32)
            ro_self_ctx.append(self_ctx_np)
            if ctx_hist_buf is not None:
                ctx_hist_buf.append(self_ctx_np)

        step_act = rule_idx.astype(np.int64, copy=False)
        next_obs_np, _rew, terms, truncs, _infos = env.step(step_act)
        obs_np = next_obs_np
        done = bool(terms["__all__"])
        trunc = bool(truncs["__all__"])

    if len(ro_obs) == 0:
        return ro_obs, np.empty((0, env.num_P), dtype=np.int64), None, None

    actions = np.stack(ro_rule_actions, axis=0)
    if len(ro_self_ctx) > 0:
        self_ctx_seq = np.stack(ro_self_ctx, axis=0).astype(np.float32, copy=False)
        self_ctx_mask = np.ones((self_ctx_seq.shape[0], env.num_P), dtype=bool)
    else:
        self_ctx_seq = None
        self_ctx_mask = None
    return ro_obs, actions, self_ctx_seq, self_ctx_mask


def _behavior_clone_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_list: List[Dict[str, np.ndarray]],
    actions: np.ndarray,
    device: torch.device,
    build_obs_fn: Callable[[Dict[str, np.ndarray], torch.device], Dict[str, torch.Tensor]],
    *,
    minibatch_size: int,
    temporal_window: int,
    self_ctx_seq: Optional[np.ndarray] = None,
    self_ctx_seq_mask: Optional[np.ndarray] = None,
) -> float:
    """对规则轨迹执行一轮行为克隆更新，返回平均交叉熵。"""
    t_max = int(actions.shape[0])
    if t_max <= 0:
        return 0.0

    unwrap = model.module if hasattr(model, "module") else model
    order = np.arange(t_max)
    np.random.shuffle(order)
    use_temporal = (
        temporal_window > 1
        and self_ctx_seq is not None
        and self_ctx_seq_mask is not None
    )

    tot_loss = 0.0
    n_mb = 0
    mb_size = max(1, int(minibatch_size))

    for start in range(0, t_max, mb_size):
        batch = order[start : start + mb_size]
        if len(batch) == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        mb_loss = 0.0
        n_valid = 0

        for t in batch:
            obs_t_np = obs_list[int(t)]
            obs_t = build_obs_fn(obs_t_np, device)
            active_np = np.asarray(obs_t_np["pursuer_active"], dtype=bool).reshape(-1)
            active_t = torch.as_tensor(active_np, dtype=torch.bool, device=device)
            if not bool(torch.any(active_t)):
                continue

            ctx_history = None
            ctx_history_mask = None
            if use_temporal:
                t_start = max(0, int(t) - int(temporal_window) + 1)
                hist_len = int(t) - t_start
                if hist_len > 0:
                    ctx_history = torch.as_tensor(
                        self_ctx_seq[t_start:int(t)], dtype=torch.float32, device=device
                    ).permute(1, 0, 2)
                    ctx_history_mask = torch.as_tensor(
                        self_ctx_seq_mask[t_start:int(t)], dtype=torch.bool, device=device
                    ).permute(1, 0)

            out = unwrap.actor_forward(
                obs_t,
                self_ctx_history=ctx_history,
                self_ctx_history_mask=ctx_history_mask,
            )
            logits = out["action_logits"]
            tgt = torch.as_tensor(actions[int(t)], dtype=torch.long, device=device)
            loss_t = F.cross_entropy(logits[active_t], tgt[active_t])
            mb_loss = mb_loss + loss_t
            n_valid += 1

        if n_valid == 0:
            continue

        loss = mb_loss / float(n_valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += float(loss.item())
        n_mb += 1

    if n_mb == 0:
        return 0.0
    return float(tot_loss / n_mb)


def run_rule_bc_warm_start(
    *,
    cfg,
    models: Dict[str, torch.nn.Module],
    envs: Dict[str, TODCMARLEnv],
    opts: Dict[str, torch.optim.Optimizer],
    device: torch.device,
    build_obs_fn: Callable[[Dict[str, np.ndarray], torch.device], Dict[str, torch.Tensor]],
    run=None,
    is_main: bool = True,
    rl_print: bool = True,
) -> None:
    """执行规则数据驱动的 BC warm-start。"""
    warm_cfg = WarmStartConfig(
        enable=bool(getattr(cfg, "warm_start_enable", False)),
        episodes=int(getattr(cfg, "warm_start_episodes", 0) or 0),
        rollout_steps=int(getattr(cfg, "warm_start_rollout_steps", 0) or 0),
        minibatch_size=int(getattr(cfg, "warm_start_minibatch_size", 32) or 32),
        temporal_window=int(getattr(cfg, "temporal_window", 0) or 0),
        hidden_dim=int(getattr(cfg, "hidden_dim", 128) or 128),
    )

    if (not warm_cfg.enable) or warm_cfg.episodes <= 0:
        return

    max_steps = warm_cfg.rollout_steps if warm_cfg.rollout_steps > 0 else int(getattr(cfg, "max_episode_steps", 200))
    mb_size = max(1, warm_cfg.minibatch_size)

    if is_main and rl_print:
        print(
            f"[WARM] start behavior cloning warm-start | episodes={warm_cfg.episodes} "
            f"rollout_steps={max_steps} minibatch={mb_size}"
        )

    for warm_ep in range(1, warm_cfg.episodes + 1):
        for scheme_name, model in models.items():
            model.train()
            env = envs[scheme_name]
            opt = opts[scheme_name]

            ro_obs, ro_rule_actions, ro_self_ctx, ro_self_ctx_mask = _collect_rule_rollout(
                model,
                env,
                device,
                build_obs_fn,
                max_steps=max_steps,
                temporal_window=warm_cfg.temporal_window,
                hidden_dim=warm_cfg.hidden_dim,
            )
            bc_loss = _behavior_clone_update(
                model,
                opt,
                ro_obs,
                ro_rule_actions,
                device,
                build_obs_fn,
                minibatch_size=max(1, min(mb_size, max(1, len(ro_obs)))),
                temporal_window=warm_cfg.temporal_window,
                self_ctx_seq=ro_self_ctx,
                self_ctx_seq_mask=ro_self_ctx_mask,
            )

            if run is not None:
                run.log(
                    {
                        f"{scheme_name}/warm_start_bc_loss": float(bc_loss),
                        f"{scheme_name}/warm_start_steps": int(len(ro_obs)),
                        "warm_episode": int(warm_ep),
                    }
                )
            if is_main and rl_print:
                print(
                    f"[WARM] ep={warm_ep}/{warm_cfg.episodes} scheme={scheme_name} "
                    f"bc_loss={bc_loss:.5f} steps={len(ro_obs)}"
                )

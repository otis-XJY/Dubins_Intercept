from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RewardConfig:
    # Interception quality terms
    alpha_dist_v: float = 1.0
    beta_delta_t: float = 1.0
    gamma_delta_d: float = 1.0
    gamma_delta_theta: float = 1.0
    lambda_path_l: float = 1.0

    # Global map-control terms
    w_global: float = 1.0
    div_sigma: float = 300.0
    assign_penalty: float = 5.0

    # Social safety terms
    safe_dist_min: float = 120.0
    safe_penalty_scale: float = 0.5

    # Efficiency and terminal terms
    step_cost: float = -0.05
    terminal_capture_bonus: float = 100.0
    terminal_asset_loss_penalty: float = 500.0
    asset_breach_radius: float = 40.0

    # Optional legacy progress term (disabled by default)
    dist_progress_scale: float = 0.0


class TODCRewardFunction:
    """Standalone reward module for TODC MARL env.

    Keep reward logic outside environment dynamics so reward tuning can be
    performed frequently with minimal side effects.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    @classmethod
    def from_env_config(cls, env_config: Optional[Dict] = None) -> "TODCRewardFunction":
        cfg = env_config or {}
        reward_cfg_raw = cfg.get("reward", {})
        reward_cfg = dict(reward_cfg_raw) if isinstance(reward_cfg_raw, dict) else {}
        # 只保留RewardConfig定义的字段
        valid_keys = set(RewardConfig.__annotations__.keys())
        reward_cfg = {k: v for k, v in reward_cfg.items() if k in valid_keys}
        return cls(RewardConfig(**reward_cfg))

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.config, k):
                raise ValueError(f"Unknown reward config key: {k}")
            setattr(self.config, k, float(v))

    def compute_step_rewards(
        self,
        *,
        actions: np.ndarray,
        obs: Dict[str, np.ndarray],
        curr_min_dist: np.ndarray,
        last_min_dist: Optional[np.ndarray],
        num_p: int,
        sim_time_elapsed: float,
        sim_dt: float,
        return_details: bool = False,
    ) -> Dict[str, float]:
        rewards = {f"p_{i}": 0.0 for i in range(num_p)}
        details = {f"p_{i}": {} for i in range(num_p)}

        prev_min_dist = curr_min_dist if last_min_dist is None else last_min_dist
        delta = prev_min_dist - curr_min_dist
        for i in range(num_p):
            details[f"p_{i}"]["r_progress"] = float(self.config.dist_progress_scale * delta[i])
            rewards[f"p_{i}"] += details[f"p_{i}"]["r_progress"]

        reward_nodes = np.asarray(obs["reward_nodes"], dtype=np.float32)
        mask = np.asarray(obs["self_pts_mask"], dtype=np.float32)
        enemies = np.asarray(obs["enemies"], dtype=np.float32)

        k_curr = reward_nodes.shape[1]
        selected_idx = np.zeros((num_p,), dtype=np.int64)
        # 支持两种动作输入格式：
        # - 每个 agent 的采样索引（形状 (num_p,) 或标量/整型）
        # - 每个 agent 的权重向量 / 概率分布（形状 (num_p, k_curr))
        # 非法索引、空掩码、维度不匹配时直接报错（与 MARL_env._normalize_action 一致，便于排查）
        actions_arr = np.asarray(actions)
        if actions_arr.ndim == 1:
            for i in range(num_p):
                idx = int(actions_arr[i])
                valid_idx = np.where(mask[i] > 0)[0]
                if valid_idx.size == 0:
                    raise ValueError(f"no valid reward candidates for pursuer {i} (self_pts_mask row is all zero)")
                if idx < 0 or idx >= k_curr or mask[i, idx] <= 0:
                    raise ValueError(f"invalid action index {idx} for pursuer {i} in reward (k={k_curr})")
                selected_idx[i] = idx
        else:
            for i in range(num_p):
                w = actions_arr[i]
                w = np.asarray(w, dtype=np.float32)
                if w.ndim == 0:
                    idx = int(w)
                    valid_idx = np.where(mask[i] > 0)[0]
                    if valid_idx.size == 0:
                        raise ValueError(f"no valid reward candidates for pursuer {i} (self_pts_mask row is all zero)")
                    if idx < 0 or idx >= k_curr or mask[i, idx] <= 0:
                        raise ValueError(f"invalid action index {idx} for pursuer {i} in reward (k={k_curr})")
                    selected_idx[i] = idx
                    continue
                if w.shape[0] != k_curr:
                    raise ValueError(
                        f"action weight length {w.shape[0]} != k_curr {k_curr} for pursuer {i}"
                    )
                weighted = w * mask[i]
                if np.sum(weighted) <= 1e-12:
                    raise ValueError(f"action weights for pursuer {i} are zero on all valid candidates")
                selected_idx[i] = int(np.argmax(weighted))

        selected = reward_nodes[np.arange(num_p), selected_idx]  # [P, 8]

        dist_v = selected[:, 7]
        delta_t = selected[:, 3]
        delta_d = selected[:, 4]
        delta_theta = selected[:, 5]
        path_l = selected[:, 6]

        r_qual = (
            self.config.alpha_dist_v * dist_v
            + self.config.beta_delta_t * delta_t
            + self.config.gamma_delta_d * delta_d
            + self.config.gamma_delta_theta * delta_theta
            + self.config.lambda_path_l * path_l
        )
        for i in range(num_p):
            details[f"p_{i}"]["r_qual"] = float(r_qual[i])

        sel_xy = selected[:, :2]
        r_div = np.zeros((num_p,), dtype=np.float32)
        if num_p > 1:
            for i in range(num_p):
                d2 = np.sum((sel_xy[i] - sel_xy) ** 2, axis=1)
                term = np.exp(-d2 / (self.config.div_sigma ** 2 + 1e-12))
                term[i] = 0.0
                r_div[i] = -float(np.sum(term))

        e_pos = enemies[0, :, :2] if enemies.ndim == 3 else np.zeros((0, 2), dtype=np.float32)
        r_assign = np.zeros((num_p,), dtype=np.float32)
        if e_pos.shape[0] > 0:
            nearest_enemy = np.argmin(np.linalg.norm(sel_xy[:, None, :] - e_pos[None, :, :], axis=2), axis=1)
            for eid in np.unique(nearest_enemy):
                ids = np.where(nearest_enemy == eid)[0]
                if ids.size > 1:
                    r_assign[ids] -= float(self.config.assign_penalty)

        r_global = self.config.w_global * (r_div + r_assign)
        for i in range(num_p):
            details[f"p_{i}"]["r_global"] = float(r_global[i])

        self_uav = np.asarray(obs["self_uav"], dtype=np.float32)
        self_pos = self_uav[:, 0, :2]
        r_safe = np.zeros((num_p,), dtype=np.float32)
        if num_p > 1:
            for i in range(num_p):
                d = np.linalg.norm(self_pos[i] - self_pos, axis=1)
                d[i] = np.inf
                d_min = float(np.min(d))
                if d_min < self.config.safe_dist_min:
                    r_safe[i] -= self.config.safe_penalty_scale * (self.config.safe_dist_min - d_min)
        for i in range(num_p):
            details[f"p_{i}"]["r_safe"] = float(r_safe[i])

        # 时间惩罚按仿真时间推进量计：step_cost * (sim_time_elapsed / sim_dt)，与 main0319 中 t/t_all 一致
        if sim_dt > 1e-12:
            r_time = float(self.config.step_cost) * (float(sim_time_elapsed) / float(sim_dt))
        else:
            r_time = 0.0
        for i in range(num_p):
            details[f"p_{i}"]["r_time"] = r_time

        for i in range(num_p):
            rewards[f"p_{i}"] += float(details[f"p_{i}"]["r_qual"] + details[f"p_{i}"]["r_global"] + details[f"p_{i}"]["r_safe"] + r_time)

        if return_details:
            return rewards, details
        return rewards

    def apply_terminal_rewards(
        self,
        rewards: Dict[str, float],
        *,
        captured_delta: int,
        num_e: int,
        asset_breached: bool,
        details: Optional[Dict[str, dict]] = None,
    ):
        terminal_bonus = 0.0
        terminal_penalty = 0.0
        if captured_delta > 0:
            terminal_bonus = float(self.config.terminal_capture_bonus) * float(captured_delta) / max(1, int(num_e))
            for k in rewards:
                rewards[k] += terminal_bonus
                if details is not None:
                    details[k]["terminal_bonus"] = terminal_bonus

        if asset_breached:
            terminal_penalty = float(self.config.terminal_asset_loss_penalty)
            for k in rewards:
                rewards[k] -= terminal_penalty
                if details is not None:
                    details[k]["terminal_penalty"] = terminal_penalty
        if details is not None:
            details["terminal_bonus"] = terminal_bonus
            details["terminal_penalty"] = terminal_penalty

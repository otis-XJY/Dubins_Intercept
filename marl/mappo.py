"""MAPPO 训练辅助：广义优势估计 GAE(λ) 与按时间片小批次的 PPO clip 更新。

与 ``marl/train_online0325`` 中按回合收集的 rollout 配合；优势在训练端可做标准化。
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    *,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    rewards, values: [T, P]; dones: [T] (True 表示该步后 episode 结束);
    last_value: [P]，对最后 next state 的 bootstrap（若已终止可在外部置零）。
    """
    t_max, _p = rewards.shape
    adv = np.zeros((t_max, _p), dtype=np.float32)
    gae = np.zeros(_p, dtype=np.float32)

    for t in range(t_max - 1, -1, -1):
        next_v = last_value if t == t_max - 1 else values[t + 1]
        non_term = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_v * non_term - values[t]
        gae = delta + gamma * lam * non_term * gae
        adv[t] = gae

    ret = adv + values
    return adv, ret


def ppo_minibatch_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_list: List[Dict[str, np.ndarray]],
    actions: np.ndarray,
    logp_old: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    build_obs_fn: Callable[[Dict[str, np.ndarray], torch.device], Dict[str, torch.Tensor]],
    *,
    clip_range: float,
    ppo_epochs: int,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    minibatch_size: int,
) -> Tuple[float, float, float]:
    """对一条轨迹做多轮 epoch；每步用 ``actor_forward``/``critic_forward``（支持 DDP 包装）。"""
    unwrap = model.module if hasattr(model, "module") else model
    t_max, _p = actions.shape
    order = np.arange(t_max)

    adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.long, device=device)

    tot_pi = 0.0
    tot_v = 0.0
    tot_ent = 0.0
    n_mb = 0

    for _ in range(ppo_epochs):
        np.random.shuffle(order)
        for start in range(0, t_max, minibatch_size):
            batch = order[start : start + minibatch_size]
            if len(batch) == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            policy_loss_acc = 0.0
            value_loss_acc = 0.0
            ent_acc = 0.0
            n_in = 0

            for t in batch:
                obs_t = build_obs_fn(obs_list[t], device)
                out = unwrap.actor_forward(obs_t)
                probs = out["action_probs"]
                dist = torch.distributions.Categorical(probs=probs)
                new_logp = dist.log_prob(act_t[t])
                entropy = dist.entropy().mean()
                vals = unwrap.critic_forward(obs_t)

                adv = adv_t[t]
                ret = ret_t[t]
                ratio = torch.exp(new_logp - logp_old_t[t])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(vals, ret)
                loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy) / len(batch)

                loss.backward()
                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                ent_acc += float(entropy.item())
                n_in += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            tot_pi += policy_loss_acc / max(n_in, 1)
            tot_v += value_loss_acc / max(n_in, 1)
            tot_ent += ent_acc / max(n_in, 1)
            n_mb += 1

    if n_mb == 0:
        return 0.0, 0.0, 0.0
    return tot_pi / n_mb, tot_v / n_mb, tot_ent / n_mb

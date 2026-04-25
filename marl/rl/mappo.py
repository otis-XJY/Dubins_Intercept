"""MAPPO 训练辅助：广义优势估计 GAE(λ) 与按时间片小批次的 PPO clip 更新。

与 ``marl.runners.online_train`` 中按回合收集的 rollout 配合；优势在训练端可做标准化。
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
    verbose: bool = False,
) -> Tuple[float, float, float, float, float, float, float]:
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
    tot_approx_kl = 0.0
    tot_clipfrac = 0.0
    tot_grad_norm = 0.0
    # collect prediction/target for explained variance (across minibatches/epochs)
    ev_y = []
    ev_yhat = []
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
            approx_kl_acc = 0.0
            clipfrac_acc = 0.0
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
                logratio = new_logp - logp_old_t[t]
                ratio = torch.exp(logratio)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(vals, ret)
                loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy) / len(batch)

                loss.backward()
                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                ent_acc += float(entropy.item())
                # PPO diagnostics (CleanRL style)
                approx_kl = ((torch.exp(logratio) - 1.0) - logratio).mean()
                clipfrac = (torch.abs(ratio - 1.0) > clip_range).float().mean()
                approx_kl_acc += float(approx_kl.item())
                clipfrac_acc += float(clipfrac.item())
                ev_y.append(ret.detach().flatten())
                ev_yhat.append(vals.detach().flatten())
                n_in += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            tot_pi += policy_loss_acc / max(n_in, 1)
            tot_v += value_loss_acc / max(n_in, 1)
            tot_ent += ent_acc / max(n_in, 1)
            tot_approx_kl += approx_kl_acc / max(n_in, 1)
            tot_clipfrac += clipfrac_acc / max(n_in, 1)
            tot_grad_norm += float(grad_norm) if isinstance(grad_norm, (float, int)) else float(grad_norm.item())
            n_mb += 1

    if n_mb == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pi_avg = tot_pi / n_mb
    v_avg = tot_v / n_mb
    ent_avg = tot_ent / n_mb
    approx_kl_avg = tot_approx_kl / n_mb
    clipfrac_avg = tot_clipfrac / n_mb
    grad_norm_avg = tot_grad_norm / n_mb

    explained_variance = 0.0
    if len(ev_y) > 0:
        y = torch.cat(ev_y, dim=0)
        yhat = torch.cat(ev_yhat, dim=0)
        var_y = torch.var(y)
        if float(var_y.item()) > 1e-12:
            explained_variance = float((1.0 - torch.var(y - yhat) / var_y).item())
    if verbose:
        print(
            f"[MAPPO] rollout_T={t_max} PPO_epochs={ppo_epochs} minibatches={n_mb} "
            f"policy_loss={pi_avg:.5f} value_loss={v_avg:.5f} entropy={ent_avg:.5f} "
            f"approx_kl={approx_kl_avg:.5f} clipfrac={clipfrac_avg:.3f} ev={explained_variance:.3f} grad={grad_norm_avg:.3f}"
        )
    return pi_avg, v_avg, ent_avg, approx_kl_avg, clipfrac_avg, explained_variance, grad_norm_avg


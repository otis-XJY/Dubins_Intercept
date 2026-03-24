import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from marl import TODCMARLEnv, build_actor_critic_schemes

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


@dataclass
class TrainConfig:
    episodes: int = 100
    max_episode_steps: int = 200
    gamma: float = 0.99
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    hidden_dim: int = 128
    num_heads: int = 4
    seed: int = 42
    device: str = "cpu"
    schemes: Optional[List[str]] = None
    time_res: float = 1.0
    k_max: int = 32
    step_mode: str = "decision"
    max_inner_ticks: int = 50
    allow_dummy_if_missing: bool = True
    enable_dwa_replan: bool = True
    wandb_project: str = "dubins-marl-online"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    log_interval: int = 1
    eval_interval: int = 10
    eval_episodes: int = 1
    save_dir: str = "output/checkpoints"
    save_best: bool = True
    save_replay: bool = True
    replay_dir: str = "output/eval_traces"


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _load_config_file(path: str) -> Dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml installed") from e
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file content must be a dict/object")
    return data


def _to_tensor(x: np.ndarray, device: torch.device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def _angle_from_cos_sin(cos_v: np.ndarray, sin_v: np.ndarray) -> np.ndarray:
    return np.arctan2(sin_v, cos_v)


def _build_model_obs(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    pursuers = np.asarray(obs["pursuers"], dtype=np.float32)  # [P, 6]
    evaders = np.asarray(obs["evaders"], dtype=np.float32)  # [E, 8]
    candidates = np.asarray(obs["candidates"], dtype=np.float32)  # [P, K, 6]
    candidate_mask = np.asarray(obs["candidate_mask"], dtype=np.float32)  # [P, K]

    num_p = pursuers.shape[0]
    num_e = evaders.shape[0]
    k_max = candidates.shape[1]

    # self_uav: [x, y, theta]
    theta_p = _angle_from_cos_sin(pursuers[:, 4], pursuers[:, 5])
    self_uav = np.stack([pursuers[:, 0], pursuers[:, 1], theta_p], axis=-1)[:, None, :]  # [P,1,3]

    # ally_uavs: each sample excludes itself, shape [P, P-1, 3]
    ally_uavs = np.zeros((num_p, max(1, num_p - 1), 3), dtype=np.float32)
    ally_mask = np.zeros((num_p, max(1, num_p - 1)), dtype=bool)
    for pid in range(num_p):
        others = [i for i in range(num_p) if i != pid]
        if len(others) == 0:
            continue
        vals = np.stack([pursuers[others, 0], pursuers[others, 1], theta_p[others]], axis=-1)
        ally_uavs[pid, : len(others)] = vals
        ally_mask[pid, : len(others)] = True

    # self_pts: pad candidate 6D -> 8D (append zeros)
    self_pts = np.zeros((num_p, k_max, 8), dtype=np.float32)
    self_pts[:, :, :6] = candidates
    self_pts_mask = candidate_mask > 0

    # ally_pts: concat others' candidate points for each sample, shape [P, (P-1)*K, 8]
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

    # enemies: [x, y, theta]
    theta_e = _angle_from_cos_sin(evaders[:, 4], evaders[:, 5])
    enemies = np.stack([evaders[:, 0], evaders[:, 1], theta_e], axis=-1)[None, :, :].repeat(num_p, axis=0)
    enemy_mask = np.ones((num_p, num_e), dtype=bool)

    # targets from evader inferred target fields [tx, ty]
    targets = evaders[:, 6:8][None, :, :].repeat(num_p, axis=0)
    target_mask = np.ones((num_p, num_e), dtype=bool)

    return {
        "self_uav": _to_tensor(self_uav, device),
        "ally_uavs": _to_tensor(ally_uavs, device),
        "self_pts": _to_tensor(self_pts, device),
        "ally_pts": _to_tensor(ally_pts, device),
        "enemies": _to_tensor(enemies, device),
        "targets": _to_tensor(targets, device),
        "ally_mask": _to_tensor(ally_mask, device, dtype=torch.bool),
        "self_pts_mask": _to_tensor(self_pts_mask, device, dtype=torch.bool),
        "ally_pts_mask": _to_tensor(ally_pts_mask, device, dtype=torch.bool),
        "enemy_mask": _to_tensor(enemy_mask, device, dtype=torch.bool),
        "target_mask": _to_tensor(target_mask, device, dtype=torch.bool),
    }


def _init_wandb(cfg: TrainConfig, schemes: Sequence[str]):
    if cfg.wandb_mode == "disabled":
        return None

    if wandb is None:
        print("[WARN] wandb is not installed, skip wandb logging.")
        return None

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config={
            **cfg.__dict__,
            "schemes": list(schemes),
        },
    )
    return run


def _run_eval_episode(
    model: torch.nn.Module,
    env: TODCMARLEnv,
    device: torch.device,
):
    model.eval()
    obs_np, _ = env.reset()
    done = False
    trunc = False
    ep_return = 0.0
    ep_steps = 0
    trace_actions = []
    trace_rewards = []
    trace_captured = []
    trace_inner_ticks = []

    with torch.no_grad():
        while (not done) and (not trunc):
            obs_t = _build_model_obs(obs_np, device)
            out = model(obs_t)
            actions = torch.argmax(out["action_probs"], dim=-1)

            next_obs_np, rewards, terms, truncs, infos = env.step(actions.detach().cpu().numpy())
            reward_vec = np.array([rewards[f"p_{i}"] for i in range(env.num_P)], dtype=np.float32)

            ep_return += float(np.mean(reward_vec))
            ep_steps += 1
            done = bool(terms["__all__"])
            trunc = bool(truncs["__all__"])

            trace_actions.append(actions.detach().cpu().numpy())
            trace_rewards.append(reward_vec)
            trace_captured.append(np.array([infos[f"p_{i}"]["captured_total"] for i in range(env.num_P)], dtype=np.int64))
            trace_inner_ticks.append(np.array([infos[f"p_{i}"]["inner_ticks"] for i in range(env.num_P)], dtype=np.int64))

            obs_np = next_obs_np

    trace = {
        "actions": np.stack(trace_actions, axis=0) if len(trace_actions) > 0 else np.empty((0,), dtype=np.int64),
        "rewards": np.stack(trace_rewards, axis=0) if len(trace_rewards) > 0 else np.empty((0,), dtype=np.float32),
        "captured_total": np.stack(trace_captured, axis=0) if len(trace_captured) > 0 else np.empty((0,), dtype=np.int64),
        "inner_ticks": np.stack(trace_inner_ticks, axis=0) if len(trace_inner_ticks) > 0 else np.empty((0,), dtype=np.int64),
    }
    return ep_return, ep_steps, trace


def train_online(cfg: TrainConfig):
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    models = build_actor_critic_schemes(
        cfg.schemes,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        device=device,
    )

    run = _init_wandb(cfg, list(models.keys()))

    envs: Dict[str, TODCMARLEnv] = {}
    opts: Dict[str, torch.optim.Optimizer] = {}

    for scheme_name, model in models.items():
        envs[scheme_name] = TODCMARLEnv(
            {
                "allow_dummy_if_missing": cfg.allow_dummy_if_missing,
                "render_mode": "none",
                "max_episode_steps": cfg.max_episode_steps,
                "k_max": cfg.k_max,
                "time_res": cfg.time_res,
                "step_mode": cfg.step_mode,
                "max_inner_ticks": cfg.max_inner_ticks,
                "enable_dwa_replan": cfg.enable_dwa_replan,
            }
        )
        opts[scheme_name] = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {name: [] for name in models.keys()}
    best_eval_return = {name: -np.inf for name in models.keys()}

    os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.save_replay:
        os.makedirs(cfg.replay_dir, exist_ok=True)

    global_step = 0
    for ep in range(1, cfg.episodes + 1):
        for scheme_name, model in models.items():
            model.train()
            env = envs[scheme_name]
            opt = opts[scheme_name]

            obs_np, _ = env.reset(seed=cfg.seed + ep)
            done = False
            trunc = False
            ep_return = 0.0
            ep_policy_loss = 0.0
            ep_value_loss = 0.0
            ep_entropy = 0.0
            ep_steps = 0

            while (not done) and (not trunc):
                obs_t = _build_model_obs(obs_np, device)
                out = model(obs_t)
                logits = out["action_logits"]
                probs = out["action_probs"]
                values = out["value"]

                dist = torch.distributions.Categorical(probs=probs)
                actions = dist.sample()  # [P]
                logp = dist.log_prob(actions)  # [P]
                entropy = dist.entropy().mean()

                next_obs_np, rewards, terms, truncs, infos = env.step(actions.detach().cpu().numpy())

                reward_vec = np.array([rewards[f"p_{i}"] for i in range(env.num_P)], dtype=np.float32)
                reward_t = _to_tensor(reward_vec, device)

                done = bool(terms["__all__"])
                trunc = bool(truncs["__all__"])
                not_done = 0.0 if (done or trunc) else 1.0

                with torch.no_grad():
                    next_obs_t = _build_model_obs(next_obs_np, device)
                    next_values = model(next_obs_t)["value"]
                    target = reward_t + cfg.gamma * not_done * next_values

                advantage = target - values
                policy_loss = -(logp * advantage.detach()).mean()
                value_loss = F.mse_loss(values, target)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                ep_return += float(reward_t.mean().item())
                ep_policy_loss += float(policy_loss.item())
                ep_value_loss += float(value_loss.item())
                ep_entropy += float(entropy.item())
                ep_steps += 1
                global_step += 1

                if run is not None:
                    mean_inner_ticks = float(np.mean([infos[f"p_{i}"]["inner_ticks"] for i in range(env.num_P)]))
                    replanned = float(np.mean([float(infos[f"p_{i}"]["replanned"]) for i in range(env.num_P)]))
                    run.log(
                        {
                            f"{scheme_name}/step_reward": float(reward_t.mean().item()),
                            f"{scheme_name}/policy_loss": float(policy_loss.item()),
                            f"{scheme_name}/value_loss": float(value_loss.item()),
                            f"{scheme_name}/entropy": float(entropy.item()),
                            f"{scheme_name}/inner_ticks": mean_inner_ticks,
                            f"{scheme_name}/replanned_ratio": replanned,
                            f"{scheme_name}/decision_step": float(
                                np.mean([infos[f"p_{i}"]["decision_step"] for i in range(env.num_P)])
                            ),
                            "global_step": global_step,
                        }
                    )

                obs_np = next_obs_np

            ep_stats = {
                "episode": ep,
                "return": ep_return,
                "steps": ep_steps,
                "policy_loss": ep_policy_loss / max(1, ep_steps),
                "value_loss": ep_value_loss / max(1, ep_steps),
                "entropy": ep_entropy / max(1, ep_steps),
            }
            history[scheme_name].append(ep_stats)

            if run is not None:
                run.log(
                    {
                        f"{scheme_name}/episode_return": ep_stats["return"],
                        f"{scheme_name}/episode_steps": ep_stats["steps"],
                        f"{scheme_name}/episode_policy_loss": ep_stats["policy_loss"],
                        f"{scheme_name}/episode_value_loss": ep_stats["value_loss"],
                        f"{scheme_name}/episode_entropy": ep_stats["entropy"],
                        "episode": ep,
                    }
                )

            if cfg.eval_interval > 0 and (ep % cfg.eval_interval) == 0:
                eval_env = TODCMARLEnv(
                    {
                        "allow_dummy_if_missing": cfg.allow_dummy_if_missing,
                        "render_mode": "none",
                        "max_episode_steps": cfg.max_episode_steps,
                        "k_max": cfg.k_max,
                        "time_res": cfg.time_res,
                        "step_mode": cfg.step_mode,
                        "max_inner_ticks": cfg.max_inner_ticks,
                        "enable_dwa_replan": cfg.enable_dwa_replan,
                    }
                )

                eval_returns = []
                eval_steps = []
                last_trace = None
                for _ in range(max(1, cfg.eval_episodes)):
                    ret, stp, trace = _run_eval_episode(model, eval_env, device)
                    eval_returns.append(ret)
                    eval_steps.append(stp)
                    last_trace = trace

                mean_eval_return = float(np.mean(eval_returns))
                mean_eval_steps = float(np.mean(eval_steps))

                if cfg.save_best and mean_eval_return > best_eval_return[scheme_name]:
                    best_eval_return[scheme_name] = mean_eval_return
                    best_path = os.path.join(cfg.save_dir, f"{scheme_name.replace(' ', '_')}_best.pt")
                    torch.save(model.state_dict(), best_path)
                    print(f"[Best Saved] {best_path} eval_return={mean_eval_return:.3f}")

                if cfg.save_replay and last_trace is not None:
                    scheme_dir = os.path.join(cfg.replay_dir, scheme_name.replace(" ", "_"))
                    os.makedirs(scheme_dir, exist_ok=True)
                    trace_path = os.path.join(scheme_dir, f"episode_{ep:05d}.npz")
                    np.savez_compressed(trace_path, **last_trace)

                if run is not None:
                    run.log(
                        {
                            f"{scheme_name}/eval_return": mean_eval_return,
                            f"{scheme_name}/eval_steps": mean_eval_steps,
                            f"{scheme_name}/best_eval_return": best_eval_return[scheme_name],
                            "episode": ep,
                        }
                    )

            if (ep % cfg.log_interval) == 0:
                print(
                    f"[Episode {ep:04d}] [{scheme_name}] return={ep_stats['return']:.3f} "
                    f"steps={ep_stats['steps']} policy={ep_stats['policy_loss']:.4f} "
                    f"value={ep_stats['value_loss']:.4f} entropy={ep_stats['entropy']:.4f}"
                )

    for scheme_name, model in models.items():
        ckpt_path = os.path.join(cfg.save_dir, f"{scheme_name.replace(' ', '_')}_last.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Saved] {ckpt_path}")

    if run is not None:
        run.finish()

    return history


def _build_parser(defaults: Optional[Dict] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online MARL trainer with optional Weights & Biases logging")
    if defaults:
        parser.set_defaults(**defaults)

    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--schemes", type=str, nargs="*", default=None)
    parser.add_argument("--time-res", type=float, default=1.0)
    parser.add_argument("--k-max", type=int, default=32)
    parser.add_argument("--step-mode", type=str, default="decision", choices=["decision", "time"])
    parser.add_argument("--max-inner-ticks", type=int, default=50)
    parser.add_argument("--allow-dummy-if-missing", type=_str2bool, default=True)
    parser.add_argument("--enable-dwa-replan", type=_str2bool, default=True)
    parser.add_argument("--wandb-project", type=str, default="dubins-marl-online")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="output/checkpoints")
    parser.add_argument("--save-best", type=_str2bool, default=True)
    parser.add_argument("--save-replay", type=_str2bool, default=True)
    parser.add_argument("--replay-dir", type=str, default="output/eval_traces")

    return parser


def _parse_args() -> TrainConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    cfg_defaults = dict(TrainConfig().__dict__)
    if pre_args.config:
        file_cfg = _load_config_file(pre_args.config)
        for k, v in file_cfg.items():
            if k in cfg_defaults:
                cfg_defaults[k] = v

    parser = _build_parser(defaults=cfg_defaults)

    args = parser.parse_args()

    return TrainConfig(
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        gamma=args.gamma,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        seed=args.seed,
        device=args.device,
        schemes=args.schemes,
        time_res=args.time_res,
        k_max=args.k_max,
        step_mode=args.step_mode,
        max_inner_ticks=args.max_inner_ticks,
        allow_dummy_if_missing=bool(args.allow_dummy_if_missing),
        enable_dwa_replan=bool(args.enable_dwa_replan),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        save_best=bool(args.save_best),
        save_replay=bool(args.save_replay),
        replay_dir=args.replay_dir,
    )


def main():
    cfg = _parse_args()
    train_online(cfg)


if __name__ == "__main__":
    main()

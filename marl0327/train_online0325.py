import argparse
import sys
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import imageio
import tempfile
import io

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
    distributed: bool = False
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None
    schemes: Optional[List[str]] = None
    time_res: float = 1.0
    step_mode: str = "decision"
    allow_dummy_if_missing: bool = True
    enable_dwa_replan: bool = True
    reward: Optional[Dict] = None
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
    # Visualization settings
    step_frame_interval: int = 50
    eval_video_fps: int = 10


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    is_main: bool


def _setup_distributed(cfg: TrainConfig) -> DistributedContext:
    if not cfg.distributed:
        return DistributedContext(False, 0, 1, 0, True)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistributedContext(False, rank, world_size, local_rank, True)

    backend = str(cfg.ddp_backend).lower().strip()
    if backend == "nccl" and (not torch.cuda.is_available()):
        raise RuntimeError("DDP backend 'nccl' requires CUDA. Use --ddp-backend gloo on CPU.")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(True, rank, world_size, local_rank, rank == 0)


def _cleanup_distributed(ctx: DistributedContext):
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _parse_gpu_ids(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 0:
        return None
    return [int(p) for p in parts]


def _state_dict_for_save(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        return model.module.state_dict()
    return model.state_dict()


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


def _build_model_obs(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    required_keys = {
        "self_uav",
        "ally_uavs",
        "self_pts",
        "ally_pts",
        "enemies",
        "targets",
        "ally_mask",
        "self_pts_mask",
        "ally_pts_mask",
        "enemy_mask",
        "target_mask",
    }
    missing = [k for k in required_keys if k not in obs]
    if missing:
        raise KeyError(f"Environment observation missing model keys: {missing}")

    return {
        "self_uav": _to_tensor(np.asarray(obs["self_uav"], dtype=np.float32), device),
        "ally_uavs": _to_tensor(np.asarray(obs["ally_uavs"], dtype=np.float32), device),
        "self_pts": _to_tensor(np.asarray(obs["self_pts"], dtype=np.float32), device),
        "ally_pts": _to_tensor(np.asarray(obs["ally_pts"], dtype=np.float32), device),
        "enemies": _to_tensor(np.asarray(obs["enemies"], dtype=np.float32), device),
        "targets": _to_tensor(np.asarray(obs["targets"], dtype=np.float32), device),
        "ally_mask": _to_tensor(np.asarray(obs["ally_mask"], dtype=bool), device, dtype=torch.bool),
        "self_pts_mask": _to_tensor(np.asarray(obs["self_pts_mask"], dtype=bool), device, dtype=torch.bool),
        "ally_pts_mask": _to_tensor(np.asarray(obs["ally_pts_mask"], dtype=bool), device, dtype=torch.bool),
        "enemy_mask": _to_tensor(np.asarray(obs["enemy_mask"], dtype=bool), device, dtype=torch.bool),
        "target_mask": _to_tensor(np.asarray(obs["target_mask"], dtype=bool), device, dtype=torch.bool),
    }


def _init_wandb(cfg: TrainConfig, schemes: Sequence[str], *, enabled: bool = True):
    if not enabled:
        return None

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
        tags=["MARL", "TODC", "reward-inspect"],
        notes="Online MARL training run; watch reward components r_qual/r_global/r_safe/r_time",
    )

    # Define commonly used metrics so W&B UI can align step/episode axes and present them nicely.
    # Step-level metrics should use `global_step` as the x-axis; episode-level use `episode`.
    step_metrics = [
        "step_reward",
        "r_qual",
        "r_global",
        "r_safe",
        "r_time",
        "terminal_bonus",
        "terminal_penalty",
        "policy_loss",
        "value_loss",
        "entropy",
        "delta_t_all",
        "global_t_all",
        "path_exec_t",
        "replanned_ratio",
        "decision_step",
    ]
    episode_metrics = [
        "episode_return",
        "episode_steps",
        "episode_policy_loss",
        "episode_value_loss",
        "episode_entropy",
        "episode_asset_breach_count",
        "episode_final_global_t_all",
    ]

    # Define metrics for each scheme (namespacing by scheme_name when logging)
    for scheme in schemes:
        prefix = f"{scheme}/"
        for m in step_metrics:
            name = prefix + m
            try:
                wandb.define_metric(name, step_metric="global_step")
            except Exception:
                try:
                    run.define_metric(name, step_metric="global_step")
                except Exception:
                    pass
        for m in episode_metrics:
            name = prefix + m
            try:
                wandb.define_metric(name, step_metric="episode")
            except Exception:
                try:
                    run.define_metric(name, step_metric="episode")
                except Exception:
                    pass

    # Also define top-level eval and global metrics
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("episode")
        for m in ["eval_return", "eval_steps", "best_eval_return"]:
            wandb.define_metric(m, step_metric="episode")
    except Exception:
        pass

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
    trace_delta_t_all = []
    trace_frames = []

    # 与训练相同：每步 env.step 内部为 main0319 的 update -> step_geometry -> check（见 MARL_env）。
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
            trace_delta_t_all.append(np.array([infos[f"p_{i}"]["delta_t_all"] for i in range(env.num_P)], dtype=np.float64))

            obs_np = next_obs_np

            # collect a render frame if available
            try:
                frame = env.render()
            except Exception:
                frame = None
            if frame is not None:
                trace_frames.append(frame)

    trace = {
        "actions": np.stack(trace_actions, axis=0) if len(trace_actions) > 0 else np.empty((0,), dtype=np.int64),
        "rewards": np.stack(trace_rewards, axis=0) if len(trace_rewards) > 0 else np.empty((0,), dtype=np.float32),
        "captured_total": np.stack(trace_captured, axis=0) if len(trace_captured) > 0 else np.empty((0,), dtype=np.int64),
        "delta_t_all": np.stack(trace_delta_t_all, axis=0) if len(trace_delta_t_all) > 0 else np.empty((0,), dtype=np.float64),
    }
    return ep_return, ep_steps, trace, trace_frames


def train_online(cfg: TrainConfig):
    dist_ctx = _setup_distributed(cfg)
    _set_seed(cfg.seed + dist_ctx.rank)

    requested_device = str(cfg.device).lower()
    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    if requested_device.startswith("cuda") and (not torch.cuda.is_available()) and dist_ctx.is_main:
        print("[WARN] CUDA requested but unavailable, fallback to CPU.")

    gpu_ids = list(cfg.gpu_ids) if cfg.gpu_ids else None
    use_data_parallel = False

    if dist_ctx.enabled:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{dist_ctx.local_rank}")
        else:
            device = torch.device("cpu")
        if cfg.multi_gpu and dist_ctx.is_main:
            print("[WARN] --multi-gpu is ignored when --distributed=true (DDP takes over).")
    else:
        if use_cuda and cfg.multi_gpu:
            n_gpu = torch.cuda.device_count()
            if gpu_ids is None:
                gpu_ids = list(range(n_gpu))
            else:
                gpu_ids = [i for i in gpu_ids if 0 <= i < n_gpu]
            use_data_parallel = len(gpu_ids) > 1

        if use_cuda:
            if use_data_parallel:
                device = torch.device(f"cuda:{gpu_ids[0]}")
            elif gpu_ids and len(gpu_ids) == 1:
                device = torch.device(f"cuda:{gpu_ids[0]}")
            else:
                device = torch.device(requested_device)
        else:
            device = torch.device("cpu")

    models = build_actor_critic_schemes(
        cfg.schemes,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        device=device,
    )

    if dist_ctx.enabled:
        wrapped = {}
        for name, model in models.items():
            wrapped[name] = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=bool(cfg.ddp_find_unused_parameters),
            )
        models = wrapped
        if dist_ctx.is_main:
            print(
                f"[INFO] DDP enabled: world_size={dist_ctx.world_size}, backend={cfg.ddp_backend}, local_rank={dist_ctx.local_rank}"
            )
    elif use_data_parallel and gpu_ids is not None:
        wrapped = {}
        for name, model in models.items():
            wrapped[name] = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        models = wrapped
        print(f"[INFO] DataParallel enabled on GPUs: {gpu_ids}")

    run = _init_wandb(cfg, list(models.keys()), enabled=dist_ctx.is_main)

    envs: Dict[str, TODCMARLEnv] = {}
    opts: Dict[str, torch.optim.Optimizer] = {}

    for scheme_name, model in models.items():
        envs[scheme_name] = TODCMARLEnv(
            {
                "allow_dummy_if_missing": cfg.allow_dummy_if_missing,
                # enable in-env rendering during training only if user requested step-frame capture
                "render_mode": "rgb_array" if cfg.step_frame_interval and cfg.step_frame_interval > 0 else "none",
                "max_episode_steps": cfg.max_episode_steps,
                "time_res": cfg.time_res,
                "step_mode": cfg.step_mode,
                "enable_dwa_replan": cfg.enable_dwa_replan,
                "reward": cfg.reward,
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

            obs_np, _ = env.reset(seed=cfg.seed + ep + dist_ctx.rank * 100000)
            done = False
            trunc = False
            ep_return = 0.0
            ep_policy_loss = 0.0
            ep_value_loss = 0.0
            ep_entropy = 0.0
            ep_steps = 0

            # 外层：一次 env.step(action) = 一次 RL 步；内层按 main0319：update -> step_geometry -> check。
            # 全局仿真时间 t_all 在 infos["global_t_all"]；路径段执行时间 t 在 infos["path_exec_t"]（每 replan 后 t 归零）。
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

                next_obs_np, env_rewards, terms, truncs, infos = env.step(actions.detach().cpu().numpy())

                # Use environment-returned rewards for training target
                reward_vec = np.array([env_rewards[f"p_{i}"] for i in range(env.num_P)], dtype=np.float32)
                reward_t = _to_tensor(reward_vec, device)

                # Use reward breakdown computed by the environment (avoid recomputing here)
                try:
                    reward_details = infos.get("_reward_details_all")
                except Exception:
                    reward_details = None

                # fallback to minimal per-agent map if env did not provide details
                if reward_details is None:
                    reward_details = {
                        f"p_{i}": {
                            "r_qual": float(env_rewards[f"p_{i}"]),
                            "r_global": 0.0,
                            "r_safe": 0.0,
                            "r_time": 0.0,
                        }
                        for i in range(env.num_P)
                    }
                    reward_details["terminal_bonus"] = 0.0
                    reward_details["terminal_penalty"] = 0.0

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
                    mean_delta_t_all = float(np.mean([infos[f"p_{i}"]["delta_t_all"] for i in range(env.num_P)]))
                    g_t_all = float(infos.get("global_t_all", infos[f"p_{0}"]["t_all"]))
                    path_t = float(infos.get("path_exec_t", infos[f"p_{0}"]["t"]))
                    replanned = float(np.mean([float(infos[f"p_{i}"]["replanned"]) for i in range(env.num_P)]))
                    # step级奖励分项统计
                    r_qual = np.mean([reward_details[f"p_{i}"]["r_qual"] for i in range(env.num_P)])
                    r_global = np.mean([reward_details[f"p_{i}"]["r_global"] for i in range(env.num_P)])
                    r_safe = np.mean([reward_details[f"p_{i}"]["r_safe"] for i in range(env.num_P)])
                    r_time = np.mean([reward_details[f"p_{i}"]["r_time"] for i in range(env.num_P)])
                    terminal_bonus = reward_details.get("terminal_bonus", 0.0)
                    terminal_penalty = reward_details.get("terminal_penalty", 0.0)
                    run.log(
                        {
                            f"{scheme_name}/step_reward": float(reward_t.mean().item()),
                            f"{scheme_name}/r_qual": r_qual,
                            f"{scheme_name}/r_global": r_global,
                            f"{scheme_name}/r_safe": r_safe,
                            f"{scheme_name}/r_time": r_time,
                            f"{scheme_name}/terminal_bonus": terminal_bonus,
                            f"{scheme_name}/terminal_penalty": terminal_penalty,
                            f"{scheme_name}/policy_loss": float(policy_loss.item()),
                            f"{scheme_name}/value_loss": float(value_loss.item()),
                            f"{scheme_name}/entropy": float(entropy.item()),
                            f"{scheme_name}/delta_t_all": mean_delta_t_all,
                            f"{scheme_name}/global_t_all": g_t_all,
                            f"{scheme_name}/path_exec_t": path_t,
                            f"{scheme_name}/replanned_ratio": replanned,
                            f"{scheme_name}/decision_step": float(
                                np.mean([infos[f"p_{i}"]["decision_step"] for i in range(env.num_P)])
                            ),
                            "global_step": global_step,
                        }
                    )

                # Step-level visual: upload a frame every cfg.step_frame_interval steps
                if run is not None and cfg.step_frame_interval and cfg.step_frame_interval > 0 and (
                    global_step % cfg.step_frame_interval == 0
                ):
                    try:
                        frame = env.render()
                    except Exception:
                        frame = None
                    if frame is not None:
                        try:
                            run.log({f"{scheme_name}/step_image": wandb.Image(frame, caption=f"step {global_step}"), "global_step": global_step})
                        except Exception:
                            # best-effort: don't crash training on logging issues
                            pass

                obs_np = next_obs_np

            # 统计本episode资产受损次数
            asset_breach_count = sum([infos[f"p_{i}"]["asset_breached"] for i in range(env.num_P)])
            ep_stats = {
                "episode": ep,
                "return": ep_return,
                "steps": ep_steps,
                "policy_loss": ep_policy_loss / max(1, ep_steps),
                "value_loss": ep_value_loss / max(1, ep_steps),
                "entropy": ep_entropy / max(1, ep_steps),
                "asset_breach_count": asset_breach_count,
                "final_global_t_all": float(infos.get("global_t_all", infos[f"p_{0}"]["t_all"])),
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
                        f"{scheme_name}/episode_asset_breach_count": ep_stats["asset_breach_count"],
                        f"{scheme_name}/episode_final_global_t_all": ep_stats["final_global_t_all"],
                        "episode": ep,
                    }
                )

            if dist_ctx.is_main and cfg.eval_interval > 0 and (ep % cfg.eval_interval) == 0:
                eval_env = TODCMARLEnv(
                    {
                        "allow_dummy_if_missing": cfg.allow_dummy_if_missing,
                        "render_mode": "rgb_array",
                        "max_episode_steps": cfg.max_episode_steps,
                        "time_res": cfg.time_res,
                        "step_mode": cfg.step_mode,
                        "enable_dwa_replan": cfg.enable_dwa_replan,
                        "reward": cfg.reward,
                    }
                )

                eval_returns = []
                eval_steps = []
                last_trace = None
                last_frames = None
                for _ in range(max(1, cfg.eval_episodes)):
                    ret, stp, trace, frames = _run_eval_episode(model, eval_env, device)
                    eval_returns.append(ret)
                    eval_steps.append(stp)
                    last_trace = trace
                    last_frames = frames

                mean_eval_return = float(np.mean(eval_returns))
                mean_eval_steps = float(np.mean(eval_steps))

                if cfg.save_best and mean_eval_return > best_eval_return[scheme_name]:
                    best_eval_return[scheme_name] = mean_eval_return
                    best_path = os.path.join(cfg.save_dir, f"{scheme_name.replace(' ', '_')}_best.pt")
                    torch.save(_state_dict_for_save(model), best_path)
                    print(f"[Best Saved] {best_path} eval_return={mean_eval_return:.3f}")

                if cfg.save_replay and last_trace is not None:
                    scheme_dir = os.path.join(cfg.replay_dir, scheme_name.replace(" ", "_"))
                    os.makedirs(scheme_dir, exist_ok=True)
                    trace_path = os.path.join(scheme_dir, f"episode_{ep:05d}.npz")
                    np.savez_compressed(trace_path, **last_trace)

                # Upload eval video to wandb (best-effort)
                if run is not None and last_frames is not None and len(last_frames) > 0:
                    try:
                        # write frames to a temporary mp4 file
                        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmpf.close()
                        with imageio.get_writer(tmpf.name, fps=cfg.eval_video_fps, codec="libx264") as writer:
                            for fr in last_frames:
                                # ensure uint8
                                fr_u8 = (fr.astype(np.uint8) if fr.dtype != np.uint8 else fr)
                                writer.append_data(fr_u8)

                        try:
                            run.log({f"{scheme_name}/eval_video": wandb.Video(tmpf.name, fps=cfg.eval_video_fps, format="mp4"), "episode": ep})
                        except Exception:
                            pass
                        try:
                            os.remove(tmpf.name)
                        except Exception:
                            pass
                    except Exception:
                        # ignore video generation errors
                        pass

                if run is not None:
                    run.log(
                        {
                            f"{scheme_name}/eval_return": mean_eval_return,
                            f"{scheme_name}/eval_steps": mean_eval_steps,
                            f"{scheme_name}/best_eval_return": best_eval_return[scheme_name],
                            "episode": ep,
                        }
                    )

            if dist_ctx.is_main and (ep % cfg.log_interval) == 0:
                print(
                    f"[Episode {ep:04d}] [{scheme_name}] return={ep_stats['return']:.3f} "
                    f"steps={ep_stats['steps']} policy={ep_stats['policy_loss']:.4f} "
                    f"value={ep_stats['value_loss']:.4f} entropy={ep_stats['entropy']:.4f}"
                )

    if dist_ctx.is_main:
        for scheme_name, model in models.items():
            ckpt_path = os.path.join(cfg.save_dir, f"{scheme_name.replace(' ', '_')}_last.pt")
            torch.save(_state_dict_for_save(model), ckpt_path)
            print(f"[Saved] {ckpt_path}")

    if run is not None:
        run.finish()

    _cleanup_distributed(dist_ctx)

    return history


def _build_parser(defaults: Optional[Dict] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online MARL trainer with optional Weights & Biases logging")
    parser.add_argument("--config", type=str, default="configs/train_online0324.yaml", help="Path to JSON/YAML config")
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
    parser.add_argument("--distributed", type=_str2bool, default=False)
    parser.add_argument("--ddp-backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--ddp-find-unused-parameters", type=_str2bool, default=False)
    parser.add_argument("--multi-gpu", type=_str2bool, default=False)
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1,2")
    parser.add_argument("--schemes", type=str, nargs="*", default=None)
    parser.add_argument("--time-res", type=float, default=1.0)
    parser.add_argument("--step-mode", type=str, default="decision", choices=["decision", "time"])
    parser.add_argument("--allow-dummy-if-missing", type=_str2bool, default=True)
    parser.add_argument("--enable-dwa-replan", type=_str2bool, default=True)
    parser.add_argument("--reward-json", type=str, default=None, help="Inline JSON for reward config")
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
    parser.add_argument("--step-frame-interval", type=int, default=1, help="Upload a training step image every N global steps (0 to disable)")
    parser.add_argument("--eval-video-fps", type=int, default=10, help="FPS for eval video uploaded to wandb")

    # Apply file/config defaults after all arguments are declared so they override add_argument's built-in defaults
    if defaults:
        parser.set_defaults(**defaults)

    return parser


def _parse_args() -> TrainConfig:
    # Debug: print raw argv to help diagnose config passing issues
    # print("[DEBUG] sys.argv:", sys.argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    # print("[DEBUG] pre_args.config:", pre_args.config)

    cfg_defaults = dict(TrainConfig().__dict__)
    if pre_args.config:
        file_cfg = _load_config_file(pre_args.config)
        for k, v in file_cfg.items():
            if k in cfg_defaults:
                cfg_defaults[k] = v
    else:
        file_cfg = None

    # print("[DEBUG] cfg_defaults keys from file:", list(file_cfg.keys()) if file_cfg is not None else None)

    parser = _build_parser(defaults=cfg_defaults)

    # Debug: show the defaults applied to the parser (from YAML)
    # try:
    #     print("[DEBUG] cfg_defaults applied to parser:", {k: cfg_defaults.get(k) for k in sorted(cfg_defaults.keys())})
    # except Exception:
    #     pass

    args = parser.parse_args()

    # Debug: print some parsed args to verify values came from YAML
    # try:
    #     print("[DEBUG] parsed args: device=", args.device, "gpu_ids=", args.gpu_ids, "episodes=", args.episodes)
    #     print("[DEBUG] parsed args reward:", getattr(args, "reward", None))
    # except Exception:
    #     pass

    reward_cfg = getattr(args, "reward", None)
    if args.reward_json:
        reward_cfg = json.loads(args.reward_json)

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
        distributed=bool(args.distributed),
        ddp_backend=args.ddp_backend,
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
        multi_gpu=bool(args.multi_gpu),
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        schemes=args.schemes,
        time_res=args.time_res,
        step_mode=args.step_mode,
        allow_dummy_if_missing=bool(args.allow_dummy_if_missing),
        enable_dwa_replan=bool(args.enable_dwa_replan),
        reward=reward_cfg,
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
        step_frame_interval=args.step_frame_interval,
        eval_video_fps=args.eval_video_fps,
    )


def main():
    cfg = _parse_args()
    train_online(cfg)


if __name__ == "__main__":
    main()

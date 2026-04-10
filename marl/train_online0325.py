"""在线 MAPPO 训练入口：多方案并行、可选 W&B、检查点与评估。

配置通过 YAML/CLI 注入 ``TrainConfig``；环境参数放在 ``env`` 字典（含 ``obs.ally_perception_radius`` 等）。
"""
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

from marl import TODCMARLEnv, build_actor_critic_schemes
from marl.mappo import compute_gae, ppo_minibatch_update


@dataclass
class TrainConfig:
    """训练超参与环境/env 片段合并规则；``env`` 会传入 ``TODCMARLEnv``。"""

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
    enable_dwa_replan: bool = True
    # MAPPO
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    ppo_minibatch_size: int = 32
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
    # 每 N 个训练 episode 将整段 rollout 帧序列编码为 mp4 上传 wandb（0 关闭）
    wandb_train_video_every: int = 0
    train_wandb_video_fps: int = 10
    # 合并进 TODCMARLEnv(config=...) 的额外项，例如 map_root、collision_dist、cap_dist（见 docs/MARL_OVERVIEW.md）
    env: Optional[Dict] = None


def _todc_marl_env_dict(cfg: TrainConfig, *, render_mode: str) -> Dict:
    if cfg.env is not None and not isinstance(cfg.env, dict):
        raise TypeError(f"TrainConfig.env must be dict or None, got {type(cfg.env)}")
    out: Dict = {
        "max_episode_steps": cfg.max_episode_steps,
        "time_res": cfg.time_res,
        "step_mode": cfg.step_mode,
        "enable_dwa_replan": cfg.enable_dwa_replan,
        "reward": cfg.reward,
    }
    if cfg.env:
        out.update(cfg.env)
    out["render_mode"] = render_mode
    return out


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_mp4_from_frames(frames: Sequence[np.ndarray], fps: int) -> str:
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmpf.name
    tmpf.close()
    with imageio.get_writer(tmp_path, fps=fps, codec="libx264") as writer:
        for fr in frames:
            fr_u8 = fr.astype(np.uint8) if fr.dtype != np.uint8 else fr
            writer.append_data(fr_u8)
    return tmp_path


def _wandb_log_video(run, key: str, frames: Sequence[np.ndarray], fps: int, episode: int) -> None:
    import wandb as _wandb

    path = _write_mp4_from_frames(frames, fps)
    try:
        run.log({key: _wandb.Video(path, fps=fps, format="mp4"), "episode": episode})
    finally:
        os.remove(path)


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
        import yaml  # type: ignore

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
    """将环境 numpy 观测转为 GPU 张量；键集与 ``UAVInterceptionNetwork`` 一致。"""
    required_keys = {
        "self_uav",
        "allies_local",
        "enemy_assigned_self",
        "enemy_assigned_per_ally",
        "asset_target_self",
        "asset_target_per_ally",
        "self_pts",
        "ally_pts",
        "ally_mask",
        "enemy_self_mask",
        "ally_enemy_mask",
        "self_pts_mask",
        "ally_pts_mask",
    }
    missing = [k for k in required_keys if k not in obs]
    if missing:
        raise KeyError(f"Environment observation missing model keys: {missing}")

    return {
        "self_uav": _to_tensor(np.asarray(obs["self_uav"], dtype=np.float32), device),
        "allies_local": _to_tensor(np.asarray(obs["allies_local"], dtype=np.float32), device),
        "enemy_assigned_self": _to_tensor(np.asarray(obs["enemy_assigned_self"], dtype=np.float32), device),
        "enemy_assigned_per_ally": _to_tensor(np.asarray(obs["enemy_assigned_per_ally"], dtype=np.float32), device),
        "asset_target_self": _to_tensor(np.asarray(obs["asset_target_self"], dtype=np.float32), device),
        "asset_target_per_ally": _to_tensor(np.asarray(obs["asset_target_per_ally"], dtype=np.float32), device),
        "self_pts": _to_tensor(np.asarray(obs["self_pts"], dtype=np.float32), device),
        "ally_pts": _to_tensor(np.asarray(obs["ally_pts"], dtype=np.float32), device),
        "ally_mask": _to_tensor(np.asarray(obs["ally_mask"], dtype=bool), device, dtype=torch.bool),
        "enemy_self_mask": _to_tensor(np.asarray(obs["enemy_self_mask"], dtype=bool), device, dtype=torch.bool),
        "ally_enemy_mask": _to_tensor(np.asarray(obs["ally_enemy_mask"], dtype=bool), device, dtype=torch.bool),
        "self_pts_mask": _to_tensor(np.asarray(obs["self_pts_mask"], dtype=bool), device, dtype=torch.bool),
        "ally_pts_mask": _to_tensor(np.asarray(obs["ally_pts_mask"], dtype=bool), device, dtype=torch.bool),
    }


def _init_wandb(cfg: TrainConfig, schemes: Sequence[str], *, enabled: bool = True):
    if not enabled:
        return None

    if cfg.wandb_mode == "disabled":
        return None

    import wandb

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
            wandb.define_metric(name, step_metric="global_step")
        for m in episode_metrics:
            name = prefix + m
            wandb.define_metric(name, step_metric="episode")

    # Also define top-level eval and global metrics
    wandb.define_metric("global_step")
    wandb.define_metric("episode")
    for m in ["eval_return", "eval_steps", "best_eval_return"]:
        wandb.define_metric(m, step_metric="episode")

    return run


def _run_eval_episode(
    model: torch.nn.Module,
    env: TODCMARLEnv,
    device: torch.device,
):
    """贪心评估一回合：argmax 动作、累计回报与步数，并可选收集轨迹供日志/可视化。"""
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

            trace_frames.append(env.render())

    trace = {
        "actions": np.stack(trace_actions, axis=0) if len(trace_actions) > 0 else np.empty((0,), dtype=np.int64),
        "rewards": np.stack(trace_rewards, axis=0) if len(trace_rewards) > 0 else np.empty((0,), dtype=np.float32),
        "captured_total": np.stack(trace_captured, axis=0) if len(trace_captured) > 0 else np.empty((0,), dtype=np.int64),
        "delta_t_all": np.stack(trace_delta_t_all, axis=0) if len(trace_delta_t_all) > 0 else np.empty((0,), dtype=np.float64),
    }
    return ep_return, ep_steps, trace, trace_frames


def train_online(cfg: TrainConfig):
    """主训练循环：按 scheme 建 env/模型/优化器，rollout + GAE + PPO，周期性评估与保存检查点。"""
    dist_ctx = _setup_distributed(cfg)
    _set_seed(cfg.seed + dist_ctx.rank)

    requested_device = str(cfg.device).lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device 指定了 CUDA，但当前环境不可用 torch.cuda。")
    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()

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
                invalid = [i for i in gpu_ids if not (0 <= i < n_gpu)]
                if invalid:
                    raise ValueError(f"gpu_ids 超出可用范围 [0, {n_gpu - 1}]: {invalid}")
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

    need_train_rgb = (cfg.step_frame_interval > 0) or (
        cfg.wandb_train_video_every > 0 and cfg.wandb_mode != "disabled"
    )
    for scheme_name, model in models.items():
        envs[scheme_name] = TODCMARLEnv(
            _todc_marl_env_dict(cfg, render_mode="rgb_array" if need_train_rgb else "none")
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

            record_train_video = (
                run is not None
                and cfg.wandb_train_video_every > 0
                and (ep % cfg.wandb_train_video_every == 0)
            )
            prev_render_mode = env.render_mode
            train_video_frames: List[np.ndarray] = []
            if record_train_video:
                env.render_mode = "rgb_array"
                train_video_frames.append(env.render())

            # MAPPO：整段 episode 为一条 rollout，再 GAE + PPO 更新
            unwrap = model.module if hasattr(model, "module") else model
            ro_obs: List[Dict[str, np.ndarray]] = []
            ro_act: List[np.ndarray] = []
            ro_logp: List[np.ndarray] = []
            ro_rew: List[np.ndarray] = []
            ro_val: List[np.ndarray] = []
            ro_done: List[bool] = []

            # 外层：一次 env.step(action) = 一次 RL 步；内层按 main0319：update -> step_geometry -> check。
            while (not done) and (not trunc):
                obs_t = _build_model_obs(obs_np, device)
                with torch.no_grad():
                    ao = unwrap.actor_forward(obs_t)
                    probs = ao["action_probs"]
                    dist_cat = torch.distributions.Categorical(probs=probs)
                    actions = dist_cat.sample()
                    logp = dist_cat.log_prob(actions)
                    vals = unwrap.critic_forward(obs_t)

                next_obs_np, env_rewards, terms, truncs, infos = env.step(actions.detach().cpu().numpy())
                reward_vec = np.array([env_rewards[f"p_{i}"] for i in range(env.num_P)], dtype=np.float32)
                reward_details = infos["_reward_details_all"]

                ro_obs.append(obs_np)
                ro_act.append(actions.detach().cpu().numpy())
                ro_logp.append(logp.detach().cpu().numpy())
                ro_rew.append(reward_vec)
                ro_val.append(vals.detach().cpu().numpy())
                step_done = bool(terms["__all__"] or truncs["__all__"])
                ro_done.append(step_done)

                ep_return += float(np.mean(reward_vec))
                ep_steps += 1
                global_step += 1

                obs_np = next_obs_np
                done = bool(terms["__all__"])
                trunc = bool(truncs["__all__"])

                if run is not None:
                    mean_delta_t_all = float(np.mean([infos[f"p_{i}"]["delta_t_all"] for i in range(env.num_P)]))
                    g_t_all = float(infos["global_t_all"])
                    path_t = float(infos["path_exec_t"])
                    replanned = float(np.mean([float(infos[f"p_{i}"]["replanned"]) for i in range(env.num_P)]))
                    r_qual = np.mean([reward_details[f"p_{i}"]["r_qual"] for i in range(env.num_P)])
                    r_global = np.mean([reward_details[f"p_{i}"]["r_global"] for i in range(env.num_P)])
                    r_safe = np.mean([reward_details[f"p_{i}"]["r_safe"] for i in range(env.num_P)])
                    r_time = np.mean([reward_details[f"p_{i}"]["r_time"] for i in range(env.num_P)])
                    terminal_bonus = reward_details["terminal_bonus"]
                    terminal_penalty = reward_details["terminal_penalty"]
                    run.log(
                        {
                            f"{scheme_name}/step_reward": float(np.mean(reward_vec)),
                            f"{scheme_name}/r_qual": r_qual,
                            f"{scheme_name}/r_global": r_global,
                            f"{scheme_name}/r_safe": r_safe,
                            f"{scheme_name}/r_time": r_time,
                            f"{scheme_name}/terminal_bonus": terminal_bonus,
                            f"{scheme_name}/terminal_penalty": terminal_penalty,
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

                if run is not None and cfg.step_frame_interval and cfg.step_frame_interval > 0 and (
                    global_step % cfg.step_frame_interval == 0
                ):
                    import wandb as _wandb

                    frame = env.render()
                    run.log(
                        {
                            f"{scheme_name}/step_image": _wandb.Image(frame, caption=f"step {global_step}"),
                            "global_step": global_step,
                        }
                    )

                if record_train_video:
                    train_video_frames.append(env.render())

            if len(ro_obs) > 0:
                rewards_np = np.stack(ro_rew, axis=0)
                values_np = np.stack(ro_val, axis=0)
                actions_np = np.stack(ro_act, axis=0)
                logp_np = np.stack(ro_logp, axis=0)
                dones_np = np.array(ro_done, dtype=bool)

                with torch.no_grad():
                    last_obs_t = _build_model_obs(obs_np, device)
                    last_v = unwrap.critic_forward(last_obs_t).cpu().numpy()

                adv_np, ret_np = compute_gae(
                    rewards_np,
                    values_np,
                    dones_np,
                    last_v,
                    gamma=cfg.gamma,
                    lam=cfg.gae_lambda,
                )
                adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

                ep_policy_loss, ep_value_loss, ep_entropy = ppo_minibatch_update(
                    model,
                    opt,
                    ro_obs,
                    actions_np,
                    logp_np,
                    adv_np,
                    ret_np,
                    device,
                    _build_model_obs,
                    clip_range=cfg.ppo_clip,
                    ppo_epochs=cfg.ppo_epochs,
                    value_coef=cfg.value_coef,
                    entropy_coef=cfg.entropy_coef,
                    max_grad_norm=1.0,
                    minibatch_size=max(1, min(cfg.ppo_minibatch_size, len(ro_obs))),
                )

            if record_train_video:
                env.render_mode = prev_render_mode
                _wandb_log_video(
                    run,
                    f"{scheme_name}/train_episode_video",
                    train_video_frames,
                    cfg.train_wandb_video_fps,
                    ep,
                )

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
                "final_global_t_all": float(infos["global_t_all"]),
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
                eval_env = TODCMARLEnv(_todc_marl_env_dict(cfg, render_mode="rgb_array"))

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

                if run is not None and last_frames is not None and len(last_frames) > 0:
                    _wandb_log_video(
                        run,
                        f"{scheme_name}/eval_video",
                        last_frames,
                        cfg.eval_video_fps,
                        ep,
                    )

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
    parser.add_argument("--enable-dwa-replan", type=_str2bool, default=True)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-minibatch-size", type=int, default=32)
    parser.add_argument("--reward-json", type=str, default=None, help="Inline JSON for reward config")
    parser.add_argument(
        "--env-json",
        type=str,
        default=None,
        help="Inline JSON merged into TODCMARLEnv config (overrides YAML env), e.g. '{\"collision_dist\": 30}'",
    )
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
    parser.add_argument(
        "--wandb-train-video-every",
        type=int,
        default=0,
        help="Every N training episodes log a full rollout video to wandb (0 to disable)",
    )
    parser.add_argument("--train-wandb-video-fps", type=int, default=10, help="FPS for training episode videos on wandb")

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

    args = parser.parse_args()

    reward_cfg = args.reward
    if args.reward_json:
        reward_cfg = json.loads(args.reward_json)

    env_cfg = args.env
    if args.env_json:
        env_cfg = json.loads(args.env_json)

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
        enable_dwa_replan=bool(args.enable_dwa_replan),
        ppo_clip=float(args.ppo_clip),
        ppo_epochs=int(args.ppo_epochs),
        gae_lambda=float(args.gae_lambda),
        ppo_minibatch_size=int(args.ppo_minibatch_size),
        reward=reward_cfg,
        env=env_cfg,
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
        wandb_train_video_every=args.wandb_train_video_every,
        train_wandb_video_fps=args.train_wandb_video_fps,
    )


def main():
    cfg = _parse_args()
    train_online(cfg)


if __name__ == "__main__":
    main()

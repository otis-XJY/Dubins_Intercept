# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning (MARL) system for UAV pursuit-evasion interception. Multiple Pursuers intercept multiple Evaders in obstacle environments using a "candidate intercept points + MARL discrete selection" paradigm — traditional Dubins path planning generates candidate interception plans, and a MAPPO policy network selects among them.

## Environment Setup

```bash
conda activate dubins
pip install -r requirements.txt
```

Set `export MPLBACKEND=Agg` on headless servers (no display).

## Common Commands

**Training (single GPU):**
```bash
python -m marl.runners.online_train --config configs/train_online0324.yaml
```

**Multi-GPU (DataParallel):**
```bash
python -m marl.runners.online_train --config configs/train_online0324.yaml --multi-gpu true --gpu-ids 0,1,2,3
```

**Multi-GPU (DDP, recommended for production):**
```bash
torchrun --standalone --nproc_per_node=4 -m marl.runners.online_train --config configs/train_online0324.yaml --distributed true --ddp-backend nccl
```

**Override reward params at runtime:**
```bash
--reward-json '{"dist_progress_scale":0.08,"entropy_scale":0.05,"capture_bonus":120}'
```

**Tests:**
```bash
pytest tests/
```

**Smoke verification:**
```bash
python scripts/smoke_verify_env.py
# or: python -c "from marl.envs.todc_env import smoke_test; smoke_test()"
```

**Run a single test:**
```bash
pytest tests/test_rewards_module.py -v
pytest tests/test_rewards_module.py::test_function_name -v
```

**Map asset generation (prerequisite for training):**
```bash
python -m main.batch_env_init --config configs/env_init.yaml
python -m main.batch_env_build --config configs/env_build.yaml
python -m main.batch_evader_paths --config configs/evader_paths.yaml
```

**Parallel map generation (multi-CPU):**
```bash
# 通过 --num-workers 指定并行 worker 数（0=CPU 核数）
python -m main.batch_env_init --config configs/env_init.yaml --num-workers 4
python -m main.batch_env_build --config configs/env_build.yaml --num-workers 4
python -m main.batch_evader_paths --config configs/evader_paths.yaml --num-workers 4  # 仅 replay 模式

# 也可在 YAML 中配置 num_workers（CLI 参数优先级更高）
```

**日志**：并行模式下所有详细日志写入 `map/_batch_<stage>.log`（init/build/epath），终端只显示进度和汇总。

**Background training:**
```bash
# screen (recommended)
screen -S marl_train
python -m marl.runners.online_train --config configs/train_online0324.yaml
# Ctrl+A D to detach, screen -r marl_train to reattach

# nohup
nohup python -m marl.runners.online_train --config configs/train_online0324.yaml > train.log 2>&1 &
```

## Architecture

```
Traditional Planner (intercept/ + A_dubins)
        │
   Candidate intercept points
        │
   TODCMARLEnv (marl/envs/todc_env.py)  ← Gymnasium env, decision-step driven
        │
   ┌────┼────────────┐
   │    │             │
Observation  Reward   Action Space
(marl/obs/)  (marl/rewards/)  (discrete candidate index)
   │    │             │
   └────┴─────────────┘
        │
   Policy Network (marl/nn/models.py)
   Design A: Concatenative Query Network
   Design B: Gated Query Network
   Design C: Point-Wise Scoring Network
   + Centralized MAPPO Critic (parameter-shared, batch dim = P pursuers)
        │
   Online Trainer (marl/runners/online_train.py)
   MAPPO + GAE(λ) + PPO clip + minibatch updates
   W&B logging, periodic eval, best/last checkpointing
```

## Key Directories

- `marl/envs/` — Gymnasium environment with IsoMap replanning and Hungarian assignment
- `marl/obs/` — Ego-centric observation tensor construction (self_uav, allies, enemies, candidate points, masks)
- `marl/rewards/` — Modular reward function (configurable via YAML, CLI JSON, or runtime API)
- `marl/nn/` — Three policy network designs (A/B/C) with multi-head attention + centralized critic
- `marl/nn/blocks/` — Active modular network components (actor, critic, attention, embeddings, fusion, schemes)
- `marl/rl/` — MAPPO algorithm: GAE computation and PPO minibatch updates
- `marl/runners/` — Online training entry point with multi-GPU, W&B, evaluation, checkpointing
- `intercept/` — Traditional planning pipeline: IsoMap, IsoPair, map processing, evader prediction
- `A_dubins/` — Dubins path planning core (with/without obstacles, A*-style search)
- `configs/` — YAML training and environment configuration files
- `map/` — Precomputed map assets (`.jbl` joblib files, `.png` previews) — tracked by Git LFS
- `trash/` — Deprecated files kept for reference (old models, obs generator, mappo)
- `model_blocks/` — Legacy model blocks; **use `nn/blocks/` instead** (active version)

## Important Conventions

- **Language**: All code comments and documentation are in Chinese.
- **Fail-fast**: Code does not use excessive robustness; raises errors directly on unexpected input (e.g., `ValueError` for illegal actions, `FileNotFoundError` for missing map assets). Do not add defensive try/except around internal logic.
- **Git LFS**: `.jbl` and `.png` files are tracked via Git LFS.
- **Map assets required**: Training and simulation load precomputed joblib resources from `map/`; missing files cause immediate errors.
- **Dynamic action space**: `k_max` changes with replanning events — `action_space`/`observation_space` are rebuilt at runtime.
- **Parameter-shared network**: All pursuers share one network; batch dimension = number of pursuers P.
- **Centralized critic, decentralized actor (CTDE)**: Critic sees all agents' features concatenated; actor only sees its own ego-centric view.
- **Decision-step semantics**: Each `env.step()` is one online decision step (may contain multiple inner simulation ticks until replanning event or termination).

## Modifying Key Components

- **Network structure** → `marl/nn/models.py` (designs A/B/C, aliases: cqn/gqn/pwsn)
- **Reward function** → `marl/rewards/todc_reward.py` (three config methods: YAML, CLI `--reward-json`, runtime `set_reward_params`)
- **Environment logic** → `marl/envs/todc_env.py` (step semantics, replanning, collision detection)
- **Observation space** → `marl/obs/generator.py` (8-dim candidate features: x, y, t_p, t_e, delta_t, cost + 2 reserved)
- **Training algorithm** → `marl/runners/online_train.py` (MAPPO main loop, multi-scheme parallel training)

## W&B Integration

Configure via YAML: `wandb_mode: online|offline|disabled`. Key metrics: `r_qual`, `r_global`, `r_safe`, `r_time`, `terminal_bonus`, `terminal_penalty`, `step_reward`, `episode_return`. Offline mode: set `WANDB_MODE=offline`, sync later with `wandb sync`.

## Training Output Structure

Each training run creates a timestamped directory under `output/runs/`:

```
output/runs/
└── 20260505_131911/          # YYYYMMDD_HHMMSS 格式
    ├── train_config.yaml     # 本次训练的配置备份（便于复现）
    ├── checkpoints/          # 模型检查点
    │   ├── {scheme}_best.pt  # eval 回报创新高时保存
    │   └── {scheme}_last.pt  # 训练结束时保存
    └── eval_traces/          # 评估轨迹（需 save_replay: true）
        └── {scheme}/
            └── *.npz
```

- W&B 运行名自动追加时间戳（如 `online_multi_scheme_20260505_131911`）
- 配置文件自动备份到运行目录，确保每次训练可复现
- 可通过 `--save-dir` 和 `--replay-dir` 覆盖默认路径（但仍会创建时间戳子目录）

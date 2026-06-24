# Dubins_Intercept 训练与调参指南（含远程服务器实测）

> 目标：回答两个核心问题：
> 1) 如何开展训练、注意事项、tmux 使用方式；
> 2) 参数如何调整，以及基于远程测试给出的可执行参数建议。

---

## 1. 训练前准备

### 1.1 资产准备（训练前置）
先保证地图与轨迹资产已生成：

```bash
python -m main.batch_env_init --config configs/env_init.yaml
python -m main.batch_env_build --config configs/env_build.yaml
python -m main.batch_evader_paths --config configs/evader_paths.yaml
```

若 `map/` 下缺失必要 `.jbl` 资产，环境会 fail-fast 报错。

### 1.2 环境准备

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dubins
export MPLBACKEND=Agg
```

无显示环境（远程服务器）必须设置 `MPLBACKEND=Agg`，否则渲染相关流程可能报错。

---

## 2. 标准训练流程

### 2.1 单卡训练（推荐起点）

```bash
python -u -m marl.runners.online_train --config configs/train_online0324.yaml
```

### 2.2 多卡训练（DDP，生产推荐）

```bash
torchrun --standalone --nproc_per_node=4 -m marl.runners.online_train \
  --config configs/train_online0324.yaml \
  --distributed true --ddp-backend nccl
```

### 2.3 运行目录与输出
训练会自动在 `output/runs/<timestamp>/` 生成：
- `train_config.yaml`
- `checkpoints/{scheme}_best.pt`（若开启 eval + save_best）
- `checkpoints/{scheme}_last.pt`
- `eval_traces/`（若开启 save_replay）

---

## 3. tmux 使用建议（强烈推荐）

### 3.1 启动

```bash
tmux new -s marl_train
```

### 3.2 在 tmux 内启动训练

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate dubins
export MPLBACKEND=Agg
python -u -m marl.runners.online_train --config configs/train_online0324.yaml --wandb-mode offline
```

### 3.3 常用操作
- 分离会话：`Ctrl+b` 后按 `d`
- 查看会话：`tmux ls`
- 恢复会话：`tmux attach -t marl_train`

建议使用 `python -u` 或 `export PYTHONUNBUFFERED=1`，避免日志缓冲造成“看起来卡住”。

---

## 4. 远程服务器验证（实践经验）

远程验证以“远程 pull 后执行结果”为准，修复应在本地完成并 push 同步。

### 4.1 非交互密码输入（避免 SSH 等待）
推荐 `expect`：

```bash
expect -c 'set timeout 60; spawn ssh -o StrictHostKeyChecking=no xujunyi@210.75.240.143 "hostname"; expect -re {[Pp]assword:}; send "xujunyi@123\r"; expect eof'
```

### 4.2 建立复用连接（减少重复输密）

```bash
expect -c 'set timeout 30; spawn ssh -o StrictHostKeyChecking=no -o ControlMaster=yes -o ControlPath=/tmp/mux_xjy -o ControlPersist=600 -fN xujunyi@210.75.240.143; expect -re {[Pp]assword:}; send "xujunyi@123\r"; expect eof'
```

后续命令可复用：

```bash
ssh -o ControlPath=/tmp/mux_xjy xujunyi@210.75.240.143 'cd ~/Dubins_Intercept && git pull'
```

用完可关闭：

```bash
ssh -o ControlPath=/tmp/mux_xjy -O exit xujunyi@210.75.240.143
```

---

## 5. 关键参数怎么调

以下参数集中在 `configs/train_online0324.yaml`，也可由 CLI 覆盖。

### 5.1 PPO 与优化稳定性
- `lr`: 学习率，默认 `3e-4`
- `ppo_clip`: PPO 裁剪，默认 `0.2`
- `ppo_epochs`: 每轮更新次数，默认 `4`
- `ppo_minibatch_size`: 默认 `32`
- `gamma`: 默认 `0.99`
- `gae_lambda`: 默认 `0.95`
- `entropy_coef`: 默认 `0.02`
- `value_coef`: 默认 `0.5`
- `kl_early_stop_threshold`: 默认 `0.03`
- `advantage_clip`: 默认 `5.0`
- `lr_decay`: 线性衰减开关

### 5.2 Design D（Two-Stage Residual）专项
- `cf_adv_coef`: 反事实信用分配系数（建议 `0.05~0.2`）
- `temporal_window`: 时序窗口（建议 `4~8`）
- `temporal_heads`: 时序注意力头数（常用 `2`）
- `temporal_layers`: 时序层数（常用 `1`）

### 5.3 奖励参数
- 进度 shaping：`dist_progress_scale`
- 安全：`safe_dist_min`, `safe_penalty_scale`
- 终局：`terminal_capture_bonus`, `terminal_asset_loss_penalty`
- 分配协同：`assign_penalty`, `div_sigma`

---

## 6. 基于远程实测的参数建议

### 6.1 已执行的远程测试（摘要）
- 远程环境：Python `3.10.19`，解释器 `/home/xujunyi/anaconda3/envs/dubins/bin/python`
- 回归测试：
  - `pytest tests/test_design_d_enhancements.py -v --tb=short` 全通过
  - `pytest tests/test_train_online_smoke.py tests/test_design_d_enhancements.py -v --tb=short` 全通过（37 passed）
- 短训观测（Design D，`temporal_window=4`, `cf_adv_coef=0.1`）可正常运行并产出 RL/PPO 指标。

### 6.2 推荐配置档位

#### A. 快速冒烟（验证链路）

```bash
python -u -m marl.runners.online_train --config configs/train_online0324.yaml \
  --episodes 3 --max-episode-steps 40 --wandb-mode disabled --eval-interval 0 \
  --device cuda:0 --save-replay false
```

用途：确认环境、模型、训练循环、保存逻辑是否通。

#### B. 单卡稳定训练（推荐起点）
- `episodes: 5000`
- `max_episode_steps: 300`
- `lr: 3e-4`
- `ppo_clip: 0.2`
- `ppo_epochs: 6`（若 `approx_kl` 持续很小可从 4 提到 6）
- `value_coef: 1.0`（value loss 偏高时建议提升）
- `entropy_coef: 0.01~0.02`
- `cf_adv_coef: 0.1`
- `temporal_window: 4`

#### C. 长训/论文档（多卡）
- `episodes: 30000`
- `distributed: true` + `torchrun`
- `wandb_mode: offline`（训练中降网络依赖）
- `eval_interval: 10`

---

## 7. 按指标调参（最实用）

### 7.1 看 `approx_kl`
- 太低（如长期 `<0.001`）：策略更新偏保守，可提高 `ppo_epochs` 或略增 `lr`
- 太高（接近阈值）：降低 `lr` 或减小 `ppo_clip`

### 7.2 看 `value_loss` 与 `explained_variance`
- `value_loss` 高且 `ev` 低：提高 `value_coef`，必要时降低 `lr`
- `ev` 稳步上升：critic 学习在改善

### 7.3 看 `entropy`
- 过高且不收敛：降 `entropy_coef`（如 0.02 -> 0.01）
- 过早塌缩：升 `entropy_coef`

### 7.4 看协同行为（任务导向）
- 多机追同目标：提高 `assign_penalty`，可适当减小 `div_sigma`
- 碰撞偏多：提高 `safe_penalty_scale`、增大 `safe_dist_min`
- 捕获效率低：提高 `dist_progress_scale` 或 `terminal_capture_bonus`

---

## 8. 常见问题

1) **日志长时间不刷新**：加 `-u` 或 `PYTHONUNBUFFERED=1`。  
2) **SSH 卡住等待密码**：用 `expect`，并配 `ControlMaster` 复用连接。  
3) **远程与本地结果不一致**：以远程 `git pull` 后结果为准。  
4) **无显示服务器报绘图错误**：`export MPLBACKEND=Agg`。  
5) **配置改了不生效**：确认 CLI/YAML 是否被正确透传，优先用一次短训验证日志中的关键参数行为。

---

## 9. 一条可直接执行的远程验证命令（示例）

```bash
ssh -o ControlPath=/tmp/mux_xjy xujunyi@210.75.240.143 \
'cd ~/Dubins_Intercept && git pull && source ~/anaconda3/etc/profile.d/conda.sh && conda activate dubins && export MPLBACKEND=Agg && pytest tests/test_design_d_enhancements.py -v --tb=short'
```

该命令适合做每次改动后的最小远程回归。

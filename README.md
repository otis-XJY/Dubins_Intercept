# Dubins Intercept MARL 设计说明

本仓库目前已经从最初的 Dubins 路径规划，扩展到了可在线训练的多智能体强化学习（MARL）框架。本文档聚焦 MARL 设计，帮助你后续有针对性地修改网络、奖励、环境与训练流程。

## 1. 项目目标与核心思路

项目目标是让多个 Pursuer（追捕者）在复杂障碍环境中在线拦截多个 Evader（逃逸者）。

当前方案采用“候选拦截点 + 策略选择”的范式：

1. 底层几何/图搜索模块生成候选拦截方案（IsoMap + 任务分配）。
2. MARL 策略网络不直接输出连续控制，而是在候选集合中选择行动。
3. 环境支持在线决策步（decision-driven），让训练步与重规划事件对齐。

---

## 2. 代码结构（MARL 相关）

核心目录在 [marl](marl)：

1. [marl/MARL_env.py](marl/MARL_env.py)
: 强化学习环境，包含真实 IsoMap 重规划与 dummy 回退模式。

2. [marl/obs_generator.py](marl/obs_generator.py)
: 观测构建器，直接生成模型输入对齐字段（self_uav / self_pts / enemies / masks）。

3. [marl/rewards.py](marl/rewards.py)
: 独立奖励模块，支持配置化与运行时调参。

4. [marl/models.py](marl/models.py)
: 三种策略网络结构（A/B/C）与统一命名、构建接口。

5. [marl/train_online0325.py](marl/train_online0325.py)
: 在线训练入口，支持单机、DataParallel、多进程 DDP（torchrun）、W&B、配置文件、评估与 best 保存。

6. [configs/train_online.example.yaml](configs/train_online.example.yaml)
: 训练配置样例。

7. [docs/MARL_OVERVIEW.md](docs/MARL_OVERVIEW.md)
: MARL 架构解读、I/O 与模型侧风险说明（与默认训练入口 `marl.train_online0325` 一致）。

---

## 3. 环境设计（TODCMARLEnv）

环境类：`TODCMARLEnv`，定义在 [marl/MARL_env.py](marl/MARL_env.py)。

### 3.1 两种运行模式

## 如何在 W&B 网页实时查看奖励变化

- 前提：在配置文件中启用 W&B（`wandb_mode: online`），并设置 `wandb_project`、`wandb_entity`（可选）和 `wandb_run_name`（可选）。推荐以包方式运行训练：

```bash
python -m marl.train_online0325 --config configs/train_online0324.yaml
```

- 启动后：控制台会输出当前 Run 的链接，或在浏览器打开：
	`https://wandb.ai/<entity>/<project>`（例如 `https://wandb.ai/your-username/dubins-marl-online`）。

- 面板操作速查：
	- 选择某个 Run → 进入 `Charts` / `Metrics` 页面。
	- 在 metrics 列表中勾选我们在训练中上报的指标：`r_qual`、`r_global`、`r_safe`、`r_time`、`terminal_bonus`、`terminal_penalty`、`step_reward`、`episode_return`。
	- 使用 smoothing、scale、并把 x 轴切换为 `global_step` 或 `episode` 以匹配实验节奏。
	- 勾选多个 Runs 后点击 `Compare`，进行跨试验对比（例如不同 reward 超参对比）。

- 进阶：
	- 在 Charts 页面创建并保存自定义 Dashboard（把常用分项放一起，便于长期观察）。
	- 若希望记录模型/梯度，代码中可以调用 `wandb.watch(model)`（需自行添加）。
	- 网络受限时，可用 `WANDB_MODE=offline` 记录本地，之后用 `wandb sync` 同步。

- 常见问题：
	- 看不到日志：确认 `wandb login` 已完成并且 `wandb_mode` 非 `disabled`。
	- Run 名不明确：在 YAML 中填 `wandb_run_name` 或运行时设置 `name`。

这个仓库已在训练脚本中按 step/episode 级别上报奖励分项，按上述步骤即可在 W&B 面板实时查看并对比它们的变化。

1. real_mode
: 从 map 目录加载真实资产（Map、IsoMap、PathE2Val_true 等），进行真实重规划。

2. dummy_mode
: 当资产缺失且 `allow_dummy_if_missing=true` 时启用，使用简化动态与虚拟候选。

这使得你既可在完整场景训练，也可在无地图资产时快速 smoke test。

### 3.2 Step 模式（关键）

环境支持两种步进语义：

1. `step_mode=time`
: 每次 `env.step()` 推进一个固定时间步。

2. `step_mode=decision`
: 每次 `env.step()` 内部推进多个 time tick，直到发生一次重规划事件（或终止/超限），更贴近 online 决策语义。

`decision` 模式下新增统计：

1. `inner_ticks`
: 当前 RL step 内部累计推进了多少时间 tick。

2. `forced_decision`
: 达到 `max_inner_ticks` 仍未触发重规划时，强制返回。

3. `decision_step`
: 累计的决策事件计数。

### 3.3 环境状态与路径更新

环境维护关键状态：

1. `PosP`, `PosE`
: 当前追捕者/逃逸者状态。

2. `PathP`, `PathE`
: 当前计划路径。

3. `PathPtrue`
: 追捕者历史真实轨迹。

4. `IC`, `IC_candidates`, `ICFinal`
: 拦截候选与最终分配结果。

### 3.4 重规划管线

在 real_mode 中，重规划由 `_replan_with_isomap` 执行，主要流程：

1. 预测逃逸者目标（`predictLikelyTargetNew`）。
2. 计算 E2TP 与 P2TP 路径。
3. 动态拼接 IsoMap。
4. 枚举时序组合并筛选可行拦截对。
5. 生成候选任务，按启发式筛选每对 (E,P) 的最佳候选。
6. 用 Hungarian（`linear_sum_assignment`）完成多对多分配。
7. 更新 `PathP/PathE` 与 `pairs_realE2P`。

---

## 4. 观测空间设计（模型直连）

观测由 [marl/obs_generator.py](marl/obs_generator.py) 生成，并直接对齐 [marl/models.py](marl/models.py) 的输入键。

当前统一字段：

1. `self_uav`: `[P, 1, 3]`
2. `ally_uavs`: `[P, max(1, P-1), 3]`
3. `self_pts`: `[P, K_t, 8]`（`K_t` 为当前时刻动态候选数）
4. `ally_pts`: `[P, max(1, (P-1)*K_t), 8]`
5. `enemies`: `[P, E, 3]`
6. `targets`: `[P, E, 2]`
7. `ally_mask`, `self_pts_mask`, `ally_pts_mask`, `enemy_mask`, `target_mask`

### 4.1 self_uav / ally_uavs / enemies

特征语义统一为几何状态 `(x, y, theta)`，并在观测维度上按自机、友机、敌机分组。

### 4.2 self_pts / ally_pts

候选点特征以 `self_pts[..., :6]` 为核心：

1. candidate_x, candidate_y
2. t_p, t_e
3. delta_t = t_e - t_p
4. cost

`self_pts` 的后 2 维是保留扩展位，便于后续加入风险、可见性等几何特征。

### 4.3 masks

所有不定长实体都使用布尔 mask 指示有效位，当前主要依赖：

1. `self_pts_mask`
2. `ally_pts_mask`
3. `ally_mask`, `enemy_mask`, `target_mask`

---

## 5. 动作空间与动作语义

动作语义是“每个 pursuer 选择一个候选索引”，候选长度随当前时刻动态变化。

环境内部会统一归一化为 one-hot 权重（`_normalize_action`），并处理：

1. 索引动作
2. one-hot/权重向量动作
3. 非法动作回退到首个有效候选

这使网络输出可以是离散索引，也可以是概率分布。

---

## 6. 奖励函数设计

奖励已独立到 [marl/rewards.py](marl/rewards.py)，环境只负责调用。

主要接口：

1. `TODCRewardFunction.compute_step_rewards`
2. `TODCRewardFunction.apply_capture_bonus`
3. `TODCMARLEnv.set_reward_params`

你可以通过三种方式改奖励：

1. 配置文件中的 `reward` 字段
2. 训练参数 `--reward-json '{...}'`
3. 运行时 `env.set_reward_params(...)` 或 `env.reset(options={"reward": {...}})`

总奖励由三部分构成：

1. 距离改善奖励
: 使用最小 P-E 距离改善量
: `r_dist = dist_progress_scale * (last_min_dist - curr_min_dist)`

2. 熵奖励（带可行性门控）
: 对动作分布熵进行鼓励，提升探索
: 仅当 top-k 候选的 `delta_t` 可行时激活
: 系数为 `entropy_scale`

3. 捕获奖励
: 每当有新 Evader 被捕获，给所有 pursuer 共享增益
: `capture_bonus * captured_new / num_E`

终止相关惩罚目前以“结束条件触发”为主，显式负奖励较少，可按需求扩展。

---

## 7. 终止与约束

终止条件：

1. 全部 Evader 被捕获。
2. 发生碰撞（障碍物碰撞或 P-P 安全距离冲突）。

截断条件：

1. `episode_step >= max_episode_steps`。

碰撞检测包括：

1. 点是否落入障碍多边形。
2. 任意 pursuer 对间距是否小于 `collision_dist`。

---

## 8. 网络结构（三方案）

实现文件：[marl/models.py](marl/models.py)。

### 8.1 统一命名

1. A: Concatenative Query Network
2. B: Gated Query Network
3. C: Point-Wise Scoring Network

别名映射由 `resolve_design_mode` 完成，支持 A/B/C、全名、缩写（cqn/gqn/pwsn）。

### 8.2 编码器与输入

网络包含独立编码器：

1. self_uav: 3 维
2. ally_uavs: 3 维
3. self_pts: 8 维
4. ally_pts: 8 维
5. enemies: 3 维
6. targets: 2 维

均通过 MLP 映射到同维度隐藏空间。

### 8.3 注意力分支

共享多头注意力：

1. ally
2. ally_pts
3. enemy
4. target

其中 A/B 使用 ego 查询 self_pts；C 使用 points 查询 ego。

### 8.4 三个策略头差异

1. Concatenative Query Network（A）
: 将多分支上下文 concat 后 MLP 融合，再用 pointer 打分候选。

2. Gated Query Network（B）
: 对各上下文学习 gate 权重后加权融合，再 pointer 打分候选。

3. Point-Wise Scoring Network（C）
: 每个候选点独立聚合上下文并直接打分，无 pointer。

### 8.5 Critic 设计

三方案共享 centralized critic：

1. 对各实体编码做 masked pooling
2. 拼接后输出 value

---

## 9. 训练系统（Online Trainer）

实现文件：[marl/train_online0325.py](marl/train_online0325.py)。

### 9.1 训练算法

当前实现是轻量 A2C 风格在线更新：

1. 按 `action_probs` 采样动作。
2. 单步 bootstrap 目标：
: `target = r + gamma * (1-done) * V(next)`
3. 策略损失：
: `L_policy = -E[logpi * advantage]`
4. 价值损失：
: `L_value = MSE(V, target)`
5. 熵正则：
: `-entropy_coef * entropy`

总损失：

1. `L = L_policy + value_coef * L_value - entropy_coef * entropy`

### 9.2 多方案并训

通过 `build_actor_critic_schemes` 一次构建多个方案，每个方案：

1. 独立环境实例
2. 独立优化器
3. 独立日志前缀

### 9.3 配置文件支持

支持 `--config`（JSON/YAML），CLI 参数可覆盖配置文件。

样例见 [configs/train_online.example.yaml](configs/train_online.example.yaml)。

### 9.4 评估与模型保存

新增能力：

1. 周期评估（`eval_interval`, `eval_episodes`）
2. best checkpoint（`*_best.pt`）
3. last checkpoint（`*_last.pt`）
4. 评估轨迹回放数据（npz）

### 9.5 多 GPU 并行

当前支持两种多卡方式：

1. DataParallel（单进程多卡）
: 适合快速试验，参数：`--multi-gpu true --gpu-ids 0,1,2,3`

2. DDP（torchrun，多进程多卡）
: 推荐正式训练，参数：`--distributed true --ddp-backend nccl`

DDP 下注意：

1. 每张卡一个进程，由 torchrun 注入 `RANK/WORLD_SIZE/LOCAL_RANK`。
2. 主进程（rank0）负责 W&B、评估与 checkpoint 保存。
3. 与 `multi_gpu` 互斥，DDP 启用时会忽略 DataParallel 参数。

---

## 10. W&B 可视化

训练支持 `wandb_mode`：

1. online
: 实时上传。

2. offline
: 本地离线记录，可后续同步。

3. disabled
: 不记录 W&B。

常见日志包括：

1. step 级
: reward, policy_loss, value_loss, entropy, replanned_ratio, inner_ticks, decision_step

2. episode 级
: episode_return, episode_steps, episode_policy_loss, episode_value_loss, episode_entropy

3. eval 级
: eval_return, eval_steps, best_eval_return

---

## 11. 快速开始（dubins 环境）

### 11.1 安装依赖

1. 激活环境
: `conda activate dubins`

2. 安装依赖
: `pip install -r requirements.txt`

### 11.2 启动训练

使用配置文件：

1. `python -m marl.train_online0325 --config configs/train_online.example.yaml`

覆盖关键参数示例：

1. `python -m marl.train_online0325 --config configs/train_online.example.yaml --episodes 500 --wandb-mode online`

单机多卡 DataParallel 示例：

1. `python -m marl.train_online0325 --config configs/train_online.example.yaml --device cuda:0 --multi-gpu true --gpu-ids 0,1,2,3`

DDP（torchrun）4 卡示例：

1. `torchrun --standalone --nproc_per_node=4 -m marl.train_online0325 --config configs/train_online.example.yaml --distributed true --ddp-backend nccl`

DDP（torchrun）8 卡示例：

1. `torchrun --standalone --nproc_per_node=8 -m marl.train_online0325 --config configs/train_online.example.yaml --distributed true --ddp-backend nccl`

DDP 同时覆盖奖励参数示例：

1. `torchrun --standalone --nproc_per_node=8 -m marl.train_online0325 --config configs/train_online.example.yaml --distributed true --reward-json '{"dist_progress_scale":0.08,"entropy_scale":0.05,"capture_bonus":120}'`

---

## 12. 后续可针对性修改建议

### 12.1 改网络结构

优先改 [marl/models.py](marl/models.py)：

1. 增加注意力层数与残差。
2. 引入跨候选 self-attention（可用于 C 方案）。
3. 将 critic 升级为共享图网络或 transformer critic。

### 12.2 改奖励函数

优先改 [marl/rewards.py](marl/rewards.py)：

1. 增加碰撞显式负奖励。
2. 增加时间惩罚（更快拦截）。
3. 增加 assignment 稳定性奖励（减少频繁切换目标）。

### 12.3 改环境在线语义

优先改 [marl/MARL_env.py](marl/MARL_env.py) 的 `step`：

1. 调整 `max_inner_ticks`。
2. 将 `should_replan_now` 判据做成更平滑的事件触发器。
3. 区分“强制决策”与“自然决策”的训练权重。

### 12.4 改训练算法

优先改 [marl/train_online0325.py](marl/train_online0325.py)：

1. 从单步 A2C 升级到 n-step GAE。
2. 增加 PPO clip 目标与 minibatch。
3. 引入 replay buffer + off-policy（如 SAC 离散变体）。
4. 从 DataParallel 迁移到多机 DDP（torchrun + rendezvous）。

### 12.5 改观测与动作

优先改 [marl/obs_generator.py](marl/obs_generator.py) 与 [marl/train_online0325.py](marl/train_online0325.py)：

1. 给候选点加入更多几何特征（障碍风险、视线代价等）。
2. 增加对 ally/enemy 的相对坐标编码。
3. 采用混合动作（先选目标，再选候选点）。

---

## 13. 已知限制

1. 当前训练器是轻量在线更新实现，尚非完整 PPO 工程版。
2. 当前 DDP 主要覆盖单机多卡，暂未内置多机 rendezvous 参数模板。
3. real_mode 的重规划依赖 map 资产完整性，资产缺失时会退到 dummy。

---

## 14. 建议的下一步实验路线

1. 固定一个方案（先 A）做奖励与稳定性消融。
2. 再做 A/B/C 三方案同配置对比。
3. 最后将最佳方案迁移到更严格真实场景并调 `step_mode=decision` 相关超参。

如果你希望，我可以下一步直接给你补一份“实验模板文档”（包含推荐超参数网格、对比实验表头、W&B 面板命名规范），让你可以直接开系统化实验。

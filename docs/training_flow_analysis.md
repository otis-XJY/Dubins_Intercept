# MAPPO 训练流程详细分析

## 1. 整体训练流程概览

整个系统采用 **MAPPO (Multi-Agent PPO)** + **CTDE (Centralized Training, Decentralized Execution)** 范式，训练多个 Pursuer 协同拦截多个 Evader。

**端到端流程：**

```
配置解析 (YAML/CLI) → 环境/模型/优化器初始化 → 多 Episode 训练循环 → 评估 & 保存检查点
```

具体步骤：

1. **配置解析**：[`_parse_args()`](../marl/runners/online_train.py#L1313-L1433) 合并 YAML 文件 + CLI 参数 → `TrainConfig`，并创建带时间戳的运行目录
2. **分布式设置**：[`_setup_distributed()`](../marl/runners/online_train.py#L230-L252) 处理 DDP/DataParallel
3. **模型构建**：[`build_actor_critic_schemes()`](../marl/runners/online_train.py#L535-L540) 按方案名（cqn/gqn/pwsn）创建 Actor-Critic 网络
4. **环境构建**：每个 scheme 创建一个 [`TODCMARLEnv`](../marl/envs/todc_env.py#L56) 实例 + 一个 Adam 优化器
5. **W&B 初始化**：[`_init_wandb()`](../marl/runners/online_train.py#L362-L437) 定义 step-level 和 episode-level 指标的 x-axis 对齐
6. **训练主循环**：每个 episode 依次跑所有 scheme → rollout 收集 → GAE → PPO 更新
7. **评估**：每 `eval_interval` 个 episode 贪心评估 → 保存 best checkpoint
8. **收尾**：保存 last checkpoint，关闭 W&B 和 live server

---

## 2. `train_online` 主循环逐步分析

入口函数：[`train_online(cfg)`](../marl/runners/online_train.py#L494)

### 2.1 初始化阶段

对应代码：[L496-L621](../marl/runners/online_train.py#L496-L621)

| 步骤 | 代码位置 | 做什么 |
|------|---------|--------|
| 分布式初始化 | [L496](../marl/runners/online_train.py#L496) | `_setup_distributed(cfg)` — 设置 DDP rank/world_size |
| 随机种子 | [L497](../marl/runners/online_train.py#L497) | `_set_seed(cfg.seed + rank)` — 保证可复现 |
| 设备选择 | [L499-L533](../marl/runners/online_train.py#L499-L533) | 解析 device（CPU/CUDA/DDP/DataParallel） |
| 模型构建 | [L535-L540](../marl/runners/online_train.py#L535-L540) | `build_actor_critic_schemes()` — 每个 scheme 一个 Actor+Critic 网络 |
| DDP/DP 包装 | [L542-L561](../marl/runners/online_train.py#L542-L561) | 可选 DDP 或 DataParallel 包装 |
| W&B 初始化 | [L563](../marl/runners/online_train.py#L563) | `_init_wandb()` — 定义 metric axis |
| Live server | [L567-L582](../marl/runners/online_train.py#L567-L582) | 可选的 MJPEG 网页实时预览 |
| 环境+优化器 | [L596-L600](../marl/runners/online_train.py#L596-L600) | 每个 scheme 创建 `TODCMARLEnv` + `Adam` 优化器 |
| 历史/统计容器 | [L602-L614](../marl/runners/online_train.py#L602-L614) | 初始化 history、moving average、best_eval_return 等 |

### 2.2 Episode 外层循环

对应代码：[L622-L1177](../marl/runners/online_train.py#L622-L1177)

```python
for ep in 1..episodes:
    for scheme_name, model in models:   # 每个方案独立跑一个完整 episode
```

每个 (episode, scheme) 的完整流程如下：

#### A. Episode 初始化（[L623-L655](../marl/runners/online_train.py#L623-L655)）

1. `model.train()` — 切换训练模式
2. `env.reset(seed=...)` — **调用环境 reset**（详见第 3 部分）
3. 初始化累计量：`ep_return`, `ep_policy_loss`, `ep_entropy`, reward 分项累加器等
4. 准备视频录制回调（可选）

#### B. Rollout 收集（[L725-L837](../marl/runners/online_train.py#L725-L837)）

```python
while not done and not trunc:    # 一个完整 episode 的决策步
```

每步：

1. **观测转张量**：[`_build_model_obs(obs_np, device)`](../marl/runners/online_train.py#L310-L359) — 将 numpy 观测转为 GPU tensor
2. **Actor 前向**：`unwrap.actor_forward(obs_t)` → `action_probs`
3. **采样动作**：`Categorical(probs).sample()` → `actions`, `logp`
4. **Critic 前向**：`unwrap.critic_forward(obs_t)` → `vals`
5. **构建实际动作**：非活跃 pursuer 设为 -1
6. **环境步进**：[`env.step(step_act)`](../marl/envs/todc_env.py#L531-L637) → `next_obs, rewards, terms, truncs, infos`
7. **存储 transition**：`(obs, action, logp, reward, value, done)` 存入 rollout buffer
8. **统计更新**：累加 ep_return、ep_steps、global_step
9. **W&B 日志**：每步记录 step_reward、r_qual/r_global/r_safe/r_time、delta_t_all 等

> **注意**：一次 `env.step()` 内部可能推进多个物理 tick（内层 while 循环），直到需要重规划或终止。

#### C. GAE 计算（[L841-L867](../marl/runners/online_train.py#L841-L867)）

1. 堆叠 rollout 中的 `rewards_np`, `values_np`, `actions_np`, `logp_np`, `dones_np`
2. 对终止状态做 bootstrap：`last_v = critic_forward(last_obs)`
3. [`compute_gae(rewards, values, dones, last_v, gamma, lam)`](../marl/rl/mappo.py#L14-L39) → `(adv, ret)`
4. **优势标准化**：`adv = (adv - adv.mean()) / (adv.std() + 1e-8)`

GAE 公式（[marl/rl/mappo.py L31-38](../marl/rl/mappo.py#L31-L38)）：

```
delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
A_t     = delta_t + gamma * lambda * (1-done) * A_{t+1}    # 从后往前递推
R_t     = A_t + V(s_t)                                      # return target
```

#### D. PPO 小批次更新（[L869-L894](../marl/runners/online_train.py#L869-L894)）

调用 [`ppo_minibatch_update()`](../marl/rl/mappo.py#L42-L162)：

```
for epoch in 1..ppo_epochs:                          # 默认 4 轮
    shuffle(timesteps)
    for minibatch in split(timesteps, mb_size):       # 默认 32
        对每个 t in minibatch:
            new_logp = actor.log_prob(action)
            ratio = exp(new_logp - old_logp)
            surr1 = ratio * advantage
            surr2 = clamp(ratio, 1-eps, 1+eps) * advantage
            policy_loss = -min(surr1, surr2)          # PPO clip
            value_loss = MSE(V, return)
            entropy = Categorical.entropy()
            loss = (policy_loss + 0.5*value_loss - 0.01*entropy) / batch_size
            loss.backward()
        clip_grad_norm_(1.0)
        optimizer.step()
```

返回：`policy_loss, value_loss, entropy, approx_kl, clipfrac, explained_variance, grad_norm`

#### E. Episode 统计与日志（[L906-L1057](../marl/runners/online_train.py#L906-L1057)）

- 计算 episode-level 统计（return、steps、各 loss 均值、SPS、reward 分项均值）
- W&B 记录 episode-level metrics
- 打印 RL 训练日志行（受 `log_interval` 控制）

#### F. 评估（[L984-L1042](../marl/runners/online_train.py#L984-L1042)）

每 `eval_interval` 个 episode：

1. 创建独立的 `eval_env`
2. 贪心评估 `eval_episodes` 次（`argmax` 动作，不采样）
3. 若 mean_eval_return > best → 保存 `{scheme}_best.pt`
4. 保存评估轨迹 `.npz` + 上传 eval video 到 W&B

#### G. 自定义图表（[L1059-L1176](../marl/runners/online_train.py#L1059-L1176)）

每 `wandb_custom_chart_every` 个 episode，生成 `wandb.plot.line` 图表（return、reward 分项、PPO 诊断指标 vs decision steps）。

### 2.3 收尾阶段（[L1178-L1192](../marl/runners/online_train.py#L1178-L1192)）

1. 保存每个 scheme 的 `{scheme}_last.pt`
2. `wandb.finish()`
3. 停止 live server
4. 清理分布式进程组

---

## 3. 环境 Reset 的详细步骤

入口：[`TODCMARLEnv.reset()`](../marl/envs/todc_env.py#L421-L529)

这是一个非常复杂的初始化过程，核心是**先跑到第一个重规划时刻，再构建初始观测**。

### 3.1 基础重置（[L421-L444](../marl/envs/todc_env.py#L421-L444)）

```python
super().reset(seed=seed)           # Gymnasium 标准 reset
选择 TimeMap -> 加载静态资产       # 可切换地图
选择 Evader Profile -> 加载逃逸者轨迹
episode_step = 0, decision_step = 0, t = 0, t_all = 0
```

### 3.2 初始化实体状态（[L445-L478](../marl/envs/todc_env.py#L445-L478)）

| 变量 | 含义 |
|------|------|
| `PosE = Evader.copy()` | 逃逸者初始位置 (num_E, 3) |
| `PosP = PStart_Point.copy()` | 追捕者初始位置 (num_P, 3) |
| `PathPtrue` | 追捕者真实走过的轨迹（逐段拼接） |
| `PathP` | 追捕者规划路径（全局 pid 索引） |
| `PathE` | 逃逸者预测路径（截取到 `length_E_max` 长度） |
| `Capflag` | 连续捕获标记 (num_E,) bool |
| `Capflag_full` | 全局捕获标记（单调置 True） |
| `UnCapPid / UnCapEid` | 未捕获的追捕者/逃逸者全局 ID |
| `pairs_realE2P` | 匹配对 [(eid_global, pid_global), ...] |
| `IC_candidates` | 拦截候选表（25 列） |
| `assigned_eid_full` | 每个 pursuer 分配到的 eid (-1=无) |

### 3.3 内层预热循环（[L480-L496](../marl/envs/todc_env.py#L480-L496)）— 核心

```python
while not np.all(Capflag) and (t_all < length_E_max / v_E):
    _phase_update()                                  # 推进时间 t, t_all；过滤已捕获配对
    _advance_from_paths()                            # 按规划路径更新 PosP, PosE
    need_replan, terminal = _phase_check_decision()  # 碰撞/资产/全捕获/DWA 检查
    if terminal: break
    if need_replan:
        _compute_isomap_intercept_candidates()       # 传统规划管线：IsoMap + 拦截候选
        _apply_hungarian_and_paths()                 # 匈牙利分配 + 路径拼接
        break
    _update_capflag_from_geometry()                  # 几何捕获检测
    _update_capflag_full_from_geometry()             # 全局捕获标记更新
```

**这段循环的目的**：从初始状态推进仿真，直到第一次需要重规划（DWA 检测到逃逸者轨迹偏离预测），此时构建出第一批拦截候选点，才能让 RL agent 做选择。

关键子函数：

- [`_phase_update()`](../marl/envs/todc_env.py#L1069-L1081) — 时间推进并同步未捕获 E-P 配对
- [`_advance_from_paths()`](../marl/envs/todc_env.py#L639-L665) — 按规划路径更新 PosP/PosE 实体位置
- [`_phase_check_decision()`](../marl/envs/todc_env.py#L1084-L1099) — 碰撞/资产 breach/全捕获/DWA 偏离检测
- [`_compute_isomap_intercept_candidates()`](../marl/envs/todc_env.py#L682-L815) — 传统规划管线核心：IsoMap 构建 + 拦截候选表生成
- [`_apply_hungarian_and_paths()`](../marl/envs/todc_env.py#L817-L867) — 匈牙利最优匹配 + 路径拼接

### 3.4 后处理（[L497-L529](../marl/envs/todc_env.py#L497-L529)）

1. **目标推断**：[`_predict_facility_ranks_for_e()`](../marl/envs/todc_env.py#L667-L680) → 推断每个逃逸者攻击的目标设施
2. **初始化稳定距离**：[`_compute_curr_min_dist_stable()`](../marl/envs/todc_env.py#L1288-L1312) → 用于第一步的 progress reward
3. **构建观测**：[`_build_obs()`](../marl/envs/todc_env.py#L1425-L1450) → 返回完整的 ego-centric 观测 dict
4. **动态 K 同步**：[`_sync_dynamic_k(obs)`](../marl/envs/todc_env.py#L186-L190) → 若候选数 K 变化则重建 action/observation space
5. **返回** `(obs, info)` — info 包含 `num_candidates`, `t_all`, `time_map`, `profile` 等元信息

---

## 4. 总体流程图

```
train_online(cfg)
  |
  +-- 初始化: 设备/模型/环境/优化器/W&B
  |
  +-- for ep in 1..N:                    <-- episode 循环
  |    for scheme in [cqn, gqn, pwsn]:   <-- 每个方案独立
  |      |
  |      +-- env.reset()                 <-- 预热到第一个重规划点
  |      |    +-- 加载地图+逃逸轨迹
  |      |    +-- 内层 while: 推进仿真直到 DWA 触发重规划
  |      |    +-- IsoMap + 匈牙利分配 -> 构建候选
  |      |    +-- 返回 (obs, info)
  |      |
  |      +-- while not done/trunc:       <-- rollout 收集
  |      |    +-- actor(obs) -> sample action
  |      |    +-- critic(obs) -> value
  |      |    +-- env.step(action) -> 内层多 tick 推进 -> next_obs, reward
  |      |
  |      +-- GAE(gamma, lambda) 计算 advantage + return
  |      |
  |      +-- PPO minibatch 更新 (4 epochs, batch=32)
  |      |    +-- clip ratio + value loss + entropy bonus
  |      |
  |      +-- 日志/评估/保存
  |
  +-- 保存 last checkpoint, 关闭 W&B
```

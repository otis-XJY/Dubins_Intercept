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
: 观测构建器，生成 V_P / V_E / V_C 及 mask。

3. [marl/models.py](marl/models.py)
: 三种策略网络结构（A/B/C）与统一命名、构建接口。

4. [marl/train_online.py](marl/train_online.py)
: 在线训练入口，支持多方案并训、W&B、配置文件、评估与 best 保存。

5. [configs/train_online.example.yaml](configs/train_online.example.yaml)
: 训练配置样例。

---

## 3. 环境设计（TODCMARLEnv）

环境类：`TODCMARLEnv`，定义在 [marl/MARL_env.py](marl/MARL_env.py)。

### 3.1 两种运行模式

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

## 4. 观测空间设计（Phase-2）

观测由 [marl/obs_generator.py](marl/obs_generator.py) 生成，兼容两套字段名：

1. Phase-2 字段
: `V_P`, `V_E`, `V_C`, `V_C_mask`。

2. 环境通用字段
: `pursuers`, `evaders`, `candidates`, `candidate_mask`。

### 4.1 Pursuer 节点 V_P（每个 P 6 维）

特征：

1. x, y
2. v_x, v_y
3. cos(theta), sin(theta)

### 4.2 Evader 节点 V_E（每个 E 8 维）

特征：

1. x, y
2. v_x, v_y
3. cos(theta), sin(theta)
4. inferred_target_x, inferred_target_y

### 4.3 Candidate 节点 V_C（每个候选 6 维）

特征：

1. candidate_x, candidate_y
2. t_p, t_e
3. delta_t = t_e - t_p
4. cost

并使用 `V_C_mask` 指示有效候选，支持 padding 到固定 `k_max`。

---

## 5. 动作空间与动作语义

动作空间定义为 `MultiDiscrete([k_max] * num_P)`，即每个 pursuer 选择一个候选索引。

环境内部会统一归一化为 one-hot 权重（`_normalize_action`），并处理：

1. 索引动作
2. one-hot/权重向量动作
3. 非法动作回退到首个有效候选

这使网络输出可以是离散索引，也可以是概率分布。

---

## 6. 奖励函数设计

定义在 [marl/MARL_env.py](marl/MARL_env.py) 的 `_compute_rewards` 及 step 捕获奖励逻辑。

总奖励由三部分构成：

1. 距离改善奖励
: 使用最小 P-E 距离改善量
: `r_dist = dist_reward_scale * (last_min_dist - curr_min_dist)`

2. 熵奖励（带可行性门控）
: 对动作分布熵进行鼓励，提升探索
: 仅当 top-k 候选的 `delta_t` 可行时激活
: 系数为 `entropy_lambda`

3. 捕获奖励
: 每当有新 Evader 被捕获，给所有 pursuer 共享增益
: `+100 * captured_new / num_E`

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

实现文件：[marl/train_online.py](marl/train_online.py)。

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

1. `python -m marl.train_online --config configs/train_online.example.yaml`

覆盖关键参数示例：

1. `python -m marl.train_online --config configs/train_online.example.yaml --episodes 500 --wandb-mode online`

---

## 12. 后续可针对性修改建议

### 12.1 改网络结构

优先改 [marl/models.py](marl/models.py)：

1. 增加注意力层数与残差。
2. 引入跨候选 self-attention（可用于 C 方案）。
3. 将 critic 升级为共享图网络或 transformer critic。

### 12.2 改奖励函数

优先改 [marl/MARL_env.py](marl/MARL_env.py) 的 `_compute_rewards`：

1. 增加碰撞显式负奖励。
2. 增加时间惩罚（更快拦截）。
3. 增加 assignment 稳定性奖励（减少频繁切换目标）。

### 12.3 改环境在线语义

优先改 [marl/MARL_env.py](marl/MARL_env.py) 的 `step`：

1. 调整 `max_inner_ticks`。
2. 将 `should_replan_now` 判据做成更平滑的事件触发器。
3. 区分“强制决策”与“自然决策”的训练权重。

### 12.4 改训练算法

优先改 [marl/train_online.py](marl/train_online.py)：

1. 从单步 A2C 升级到 n-step GAE。
2. 增加 PPO clip 目标与 minibatch。
3. 引入 replay buffer + off-policy（如 SAC 离散变体）。

### 12.5 改观测与动作

优先改 [marl/obs_generator.py](marl/obs_generator.py) 与 [marl/train_online.py](marl/train_online.py)：

1. 给候选点加入更多几何特征（障碍风险、视线代价等）。
2. 增加对 ally/enemy 的相对坐标编码。
3. 采用混合动作（先选目标，再选候选点）。

---

## 13. 已知限制

1. 当前训练器是轻量在线更新实现，尚非完整 PPO 工程版。
2. `_build_model_obs` 中部分特征是从环境字段映射而来，仍有进一步精细化空间。
3. real_mode 的重规划依赖 map 资产完整性，资产缺失时会退到 dummy。

---

## 14. 建议的下一步实验路线

1. 固定一个方案（先 A）做奖励与稳定性消融。
2. 再做 A/B/C 三方案同配置对比。
3. 最后将最佳方案迁移到更严格真实场景并调 `step_mode=decision` 相关超参。

如果你希望，我可以下一步直接给你补一份“实验模板文档”（包含推荐超参数网格、对比实验表头、W&B 面板命名规范），让你可以直接开系统化实验。

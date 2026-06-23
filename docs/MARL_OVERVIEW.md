# Dubins_Intercept 项目全景与 MARL 设计实现总览（基于当前代码）

> 目标：给后续分析/改造提供“可直接对照代码”的统一认知。
> 范围：覆盖项目主干（地图资产生成、传统规划、MARL 环境/观测/奖励/网络/MAPPO/训练器、测试与运行链路）。

---

## 1. 项目在解决什么问题

该项目面向 **多追捕者（Pursuer）拦截多逃逸者（Evader）** 的任务，场景包含障碍、设施目标点（`ValuePos`）与时间演化约束。

核心不是直接端到端输出底层控制，而是采用混合范式：

1. **传统几何/图搜索规划** 先生成可行拦截候选；
2. **MARL 策略** 在候选集中做离散选择；
3. 环境以“重规划事件”为步长推进，使 RL 步与在线决策时刻对齐。

一句话总结：

- **传统规划负责“找候选”**；
- **MARL 负责“选候选”**。

---

## 2. 全项目分层结构（从离线到在线）

### 2.1 地图与轨迹资产离线构建（训练前置）

- `main/batch_env_init.py`：批量生成 `Map.jbl`（地形、障碍、起点、转移点等）。
- `main/batch_env_build.py`：批量生成 IsoMap 与路径资产（`IsoMap*`、`pathFinal*`）。
- `main/batch_evader_paths.py`：批量生成/编辑逃逸轨迹 `PathE2Val_true.jbl` 等。

产物组织在 `map/<TimeMap>/<TimeEpath>/...`，训练环境会直接加载这些 `joblib` 文件。

### 2.2 传统规划内核

- `intercept/`：IsoMap/IsoPair/Prediction 等传统流程。
- `A_dubins/`：Dubins 路径求解和搜索子模块。

MARL 环境中实际调用的关键函数包括：

- `predictLikelyTargetNew`
- `obtainDWAprePath`
- `obtainPE2TP`
- `insertIsoMapP2TP`
- `obtainPTP2TP_IsoPos_timeShift2`
- `obtainTask_timeShift2`

### 2.3 MARL 主体

- `marl/envs/todc_env.py`：环境动力学与决策语义。
- `marl/obs/generator.py`：观测构造（含 mask 与存活筛选）。
- `marl/rewards/todc_reward.py`：模块化奖励。
- `marl/model_blocks/`：当前实际被网络入口调用的模型实现。
- `marl/rl/mappo.py`：GAE + PPO 更新。
- `marl/runners/online_train.py`：在线训练入口（单机/DP/DDP/W&B/eval/checkpoint/live）。

---

## 3. 环境（`TODCMARLEnv`）设计与执行语义

## 3.1 关键语义

- 一次 `env.step(action)` = 一次 **在线决策步**。
- 该决策步内部会跑多个物理 tick（`_phase_update -> _advance_from_paths -> _phase_check_decision`）。
- 直到触发以下之一才返回给 RL：
  - 需要重规划（`need_replan`）
  - 终止条件（全捕获/碰撞/资产突破/时间上限）
  - 回合截断（`max_episode_steps`）

## 3.2 reset 流程（重要）

`reset()` 并不只是清状态，而是会先推进到“下一次可决策点”：

1. 根据 `time_map_mode`/`time_map_id` 选 TimeMap；
2. 根据 `evader_profile_mode`/`profile_id` 选 TimeEpath；
3. 初始化 `PosP/PosE/PathP/PathE/Capflag_full/pairs_realE2P`；
4. 内层循环推进，直到触发重规划；
5. 完成首次候选构建与匈牙利分配；
6. 构建 obs 并同步动态 `k_max`。

因此 `reset` 返回的 obs 已经是“可行动”的决策时刻观测。

## 3.3 动态动作空间与观测空间

候选数 `K` 随重规划变化，`_sync_dynamic_k` 会重建：

- `action_space = MultiDiscrete([K]*num_P)`
- `observation_space` 中 `self_pts/ally_pts/*_mask` 的相关维度

这意味着该环境不是固定维 action/obs 空间。

## 3.4 动作归一化与 fail-fast

`_normalize_action` 兼容输入类型：

- 每机离散索引；
- 每机概率/权重向量；
- dict 形式 `{"p_i": ...}`。

规则：

- 对无任务机（`pursuer_active=0`）动作索引强制设为 `-1`；
- 对有任务机若选到 mask=0 或越界，直接抛错（fail-fast）；
- 返回 one-hot `norm`、离散 `idx`、以及决策时刻 `obs` 快照（用于奖励对齐）。

## 3.5 配对、捕获与一致性校验

环境维护两类关键状态：

- `Capflag_full[num_E]`：全局 eid 捕获标记（单调 True）；
- `pairs_realE2P[n_pair,2]`：当前活跃配对（全局 eid, pid）。

并通过以下机制保证一致：

- `_sync_assigned_eid_full_from_pairs`
- `_validate_pair_data_integrity`
- `_validate_capflag_consistency`
- `_update_capflag_full_from_geometry`

`_phase_update` 会按 `Capflag_full` 过滤已捕获目标对应配对。

## 3.6 重规划与 fallback 机制

重规划路径为：

1. `predictLikelyTargetNew` 推断 Evader 攻击目标；
2. `obtainPE2TP` 计算 E->TP / P->TP 路径；
3. `insertIsoMapP2TP` 拼接动态等时面；
4. `obtainPTP2TP_IsoPos_timeShift2` + `obtainTask_timeShift2` 生成候选；
5. Hungarian 计算分配。

若候选为空或配对不完整，进入 `_apply_fallback_assignment_and_candidates`：

- 用最近直线/可用路径补全分配；
- 生成带 `-2` sentinel 的 fallback 候选；
- 奖励模块会对该类候选施加 fallback 惩罚。

---

## 4. 观测构造（`TODCObservationGenerator`）

## 4.1 输入中间量

观测由三类节点构建：

- `v_p_nodes`：Pursuer 几何状态（位置、速度分量、朝向 sin/cos）
- `v_e_nodes`：Evader 几何 + 推断目标坐标
- `v_c_nodes`：每机候选点特征（8 维）

`CandidateColumns` 明确了候选表列语义（`Delta_t/Delta_d/path_L/cost_*` 等）。

## 4.2 输出键与维度语义

观测是 dict，核心键包括：

- 自机与友机：`self_uav`, `allies_local`, `ally_mask`
- 分配敌机：`enemy_assigned_self`, `enemy_assigned_per_ally`, `enemy_self_mask`, `ally_enemy_mask`
- 设施目标：`asset_target_self`, `asset_target_per_ally`
- 候选：`self_pts`, `ally_pts`, `self_pts_mask`, `ally_pts_mask`
- 全局实体：`enemies`, `targets`, `assets`, 对应 `*_mask`
- 训练辅助：`pursuer_active`, `reward_nodes`, `self_Capflag`

## 4.3 存活约束与清零策略

- 存活集合由 `pairs_realE2P` 推导（`alive_pid_eid_sets`）。
- 死亡 pursuer / evader 的对应通道全部置零并 mask=False。
- `ally_pts` 由其它存活 pursuer 的 `self_pts` 重建。

这保证网络输入维度固定、语义随存活集变化。

---

## 5. 奖励模块（`TODCRewardFunction`）

## 5.1 参数配置

`RewardConfig` 支持以下大类：

- 拦截质量项权重：`alpha_dist_v/beta_delta_t/gamma_delta_d/gamma_delta_theta/lambda_path_l`
- 全局协同项：`w_global/div_sigma/assign_penalty`
- 安全项：`safe_dist_min/safe_penalty_scale`
- 时间与终局：`step_cost/terminal_capture_bonus/terminal_asset_loss_penalty`
- fallback 与进度项：`fallback_penalty/dist_progress_scale`

## 5.2 步级奖励分解

`compute_step_rewards` 中每个智能体奖励由以下项累加：

1. `r_progress`：与上一步最小分配距离差相关（可关闭）；
2. `r_fallback`：若选中 sentinel `-2` 候选则惩罚；
3. `r_qual`：候选质量项加权和（从 `reward_nodes` 取特征）；
4. `r_global`：分散奖励 + 多机同目标惩罚；
5. `r_safe`：机间过近惩罚；
6. `r_time`：按 `sim_time_elapsed/sim_dt` 缩放的步时成本。

并支持 `return_details=True` 产出细分字典，训练器会完整记录。

## 5.3 终局奖励

`apply_terminal_rewards` 对所有 agent 同时加/减：

- 捕获增益：`terminal_capture_bonus * captured_delta / num_e`
- 资产突破惩罚：`terminal_asset_loss_penalty`

---

## 6. 策略网络与集中价值（当前实现）

## 6.1 实际导入链路

训练入口 `marl.nn.models` 最终走到：

- `marl.nn.blocks.network` -> `marl.model_blocks.network`

即当前网络主实现在 `marl/model_blocks/`。

## 6.2 方案与别名

支持四种设计模式：

- A: `Concatenative Query Network`
- B: `Gated Query Network`
- C: `Point-Wise Scoring Network`
- D: `Two-Stage Residual Network`

`resolve_design_mode` 支持全名/缩写别名解析（如 `cqn/gqn/pwsn/tsr`）。

## 6.3 编码器（Embeddings）

当前编码器做了较强结构化处理：

1. 坐标先转到 **ego frame**（平移+旋转）；
2. 候选点拆为几何分支(5维)+质量分支(6维)编码；
3. `self_pts` 与 `ally_pts` 共享编码器，用 role embedding 区分来源；
4. 自机使用可学习 ego token；
5. 统一 LayerNorm 规范化。

## 6.4 注意力与融合

- A/B：`HeterogeneousAttentionAB`，自机 query 对 7 路上下文并行 cross-attn。
- A：`ConcatMLPFusion8` 融合后 pointer 打分。
- B：`SoftGatingFusion7` 门控融合后 pointer 打分。
- C：候选作为 query，对上下文做 point-wise cross-attn，再逐点 MLP 打分。
- D：两阶段结构：
  - 阶段1：self context 逐点打分；
  - 阶段2：友军协同注意力产生残差 delta 并加到阶段1 logits。

## 6.5 actor 输出后处理

`postprocess_mask_and_sample` 统一处理：

- `mask=0` 候选 logits 置极小值；
- 若整行无可用点，回退为 index=0 概率1（后续再由 active 逻辑屏蔽）；
- 输出 `action_logits/action_probs/best_candidate_idx`。

## 6.6 critic：CTDE 的集中价值实现

当前 critic 是 `PermInvariantCritic`：

- 输入每机 `h_env(P,D)` + 全局编码 `h_global(D)`；
- 用可学习 query 对 agent token 做注意力池化得全局摘要；
- 对每机输出 `V_i`。

关键特性：

- 参数在 `__init__` 固定创建，不依赖 P 动态重建；
- 对 agent 顺序置换不敏感；
- 避免了旧式“前向时临时建 critic 导致优化器跟踪不到参数”的风险。

---

## 7. MAPPO 算法实现（`marl/rl/mappo.py`）

## 7.1 GAE

`compute_gae` 输入形状：

- `rewards[T,P]`
- `values[T,P]`
- `dones[T]`
- `last_value[P]`

按时间反向递推：

- `delta_t = r_t + gamma * V_{t+1} * non_term - V_t`
- `gae_t = delta_t + gamma * lam * non_term * gae_{t+1}`

输出 `adv[T,P]` 与 `ret[T,P]`。

## 7.2 PPO 小批更新

`ppo_minibatch_update` 关键点：

1. rollout 按时间步打乱分 minibatch；
2. ratio 使用 logratio clamp 防溢出；
3. policy 用 clip surrogate；
4. value 用 clipped value loss；
5. 每批梯度裁剪 `clip_grad_norm_`；
6. 用 `approx_kl` 做 early stop；
7. 输出 `policy/value/entropy/approx_kl/clipfrac/ev/grad_norm`。

---

## 8. 在线训练器（`online_train.py`）

## 8.1 配置合并与运行目录

- 支持 YAML + CLI 覆盖。
- `reward` 可用 `--reward-json` 覆盖。
- `env` 可用 `--env-json` 覆盖。
- 每次运行自动创建：`output/runs/<timestamp>/`
  - `checkpoints/`
  - `eval_traces/`
  - `train_config.yaml` 备份
- W&B 运行名自动追加时间戳。

## 8.2 设备与并行

支持三类：

1. 单设备（CPU/GPU）；
2. DataParallel（非分布式多卡）；
3. DDP（`torchrun`，推荐）。

并带 fail-fast 检查：如指定 CUDA 但环境无 CUDA 会直接报错。

## 8.3 多 scheme 训练组织

每个 scheme 都独立创建：

- env
- model
- optimizer
- scheduler

循环结构：

- `for episode`
  - `for scheme`
    - 收集一整条 rollout
    - GAE
    - PPO 更新
    - 记录日志
    - 周期 eval 与 best 保存

不同 scheme 间不共享参数。

## 8.4 rollout 与动作执行

每个 RL 步：

1. `actor_forward` 产生 `action_probs`；
2. Categorical 采样动作；
3. 对 `pursuer_active=False` 的 agent 强制动作 `-1`；
4. 调 `env.step`；
5. 收集 obs/action/logp/reward/value/done。

之后做标准化 advantage（可裁剪）。

## 8.5 评估、可视化与实时服务

- 评估 `_run_eval_episode` 使用 greedy `argmax`。
- 可记录：step 图、tick 图、训练视频、评估视频。
- 可启用 `live_server` 提供 MJPEG + status JSON 页面。
- 训练结束保存 `{scheme}_last.pt`。

---

## 9. 端到端运行链路（建议）

1. 批量构建地图资产：
   - `python -m main.batch_env_init --config configs/env_init.yaml`
   - `python -m main.batch_env_build --config configs/env_build.yaml`
   - `python -m main.batch_evader_paths --config configs/evader_paths.yaml`
2. 环境冒烟：
   - `python scripts/smoke_verify_env.py`
3. 训练：
   - `python -m marl.runners.online_train --config configs/train_online0324.yaml`

---

## 10. 代码级重要约束与易踩点

1. **动态 K**：候选数变化会重建空间，和固定空间假设不兼容。
2. **pairs_realE2P 必须存在**：观测生成依赖该映射，缺失会抛错。
3. **fallback 行为是显式语义**：`-2` sentinel 会影响奖励与路径处理。
4. **资产突破判定使用环境 `collision_dist`**（与奖励配置中的 `asset_breach_radius` 不完全同源）。
5. **严格 fail-fast**：非法 action/mask/shape 多处直接 `ValueError`。
6. **无 `env.captured` 数组**：捕获语义以 `Capflag_full` 和 info 字段为准。

---

## 11. 后续改造时的“改哪里”速查

- 改候选生成/配对：`marl/envs/todc_env.py` + `intercept/IsoPair/*`
- 改观测字段：`marl/obs/generator.py`
- 改奖励分解：`marl/rewards/todc_reward.py`
- 改网络结构：`marl/model_blocks/*`（`marl/nn/blocks/network.py` 为转发）
- 改算法细节：`marl/rl/mappo.py`
- 改训练流程与工程能力：`marl/runners/online_train.py`

---

## 12. 总结

该项目是一个工程化程度较高的“**传统规划 + MARL 离散决策**”系统：

- 环境语义明确对齐在线重规划；
- 观测、奖励、策略、训练器模块边界清晰；
- 网络已从早期 A/B/C 扩展到 D，并采用置换不变集中 critic；
- 训练侧支持多 scheme、多卡、评估、可视化、在线监控。

若后续要做性能/稳定性提升，优先建议从三条线并行：

1. 观测-动作因果对齐（候选质量与任务协同信息）；
2. 奖励信用分配细化（终局共享奖励与个体贡献平衡）；
3. PPO 训练稳定性调优（KL/clip/entropy/adv clip 与 curriculum）。

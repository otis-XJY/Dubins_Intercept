---
name: MARL 升级迭代计划
overview: 根据用户确认：删除 dummy 配置；参数全部由外置 YAML 驱动；重构观测/网络（保留多头注意力并扩展融合）；训练升级为 MAPPO；动态动作维与 pairs/非法动作等行为保持原样。
todos:
  - id: remove-dummy-config
    content: 删除 allow_dummy_if_missing 及相关 README/YAML/测试引用（原为鲁棒性预留，现直接移除）
    status: pending
  - id: yaml-all-params
    content: 统一外置 YAML：TrainConfig、TODCMARLEnv、RewardConfig 全量键；加载合并与校验；减少代码内隐式默认
    status: pending
  - id: obs-spec-rewrite
    content: 按新观测规格重写 obs_generator + env observation_space + _build_model_obs
    status: pending
  - id: model-fusion-mha
    content: 扩展 UAVInterceptionNetwork：新编码器与 MHA 分支、融合维度调整（A/B/C 三方案对齐新输入）
    status: pending
  - id: mappo-train
    content: 实现 MAPPO（GAE、clip、minibatch、集中式 critic）；更新 train 脚本与配置项
    status: pending
isProject: true
---

# MARL 升级迭代计划（用户确认版）

## 用户决策摘要


| 编号    | 决策                                                                                                  |
| ----- | --------------------------------------------------------------------------------------------------- |
| **1** | **删除** `allow_dummy_if_missing` 及文档中的 dummy 模式描述；该开关原为鲁棒性预留但环境未使用，清理训练脚本、YAML、README、测试中的相关字段，避免误导。 |
| **2** | **所有可调参数**通过**外置 YAML** 传入（训练、环境动力学、奖励等）；代码侧仅保留极小 fallback 或显式报错缺键，避免“改 YAML 不生效”。                  |
| **3** | **重构网络输入**（见下节）；**保留**现有 **多头注意力** 结构，并因输入语义变化**重做信息融合**（各 design A/B/C 同步调整）。                      |
| **4** | 训练算法 **升级为 MAPPO**（多智能体 PPO：GAE、ratio clip、多 epoch 等；critic 使用 CTDE 式集中信息）。                         |
| **5** | **忽略**“变长输出”相关额外处理：当前指针/打分头已按 mask 处理动态 K，**保持原样**。                                                 |
| **6** | `pairs_realE2P` 依赖与观测生成逻辑：**先保持原样**（用户暂无法判断）。                                                       |
| **7** | 动作合法性 / fail-fast 行为：**先保持原样**。                                                                     |


---

## 一、删除 dummy 相关（任务 1）

- 从 [marl/train_online0325.py](marl/train_online0325.py) 移除 `TrainConfig.allow_dummy_if_missing`、`_todc_marl_env_dict` 中的对应键、argparse 参数。
- 从 [configs/train_online0324.yaml](configs/train_online0324.yaml)、[configs/train_online.example.yaml](configs/train_online.example.yaml) 删除该键。
- 更新 [README.md](README.md) [docs/MARL_OVERVIEW.md](docs/MARL_OVERVIEW.md)：去掉 `dummy_mode` / `allow_dummy` 小节。
- 更新测试 [tests/test_*.py](tests/) 中传入 env config 的字段，不再包含该项。

---

## 二、参数全部由 YAML 驱动（任务 2）

**目标**：单一配置文件列出全部有效超参，加载路径清晰。

**建议结构**（可在默认 YAML 中分块，加载时合并）：

- **train**：`episodes`、`gamma`、`lr`、`ppo_`*（见 MAPPO）、`device`、`schemes`、`wandb_`*、`eval_*`、`save_*` 等（对应 `TrainConfig` 全字段）。
- **env**：`map_root`、`data_dir`、`time_map`、`collision_dist`、`cap_dist`、`cap_angle`、`cap_dist_ref`、`max_episode_steps`、`time_res`、`step_mode`、`enable_dwa_replan`、`evader_profile_mode`、`render_mode` 等（对应 [marl/MARL_env.py](marl/MARL_env.py) `config.get` 使用的键）。
- **reward**：与 [marl/rewards.py](marl/rewards.py) `RewardConfig` 一致。
- **obs**（新增，配合第三节）：如 `ally_perception_radius`（友机“一定范围内”的距离阈值，单位与地图一致）等。

**实现要点**：

- 在 [marl/train_online0325.py](marl/train_online0325.py)（或抽出的 `config_loader.py`）中：**从 YAML 构建完整 `TrainConfig` 与嵌套 `env` / `reward` 字典**；对 `TODCMARLEnv` **只传合并后的 env 段**，避免顶层键与环境键混淆。
- 可选：`--config` 为唯一主入口，CLI 仅作覆盖；缺键时 **打印警告或报错**（按团队偏好二选一，计划中默认：**关键路径缺键报错**）。

---

## 三、新观测规格与网络融合（任务 3 + 4）

### 3.1 语义定义（按用户描述）

对**每个 Pursuer**（批维仍为 `P`，与现有一致）：


| 张量            | 形状    | 含义                                                                                    |
| ------------- | ----- | ------------------------------------------------------------------------------------- |
| 自身状态          | `1×3` | x, y, θ（与现 `self_uav` 对齐）                                                             |
| 友方（一定范围内）     | `A×3` | 仅包含**感知半径内**的其它 Pursuer；**mask** 区分有效槽位与 padding                                      |
| 敌方（**分配的**敌方） | `1×3` | 当前 `pairs_realE2P` 中与该机绑定的 Evader 状态（非全敌复制）                                           |
| 友军相对**该分配敌方** | `A×3` | 其它友机在“针对同一分配目标”的语义下的特征（建议实现为：友机状态在分配敌机坐标系下的相对量，或与该敌机的相对位置/航向；*实现细节在编码阶段与 `enc`_ 对齐**） |
| 拦截点（自身候选）     | `N×8` | 与原 `self_pts` / 候选特征一致                                                                |
| 友方拦截点         | `M×8` | 其它友机候选拼接，与现 `ally_pts` 类似但维数与 mask 随 A、K 变化                                           |


**掩码**：`ally_mask`、`ally_vs_enemy_mask`（若与 ally 槽位共享同一集合，可复用一支 mask，否则分开）、`self_pts_mask`、`ally_pts_mask`；**不再**向策略网络喂全量 `enemies [E]` × 全 `targets [E]` 的复制张量（除非 MAPPO critic 需要全局状态时再在 critic 分支拼接，见 4.2）。

### 3.2 环境侧改动

- [marl/obs_generator.py](marl/obs_generator.py)：`generate` / `_build`_* 按上表产出新键名（建议显式命名如 `self_uav`、`allies_local`、`enemy_assigned`、`allies_wrt_assigned_enemy`、`self_pts`、`ally_pts` 及对应 `*_mask`）。
- [marl/MARL_env.py](marl/MARL_env.py)：`observation_space` 与 `_build_obs` 对齐；`_compute_rewards` 若仍依赖 `reward_nodes` / 旧键，**在 env 内保留内部完整张量用于计奖**，对外暴露给网络的键按新规格（避免破坏 [marl/rewards.py](marl/rewards.py)）。
- 感知半径：由 YAML `obs.ally_perception_radius`（或 `env` 下同名）读入 `TODCObservationGenerator`。

### 3.3 网络侧改动（保留多头注意力）

- [marl/models.py](marl/models.py) `UAVInterceptionNetwork`：
  - 为 **新实体类型**增加独立 `enc`_*（3 维与 8 维分支复用/拆分按设计）。
  - **MHA**：保留 `nn.MultiheadAttention` 模式；将原 `mha_ally / mha_enemy / mha_target / ...` **映射到新序列**（例如：ego query → allies_local；ego query → enemy_assigned 序列长为 1；ego query → allies_wrt_assigned_enemy；ego query → self_pts / ally_pts）。
  - **融合层**：Design A 的 `fusion_mlp`、Design B 的 `gate`_*、Design C 的 `scoring_mlp` 的 **输入维数**随分支数量变化，需重新计算 `hidden_dim * n` 并调整 `Linear`。
- **Critic（MAPPO 前可先改观测输入，见下）**：在 MAPPO 中 critic 使用 **全局或拼接状态**，与 actor 的局部观测分离。

---

## 四、MAPPO（任务 5）

### 4.1 算法要点

- **Rollout 缓冲**：每个环境步存储每智能体 `obs`、`action`、`log π`、`reward`、`done`；按 **N 步或整段 episode** 截断计算优势。
- **GAE(λ)**：与单智能体 PPO 相同，对每个 agent **独立**或对 team **共享 bootstrap**（需选定一种；默认：**每智能体独立 GAE**，与现有 per-agent reward 一致）。
- **PPO clip**：`L_clip` + value loss + entropy；可多 epoch 小批量更新。
- **梯度**：保留 `clip_grad_norm`；学习率、clip 范围、epoch 数、batch 大小 **全部来自 YAML**。

### 4.2 Actor / Critic 信息集（CTDE）

- **Actor（分散）**：输入 **第三节**的局部观测（每机一份），输出该机候选上的分布。
- **Critic（集中）**：输入 **拼接所有智能体局部观测** 或 **环境提供的全局向量**（若需新增 `global_state` 字段，在 `MARL_env.step` 的 `info` 或单独 `build_central_obs()` 中提供）。默认实现：**将所有 agent 的局部 obs 在特征维或序列维拼接** 后送入 `critic`，与论文中 MAPPO 常见做法一致。

### 4.3 代码位置

- 新模块建议：`marl/mappo.py` 或 `marl/rollout_buffer.py` + 修改 [marl/train_online0325.py](marl/train_online0325.py) 主循环；或新建 `marl/train_mappo.py` 以免旧脚本混杂。

---

## 五、明确保持原样（用户 5–7）

- **动态 K**、`_sync_dynamic_k`、`action_space` 重建：**不**增加额外抽象；继续 mask softmax。
- `**pairs_realE2P` 与 `_build_c_nodes` 前置条件**：逻辑不改，仅观测内容按新语义从现有数据派生。
- **非法动作 / ValueError**：与现有一致。

---

## 六、依赖文件清单（实施时）


| 模块  | 文件                                                                                                                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 配置  | [configs/train_online0324.yaml](configs/train_online0324.yaml)、example、加载逻辑                                                                                                                                |
| 环境  | [marl/MARL_env.py](marl/MARL_env.py)                                                                                                                                                                       |
| 观测  | [marl/obs_generator.py](marl/obs_generator.py)                                                                                                                                                             |
| 网络  | [marl/models.py](marl/models.py)                                                                                                                                                                           |
| 训练  | [marl/train_online0325.py](marl/train_online0325.py) 或新入口                                                                                                                                                  |
| 测试  | [tests/test_phase2_obs_generator.py](tests/test_phase2_obs_generator.py)、[tests/test_phase3_models.py](tests/test_phase3_models.py)、[tests/test_train_online_smoke.py](tests/test_train_online_smoke.py) 等 |
| 文档  | [README.md](README.md)、[docs/MARL_OVERVIEW.md](docs/MARL_OVERVIEW.md)                                                                                                                                      |


---

## 七、可选澄清（实施前若语义仍歧义）

- “友军负责敌方（分配的敌方）的 `A×3`” 若需与任务分配模块 **严格一致**（例如仅同一 Eid 的友机），可在实现阶段用 `pairs_realE2P` 过滤友机槽位；当前计划按 **几何相对特征** 实现，与用户 **第 6 条“保持原样”** 不冲突。

---

*本文件替代原「风险盘点」中的待办方向；执行时请用户确认「开始实施」后再改代码。*
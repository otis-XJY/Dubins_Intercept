# Chapter Plan + INSIGHT Collection

> **论文标题（暂定）**: Dynamic Intercept Point Selection for Multi-UAV Interception in Large-Scale Obstacle Environments via Transformer-based MARL
>
> **目标会议**: ICRA (IEEE International Conference on Robotics and Automation)
>
> **页数限制**: 6 页
>
> **生成日期**: 2026-05-23

---

## INSIGHT Collection

### INSIGHT-1: 大场景拦截的核心困难
大场景下敌方可执行 fake attack（急转弯、假动作）进行欺骗。盲目飞向敌方会导致从拦截态势变为追击态势（态势恶化），且浪费 UAV 有限的能耗。因此需要"逐步飞行、动态调整"的策略，而非一次性锁定最终拦截点。

### INSIGHT-2: 知识注入将连续控制转化为离散选择
直接输出连续控制指令（速度 v、角速度 ω）在大场景下质量差、容易陷入局部最优，且易被敌方 fake attack 欺骗——盲目追踪一个方向，敌方急转弯就从拦截变为追击。利用 Dubins 等时线的领域知识构造离散候选拦截点，将低质量的连续控制问题转化为高质量的离散选择问题，每个候选点都经过路径规划验证，确保物理可行性。

### INSIGHT-3: 动态候选点需要动态选择
候选拦截点不是静态的——敌方轨迹偏离时触发重规划，候选点集合随之变化（数量和位置都变）。因此选择策略必须能处理变维输入，Transformer 的注意力机制天然适合这种动态性。

### INSIGHT-4: Voronoi 拓扑瓶颈点的价值
Voronoi 图的顶点位于障碍物边界的几何瓶颈处，这些位置是 UAV 穿越障碍物密集区域的关键通道。将这些拓扑瓶颈点作为离散点的候选，比均匀网格或随机采样更高效。

### INSIGHT-5: CTDE 范式的适用性
多 UAV 拦截是协作任务（所有 pursuer 共享捕获奖励和资产损失惩罚），但执行时每架 UAV 只能基于自身观测决策。CTDE（集中训练、分散执行）完美匹配这一需求。

### INSIGHT-6: DWA 偏离检测作为重规划触发器
通过 DWA（Dynamic Window Approach）前向模拟敌方轨迹并与预规划轨迹比较，当偏差超过阈值时触发重规划。这使得系统能感知敌方的 fake attack 并动态调整，而非盲目执行初始计划。

---

## Chapter Plan

### I. Introduction（~1 页）

**核心叙事线**: 问题紧迫性 → 现有方法不足 → 我们的方案 → 贡献点

#### 1.1 问题背景与动机（~0.3 页）
- 大场景 UAV 拦截问题的实际意义（安防、边境巡逻、设施保护）
- 大场景的核心挑战：敌方可执行 fake attack 进行欺骗，盲目追踪导致态势恶化（从拦截变追击）
- UAV 能耗约束：不必要的飞行浪费有限续航
- 有障碍物环境增加复杂性

**[INSIGHT-1]** 引用

#### 1.2 现有方法的不足（~0.2 页）
- 传统等时线方法：适用于小场景、无障碍物环境，无法扩展到大场景
- 直接追踪方法：易被 fake attack 欺骗，导致态势恶化
- 纯 MARL 方法：在大场景连续空间中搜索效率低

#### 1.3 我们的方案（~0.2 页）
- 核心思想：Dubins 等时线知识构造候选点 + MARL 动态选择
- "逐步飞行、动态调整"策略，避免盲目追踪
- DWA 偏离检测 + 重规划机制应对 fake attack

#### 1.4 贡献点（~0.1 页）
1. 提出基于 Voronoi 拓扑瓶颈点 + Dubins A* IsoMap 的候选拦截点构造方法
2. 设计基于 Transformer 的 MARL 动态拦截点选择框架（CQN 架构），处理变维候选点
3. 在大场景有障碍物环境中验证方法的可扩展性、鲁棒性和 fake attack 抵抗能力

---

### II. Related Work（~0.5 页）

#### 2.1 传统 UAV 拦截方法（~0.15 页）
- 等时线方法：Isochron-based pursuit-evasion [引用用户提供的文献]
- 边界防御：Perimeter defense games [引用用户提供的文献]
- 微分博弈：TAD differential games [引用用户提供的文献]
- **不足**：局限于小场景、无障碍物、确定性环境

#### 2.2 MARL 多智能体协作（~0.15 页）
- **MAPPO**：Yu et al. 证明 PPO 在合作多智能体游戏中出人意料地有效 [Yu et al., NeurIPS 2022, arXiv:2103.01955]
- **注意力机制 + MARL**：Actor-Attention-Critic 使用注意力机制聚合多智能体信息 [Iqbal & Sha, ICML 2019, arXiv:1810.02912]
- **QMIX**：单调值函数分解实现集中训练分散执行 [Rashid et al., ICML 2018, arXiv:1803.11485]
- **MADDPG**：混合合作-竞争环境的多智能体 Actor-Critic [Lowe et al., NeurIPS 2017, arXiv:1706.02275]
- **多 UAV MARL**：Chen et al. 提出基于深度 RL 的多 UAV 追逃在线规划方法 [Chen et al., IEEE RA-L 2025]；Peng et al. 研究有限视场下的多 UAV 协同追击 [Peng et al., IEEE/CAA JAS 2025]
- **Transformer + MARL**：将合作 MARL 建模为序列生成问题 [Wen et al., NeurIPS 2022, arXiv:2205.14953]；Cai et al. 将 Transformer 注意力用于异构多机器人合作泛化 [Cai et al., IROS 2024]

#### 2.3 混合方法：传统规划 + 学习（~0.15 页）
- 图划分 + 多机器人协调：Voronoi 图用于多 UAV 编队包围和捕获 [用户提供的文献]
- Dubins 路径规划 + 拦截：利用 Dubins 路径约束实现最优拦截点选择 [用户提供的文献]
- **[MARK-红色] 此部分混合方法文献可能不足，用户可补充或弱化为一句话带过**

#### 2.4 区别总结（~0.05 页）
- 一句话总结：现有方法未考虑大场景 + fake attack + 有障碍物的综合挑战

---

### III. Problem Formulation（~0.7 页）

#### 3.1 场景描述（~0.2 页）
- 2D 有障碍物环境，包含被保护设施（assets）
- P 架 pursuer UAV，E 架 evader UAV
- Evader 沿预规划轨迹飞行，可执行 fake attack（轨迹偏离）
- Pursuer 需拦截 evader 防止其到达设施
- 约束：UAV 最小转弯半径（Dubins 约束）、碰撞避免

#### 3.2 Dubins 路径规划基础（~0.15 页）
- Dubins 车辆模型：前进速度恒定，受限转弯半径
- Dubins A* 算法：在有障碍物环境中搜索最短 Dubins 路径
- 等时线（Isochrone）：从起点出发在时间 t 能到达的所有位置集合

#### 3.3 Dec-POMDP 形式化（~0.35 页）
- **状态空间 S**: 所有 agent 位置、速度、航向，候选点集合，匹配关系
- **观测空间 O** (ego-centric, 8 路分支):
  - `self_uav` (3-dim): 自身位置 (x, y) 和航向 θ
  - `allies_local` (3-dim per ally): 友方相对位置和航向
  - `self_pts` (8-dim per candidate): 候选拦截点特征（x, y, t_p, t_e, Δt, cost, reserved, reserved）
  - `ally_pts` (8-dim per candidate): 友方的候选点特征
  - `enemy_assigned_self` (3-dim): 分配敌方的位置和航向
  - `enemy_assigned_per_ally` (3-dim per ally): 友方分配敌方的信息
  - `asset_target_self` (2-dim): 当前目标设施位置
  - `asset_target_per_ally` (2-dim per ally): 友方目标设施位置
  - 掩码：`self_pts_mask`, `ally_mask`, `enemy_self_mask`, `ally_enemy_mask`（标记有效元素）
- **动作空间 A**: 离散索引，选择候选拦截点（K 维，K 随重规划动态变化）
- **奖励函数 R**: 步级奖励（距离进度、全局分散、安全距离、时间代价）+ 终端奖励（捕获奖励、资产损失惩罚）
- **目标**: 最大化期望累积奖励 = 最大化捕获率 + 最小化资产损失 + 最小化飞行距离

---

### IV. Proposed Method（~2.3 页）

#### 4.1 Overall Framework（~0.5 页）

**[需要一张系统框架图，展示完整流程]**

```
地图 → Voronoi 拓扑瓶颈点 + 覆盖采样 → 离散点集 (Trans_Point)
                                              ↓
                        预计算：Dubins A* IsoMap 时间场
                                              ↓
运行时：当前状态 → 候选拦截点构造 → MARL 选择 → 执行飞行
                ↑                              ↓
                ←── DWA 偏离检测 ←── 轨迹推进 ←┘
                （偏离 > 阈值 → 重规划）
```

- 动态重规划机制：DWA 基于 unicycle 模型，用多种 (v, ω) 组合前向模拟敌方可能的轨迹，与算法预测的敌方路径（来自逃逸者轨迹数据库）比较，选择最匹配的模拟轨迹。若最匹配轨迹与预测路径的偏差超过阈值（距离单位 > 50），说明敌方偏离了预期行为（fake attack），触发重规划
- 重规划范围：从 IsoMap 中重新选点 + 匈牙利重新分配（不重建 Voronoi 和 IsoMap，因为离散点集和时间场是预计算的）

**[INSIGHT-3, INSIGHT-6]** 引用

#### 4.2 Candidate Intercept Point Construction（~0.6 页）

**4.2.1 Topology-aware Discrete Point Sampling**
- Voronoi 图以障碍物边界采样点 + 地图角点 + 中心点为种子
- Voronoi 顶点中位于障碍物安全缓冲区外的作为拓扑瓶颈候选
- KMeans 聚类为 `n_topo` 个瓶颈节点
- 额外 `n_blank` 个覆盖采样点（拒绝采样，有最小间距约束）
- 最终离散点集 = 拓扑瓶颈点 + 覆盖点

**[INSIGHT-4]** 引用

**4.2.2 IsoMap Time Field Construction**
- 对每个 (离散点, 起点) 对，运行 Dubins A* 得到无障碍物路径
- 沿路径按时间采样，构建等时线位置序列
- 运行时：从 pursuer 当前位置规划到最近等时线点，拼接预计算路径

**4.2.3 Candidate Evaluation**
- 对每个 (evader, pursuer, 候选点) 组合，计算 5 项代价：
  - 路径长度、空间距离、时间匹配度、航向偏差、目标安全性
- 匈牙利匹配确定 pursuer-evader 分配

#### 4.3 MARL-based Intercept Point Selection（~0.9 页）

**4.3.1 CTDE Paradigm**
- 参数共享：所有 pursuer 共享一个网络，batch 维度 = P
- 集中式 Critic：输入所有 agent 的联合特征，输出每个 agent 的 V 值
- 分散式 Actor：每架 pursuer 基于 ego-centric 观测独立决策

**[INSIGHT-5]** 引用

**4.3.2 Transformer-based Policy Network (CQN)**
**[需要一张网络架构图]**

- 候选点编码器：8 维特征 → MLP → d 维嵌入
- 上下文编码：self_uav、allies、enemies 分别编码
- Cross-attention：候选点作为 query，上下文作为 key/value
- 输出：每个候选点的 logit，通过 softmax + mask 得到动作概率

**4.3.3 Ego-centric Observation Construction**
- 每架 pursuer 以自身为参考系构建 8 路分支观测（TODCEmbeddings 编码）：
  - `self_uav` (3-dim): 自身 (x, y, θ)
  - `allies_local` (3-dim × (P-1)): 友方相对位置和航向
  - `self_pts` (8-dim × K): 候选点特征——(x, y, t_p, t_e, Δt, cost, reserved, reserved)
  - `ally_pts` (8-dim × (P-1)×K): 友方候选点特征
  - `enemy_assigned_self` (3-dim): 分配敌方 (x, y, θ)
  - `enemy_assigned_per_ally` (3-dim × (P-1)): 友方分配敌方
  - `asset_target_self` (2-dim): 目标设施位置
  - `asset_target_per_ally` (2-dim × (P-1)): 友方目标设施
- 每路通过独立 MLP 编码为 hidden_dim 维嵌入，再通过 HeterogeneousAttentionAB 做跨路注意力融合
- 动态 K：候选数随重规划变化，通过 self_pts_mask 处理变维

**4.3.4 Reward Function**
- 步级奖励（每步累积）：
  - r_progress: 距离进度（与分配 evader 的距离减小，权重 dist_progress_scale=0.01）
  - r_global: 全局分散/分配惩罚（鼓励 pursuer 分散覆盖不同 evader，assign_penalty=4.0, div_sigma=300）
  - r_safe: 安全距离惩罚（pursuer 间距离 < safe_dist_min=120 时触发，safe_penalty_scale=0.1）
  - r_time: 时间代价（step_cost=-0.1，鼓励快速拦截）
- 终端奖励（episode 结束时）：
  - 捕获奖励（terminal_capture_bonus=30，按 captured_delta/num_E 缩放后加到每个 pursuer）
  - 资产损失惩罚（terminal_asset_loss_penalty=50，全员相同惩罚）
- 5 项候选质量加权：α_dist_v=1.0, β_delta_t=1.0, γ_delta_d=1.0, γ_delta_theta=1.0, λ_path_l=1.0
- **[INSIGHT-1]** 引用：捕获奖励和资产损失惩罚是最重要的项，具体的权重值通过 YAML 配置调优

#### 4.4 Training with MAPPO（~0.3 页）
- GAE(λ) 估计优势函数
- PPO clip 更新策略
- 多 scheme 并行训练（CQN / GQN / PWSN，论文以 CQN 为主）
- 超参数（from train_online0324.yaml）：
  - γ=0.99, λ=0.95, lr=3e-4 (线性衰减到 10%)
  - PPO clip=0.2, epochs=4, minibatch=32
  - entropy_coef=0.02, value_coef=0.5
  - hidden_dim=128, num_heads=4
  - kl_early_stop=0.03, advantage_clip=5.0
- 奖励权重：terminal_capture_bonus=30, terminal_asset_loss_penalty=50, step_cost=-0.1

---

### V. Experiments（~1.2 页）

#### 5.1 Experimental Setup（~0.2 页）

**场景参数（[MARK-红色] 用变量 X 表示，待确认）：**
- 环境：2D 有障碍物地图，尺寸 X × X（默认 2000×2000 距离单位）
- 障碍物：X 个多边形随机障碍物，半径范围 [X, X]，安全预留距离 X
- 被保护设施数量：X 个
- UAV 参数：v_P = v_E = X m/s，最小转弯半径 r = X

**训练数据构造流程（离线预计算）：**
1. **地图生成**（mainObtainMap.py）：Voronoi 拓扑瓶颈点提取 + 覆盖采样 → 离散点集 (Trans_Point)；保存为 joblib 资源
2. **IsoMap 预计算**（mainObtainIsoMap0901.py）：对每个 (起点, 离散点) 对运行 Dubins A* 构建时间场；生成 3 组 IsoMap：Pursuer→Trans_Point、Evader→Trans_Point、Trans_Point 间
3. **逃逸者轨迹**（mainObtainPathETrue.py）：人工指定或交互式选取 evader 的中间路径点，生成参考逃逸轨迹

**训练细节：**
- 30000 episodes，种子 42
- MAPPO 超参见 §4.4
- 每 10 episode 做一次贪心评估

#### 5.2 Comparison Experiments（~0.3 页）

| 方法 | 拦截点选择 |
|------|-----------|
| Ours (MARL) | Transformer-based MARL 选择 |
| Cost-based | 5 项代价最优的候选点 |
| Greedy | 距离最近的候选点 |
| Random | 随机选择 |
| Vanilla MAPPO | 无 IsoMap 知识的标准 MAPPO |

**表格**: 所有方法 × 3-5 场景 × 4 指标（捕获率、拦截时间、路径长度、资产损失率）

**关键发现预期**: MARL > Cost-based > Greedy > Random；Ours > Vanilla MAPPO（知识注入的价值）

#### 5.3 Scalability Experiment（~0.15 页）
- 4 种规模：3v5, 5v8, 8v12, 12v20
- 折线图：捕获率 vs agent 数量，拦截时间 vs agent 数量
- **[INSIGHT-3]** 引用：Transformer 处理变维输入的能力

#### 5.4 Robustness to Fake Attack（~0.2 页）
- Normal: 敌方按预规划参考轨迹飞行
- Fake Attack: 敌方偏离参考轨迹（急转弯、假动作），DWA 检测偏离后触发重规划
- Fake + No Replan: 敌方同样偏离，但系统不触发重规划（固定初始计划）

**表格/柱状图**: 3 条件 × 4 指标

**关键发现预期**: Ours (with DWA replan) >> Fake + No Replan，验证 DWA 偏离检测 + 重规划机制的必要性

#### 5.5 Environment Complexity（~0.15 页）
- 无障碍物、稀疏障碍物、密集障碍物
- **表格**: 3 条件 × 4 指标

#### 5.6 Ablation Study（~0.2 页）

| 变体 | 去掉什么 |
|------|---------|
| Full (Ours) | 无 |
| w/o Transformer | MLP 替代 |
| w/o IsoMap 知识 | 随机采样点 |
| w/o Voronoi 拓扑点 | 均匀网格 |
| w/o DWA 重规划 | 固定初始计划 |

**表格**: 5 变体 × 4 指标

**关键发现预期**: 完整方法优于所有变体；w/o DWA 重规划在 fake attack 场景下降最明显

#### 5.7 Qualitative Results（~0.1 页）
- 轨迹可视化：展示 pursuer 如何"逐步逼近"而非"盲目追踪"
- 重规划时刻可视化：展示 DWA 触发重规划的过程

---

### VI. Conclusion（~0.3 页）

- 总结：提出大场景有障碍物环境下的 UAV 拦截方法
- 核心贡献回顾：Voronoi + Dubins IsoMap 知识注入 → Transformer-based MARL 动态选择 → DWA 偏离检测 + 重规划
- 实验验证：可扩展性、鲁棒性、fake attack 抵抗能力
- **[MARK-红色] 未来工作（根据文献调研撰写，待用户确认）：**
  - **3D 场景扩展**：当前方法基于 2D 平面假设，扩展到 3D 空间需要考虑高度维度的 Dubins 约束和垂直面内的等时线构建 [参考: Chen et al., RA-L 2025 中的 3D 追逃讨论]
  - **对抗性 evader**：当前 evader 由规则驱动，未来可引入可学习的对抗策略（self-play 或 population-based training），使 pursuer 策略更具鲁棒性 [参考: Lowe et al., NeurIPS 2017 的混合合作-竞争框架]
  - **Sim-to-real 迁移**：在 Gazebo/ROS 仿真器中验证后，部署到真实四旋翼 UAV 平台，处理传感器噪声、通信延迟和动力学不确定性 [参考: Cai et al., IROS 2024 的多机器人 sim-to-real 讨论]
  - **通信约束下的分布式执行**：当前 CTDE 假设训练时可获取全局信息，未来可研究有限通信带宽下的分布式策略 [参考: Peng et al., IEEE/CAA JAS 2025 的有限视场研究]
  - **动态障碍物和多类型威胁**：当前障碍物为静态，扩展到动态障碍物（如移动禁区）和多类型威胁（如电子干扰区域）可增强实用性

---

## 待用户补充/确认的事项

1. **[已完成]** Related Work 中 MARL 文献已补充完整
2. **[MARK-红色]** "Topology-aware Discrete Point Sampling" 命名，用户确认或修改
3. **[MARK-红色]** "Trajectory Deviation Detection" 命名，用户确认或修改
4. **[已完成]** 训练超参数已从 YAML 填入
5. **[MARK-红色]** 地图具体尺寸和 UAV 参数用变量 X 表示，待用户确认
6. **[MARK-红色]** 未来工作方向 5 条已撰写，待用户确认/增删
7. 实验结果待训练完成后补充
8. 网络架构图和系统框架图需要绘制

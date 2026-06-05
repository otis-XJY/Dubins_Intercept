# NetworkDesign

## 1. 文档目的

本文档给出当前代码库中 MARL 决策网络的**可复现实现说明**，用于论文撰写与工程对齐。内容覆盖：

- 网络输入与符号定义
- 异构并行注意力主干
- 四种 actor 设计（A/B/C/D）
- 置换不变 centralized critic
- 在 MAPPO 训练中的作用与更新流程
- 可解释性输出项

对应实现路径：

- `marl/model_blocks/embeddings.py`
- `marl/model_blocks/attention.py`
- `marl/model_blocks/fusion.py`
- `marl/model_blocks/actor.py`
- `marl/model_blocks/critic.py`
- `marl/model_blocks/network.py`
- `marl/nn/models.py`
- `marl/runners/online_train.py`
- `marl/rl/mappo.py`

> 备注：`marl/nn/blocks/network.py` 仅做转发导入，真实网络实现在 `marl/model_blocks/network.py`。

---

## 2. 问题定义与符号

设：

- 追捕者数量：$P$
- 逃逸者数量：$E$
- 单机候选拦截点数量：$N$（即动态 $k_{\max}$）
- 单机友机槽位：$A=P-1$
- 隐空间维度：$D$

策略在每个决策步输出离散动作（候选点索引）：

$$
\pi_\theta(a_i \mid o_i),\quad a_i \in \{0,1,\dots,N-1\}
$$

其中每次 `env.step()` 是一个 RL 决策步；内部可能包含多个物理推进 tick。

---

## 3. 观测输入与张量组织

观测由 `TODCObservationGenerator` 生成（`marl/obs/generator.py`），训练时由 `_build_model_obs` 转成 Tensor（`online_train.py`）。核心键与形状：

- `self_uav`: $(P,1,3)$
- `allies_local`: $(P,A,3)$
- `enemy_assigned_self`: $(P,1,3)$
- `enemy_assigned_per_ally`: $(P,A,3)$
- `asset_target_self`: $(P,1,2)$
- `asset_target_per_ally`: $(P,A,2)$
- `self_pts`: $(P,N,8)$
- `ally_pts`: $(P,A\cdot N,8)$
- 掩码：`ally_mask`、`enemy_self_mask`、`ally_enemy_mask`、`self_pts_mask`、`ally_pts_mask` 等

其中 `self_pts_mask` 直接决定动作可行性；无可行候选时执行退化策略（固定概率落到第 0 位，输出动作为 -1 表示 inactive）。

---

## 4. 步骤一：独立编码（Independent Embedding）

实现：`TODCEmbeddings`（`embeddings.py`）。

### 4.1 Ego-centric 坐标归一化

所有实体先转换到本机参考系：

1. 平移到本机原点
2. 旋转对齐本机航向到 $+x$
3. 距离量按 `pos_scale` 缩放
4. 角度使用 $\sin/\cos$ 或自动弧度转换，避免跳变

### 4.2 分类型编码器

不同实体使用独立编码器（含结构化候选编码）：

- 本机：使用可学习 `ego token` 作为 query token
- 友机：`enc_ally`
- 敌机：`enc_enemy`
- 资产/目标：`enc_asset`
- 候选点：`StructuredCandidateEncoder`

候选点 8 维特征经转换后按“几何分支 + 质量分支”编码：

- 几何（5维）：相对位置/距离/相对朝向
- 质量（6维）：$\Delta t$、$\Delta d$、$\Delta \theta$、$path_L$、$\Delta V$ 等经 `asinh/log1p/signed-log` 压缩

`self_pts` 与 `ally_pts` 共享候选编码器权重，通过 role embedding 区分来源。

---

## 5. 步骤二：异构并行注意力分支（Heterogeneous Attention Streams）

实现：`HeterogeneousAttentionAB`（`attention.py`）。

以 $e_{self}$ 为 query，向 7 路上下文并行发起 cross-attention（每路独立参数）：

1. allies
2. self points
3. ally points
4. enemy assigned self
5. enemy assigned per ally
6. asset target self
7. asset target per ally

形式上每路可写为：

$$
\mathbf{h}_b = \mathrm{Attn}_b(\mathbf{q}_b, \mathbf{K}_b, \mathbf{V}_b),\quad b\in\mathcal{B},\ |\mathcal{B}|=7
$$

并返回可解释权重 `attn_weights`。

> 与“三分支 ally/enemy/points”思想一致，但当前实现做了更细粒度拆分（7 路），语义解耦更强。

---

## 6. 步骤三：多源融合与 Actor 设计

统一包装类：`UAVInterceptionNetwork`（`network.py`）。

### 6.1 设计 A：Concatenative Query Network

实现：`ConcatMLPFusion8` + `PointerActor`。

$$
\mathbf{h}_{env} = \mathrm{MLP}_{fusion}\left([\mathbf{e}_{self}\parallel\mathbf{h}_1\parallel\cdots\parallel\mathbf{h}_7]\right)
$$

再由 pointer 打分：

$$
\mathbf{q}_{act}=\mathrm{MLP}_{actor}(\mathbf{h}_{env}),\quad
\text{logits}=\frac{\mathbf{q}_{act}(W_K\mathbf{E}_{self\_pts})^T}{\sqrt{D}}
$$

### 6.2 设计 B：Gated Query Network

实现：`SoftGatingFusion7` + `PointerActor`。

$$
\alpha_b = \mathrm{softmax}\left(\frac{\mathbf{e}_{self}^T\mathbf{h}_b}{\sqrt{D}}\right),\quad
\mathbf{h}_{env}=\mathrm{LN}\left(\mathbf{e}_{self}+\sum_b\alpha_b\mathbf{h}_b\right)
$$

随后走同样 pointer 动作头。输出 `gate_weights` 具备可解释性。

### 6.3 设计 C：Point-Wise Scoring Network

实现：`PointsContextCrossAttention` + `PointwiseScoringActor`。

该设计对应 Independent Action Evaluation：

- Query：每个候选点 $\mathbf{E}_{points}$
- Key/Value：环境上下文池（self/ally/enemy/asset 等）
- 不做候选点之间 self-attention

逐点评分：

$$
\tilde{\mathbf{h}}_n = \mathrm{CrossAttn}(\mathbf{e}_{point,n},\mathbf{E}_{context})
$$
$$
\text{logit}_n = \mathrm{MLP}_{score}(\tilde{\mathbf{h}}_n)
$$

输出 `points_ctx_attn_weights`。

### 6.4 设计 D：Two-Stage Residual Network（实现细节）

实现入口：`UAVInterceptionNetwork._actor_forward_d`（`marl/model_blocks/network.py`），由三部分组成：

- 阶段1打分器：`SelfStageScorer`
- 阶段2协同注意力：`AllyCoordCrossAttention`（ISAB 风格候选压缩 + cross-attn）
- 阶段2残差头：`CoordResidualHead`

其核心思想是：先得到“只看自身局部信息”的候选点基线分数，再引入友军协同信息做加性残差修正。

#### 6.4.1 阶段0：构造自身上下文 `self_ctx`

虽然设计名为“两阶段”，但代码中先有一个轻量上下文融合，用于给阶段1提供条件信息：

$$
\mathbf{c}_{self}=\mathrm{LN}\left(\mathrm{GELU}\left(W\,[\mathbf{e}_{self}\parallel\mathbf{e}_{e\_self}\parallel\mathbf{e}_{ast\_self}]\right)\right)
$$

其中：

- $\mathbf{e}_{self}\in\mathbb{R}^{P\times D}$：本机 token
- $\mathbf{e}_{e\_self}\in\mathbb{R}^{P\times D}$：本机当前分配敌机编码
- $\mathbf{e}_{ast\_self}\in\mathbb{R}^{P\times D}$：该敌机对应资产目标编码

实现为 `self.self_ctx_fuse = Linear(3D,D) + GELU + LayerNorm`，输出形状为 $(P,D)$。

#### 6.4.2 阶段1：自身主导的候选点评分（`SelfStageScorer`）

阶段1只用 `self_ctx` 与本机候选点编码 `e_self_pts` 逐点打分，不引入友军协同：

$$
\text{logit}^{(1)}_{i,n}=f_{self}\big([\mathbf{e}_{self\_pts,i,n}\parallel\mathbf{c}_{self,i}]\big)
$$

其中：

- $\mathbf{e}_{self\_pts}\in\mathbb{R}^{P\times N\times D}$
- 先把 $\mathbf{c}_{self,i}$ broadcast 到 $N$ 个候选点，再做拼接
- `SelfStageScorer` 结构：`Linear(2D,H) -> GELU -> Linear(H,1)`（默认 $H=D$）

输出：

$$
\text{logits}_{self}\in\mathbb{R}^{P\times N}
$$

物理语义：给出“单机视角下，这个点值不值得去”的基线判断。

#### 6.4.3 阶段2：友军协同残差修正（`AllyCoordCrossAttention` + `CoordResidualHead`）

阶段2让每个本机候选点去查询“友军协同上下文池”，获得协同特征后输出残差：

1) 先对 `ally_pts` 做 ISAB 风格压缩：

- 先将展平的 `e_ally_pts`（形状 $(P,A\cdot N,D)$）按友机槽位还原为 $(P,A,N,D)$。
- 对每个友机槽位引入 $M$ 个 inducing token，执行一次 cross-attn：

$$
\mathbf{E}^{compact}_{ally\_pts}=\mathrm{Attn}(Q=\mathbf{I}_{M},K=\mathbf{E}_{ally\_pts},V=\mathbf{E}_{ally\_pts})
$$

压缩后形状为 $(P,A\cdot M,D)$，其中当前实现默认 $M=4$。

2) 协同上下文池（K/V）改为紧凑拼接：

$$
\mathbf{E}_{coord}=[\mathbf{E}_{ally}\parallel\mathbf{E}^{compact}_{ally\_pts}\parallel\mathbf{E}_{e\_ally}\parallel\mathbf{E}_{ast\_ally}]
$$

对应形状为 $(P, A + A\cdot M + A + A, D)$，不再随 $N$ 线性膨胀。

3) 以本机候选点为 Query 做 cross-attention：

$$
\mathbf{C}_{coord}=\mathrm{Attn}(Q=\mathbf{E}_{self\_pts},K=\mathbf{E}_{coord},V=\mathbf{E}_{coord})
$$

输出 $\mathbf{C}_{coord}\in\mathbb{R}^{P\times N\times D}$，并返回 `coord_attn_weights` 供解释分析。

4) 残差头逐点输出：

$$
\Delta_{i,n}=f_{res}\big([\mathbf{e}_{self\_pts,i,n}\parallel\mathbf{c}_{coord,i,n}]\big)
$$

`CoordResidualHead` 结构：`Linear(2D,H) -> GELU -> Linear(H,1)`。

最终融合：

$$
\text{logit}_{i,n}=\text{logit}^{(1)}_{i,n}+\Delta_{i,n}
$$

关键实现细节：

- `CoordResidualHead` 最后一层权重与偏置 **零初始化**，使训练初期 $\Delta\approx0$。
- 这意味着初始策略近似退化为阶段1纯自身策略，协同修正从 0 平滑学习，减少训练早期不稳定。

#### 6.4.4 Mask 与概率输出（两阶段共享后处理）

两阶段得到的 `logits` 统一交给 `postprocess_mask_and_sample`：

- 非法候选（`self_pts_mask=False`）置为 `-1e9`
- softmax 得 `action_probs`
- 若某个体无任何合法候选：该体概率退化为第0位=1，`best_candidate_idx=-1`

因此设计 D 对动态动作空间和失活体有完整兼容。

#### 6.4.5 与训练/解释性的直接关系

- 训练时 `actor_forward` 直接走 `_actor_forward_d`，PPO 梯度同时更新阶段1与阶段2参数。
- 可解释性上，阶段1提供“自决策基线”，阶段2的 `coord_attn_weights` 提供“协同修正来源”。
- 对论文叙述可写为：**先单机可行性评估，再多机协同残差对齐**。

`CoordResidualHead` 的零初始化，使“基线策略 + 残差策略”具备清晰可解释的学习路径。
---

## 7. 步骤四：动作概率与掩码

统一后处理：`postprocess_mask_and_sample`（`actor.py`）。

- 对非法候选位置加 `-1e9`
- softmax 得到 `action_probs`
- 训练时用 `Categorical(probs)` 采样
- 推理可用 `argmax`

概率形式：

$$
\pi(a\mid s)=\mathrm{Softmax}(\text{logits}+\text{mask})
$$

---

## 8. 集中式 Critic（CTDE）

实现：`PermInvariantCritic`（`critic.py`）。

输入：

- 每机环境表征：$h_{env,i}\in\mathbb{R}^{D}$
- 全局表征：$h_{global}\in\mathbb{R}^{D}$（由 `_GlobalStateEncoder` 生成）

先做跨机 attention pooling（对 agent 顺序置换不变）：

$$
\beta_i = \mathrm{softmax}(s_i),\quad
\mathbf{g}=\sum_i\beta_i W_v h_{env,i}
$$

再输出每机价值：

$$
V_i = f_v([h_{env,i}\parallel \mathbf{g}\parallel h_{global}])
$$

该结构避免了旧版 concat-over-P 动态重建 critic 的参数管理问题，DDP 友好。

---

## 9. 在训练中的作用机制（MAPPO）

入口：`train_online`（`marl/runners/online_train.py`）。

每个 scheme（如当前配置中的 C 与 D）独立维护：

- `env`
- `model`
- `optimizer`
- `scheduler`

单回合训练流程：

1. rollout 期间每个决策步：
   - `actor_forward(obs_t)` -> `action_probs`
   - 采样动作并执行 `env.step(action)`
   - `critic_forward(obs_t)` 得到 $V_t$
   - 缓存 $(o_t,a_t,\log\pi_t,r_t,V_t,d_t)$

2. 回合结束后：
   - `compute_gae` 计算优势与回报
   - 优势标准化并可选裁剪
   - `ppo_minibatch_update` 执行多轮 PPO 更新

GAE：

$$
\delta_t = r_t + \gamma V_{t+1}(1-d_t)-V_t
$$
$$
A_t = \delta_t + \gamma\lambda(1-d_t)A_{t+1},\quad R_t=A_t+V_t
$$

PPO 目标（含 value clipping 与 entropy 正则）：

$$
L_\pi = -\mathbb{E}[\min(r_tA_t,\mathrm{clip}(r_t,1\pm\epsilon)A_t)]
$$
$$
L_V = \max((V_t-R_t)^2,(V_t^{clip}-R_t)^2)
$$
$$
L = L_\pi + c_vL_V - c_eH
$$

训练稳定性策略：

- KL early stopping
- gradient clip
- learning rate linear decay

---

## 10. 可解释性与论文可视化建议

当前实现可直接导出以下解释量：

- `attn_weights`（A/B/C 主干异构注意力）
- `gate_weights`（B 的动态门控）
- `points_ctx_attn_weights`（C 的候选点对上下文关注）
- `coord_attn_weights`（D 的协同残差来源）

建议在论文中展示：

1. 不同战术阶段的注意力迁移热图
2. A/B/C/D 在样本效率和收敛速度上的对比曲线
3. C 与 D 在“单点评估 vs 协同修正”上的行为差异案例

---

## 11. 当前配置结论（与训练脚本一致）

根据 `configs/train_online0324.yaml`，当前并行训练方案为：

- `Point-Wise Scoring Network`（设计 C）
- `Two-Stage Residual Network`（设计 D）

即：当前主实验不是单一 pointer 架构，而是 **C + D 双方案并行评估/训练**。

---

## 12. 一句话总结

本项目当前网络可概括为：

**Ego-centric 结构化编码 + 细粒度异构并行注意力 + 多方案 actor（A/B/C/D）+ 置换不变 centralized critic + MAPPO 更新闭环**。

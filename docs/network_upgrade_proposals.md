# 策略网络升级方案与建议

本文档汇总拦截策略网络（`marl/model_blocks/`）的历史改进方案与外部建议，供后续迭代与消融实验参考。

**生效代码路径**：`marl/model_blocks/`（`marl/nn/blocks/network.py` 仅转发）。`marl/nn/blocks/` 下同名文件为旧副本，勿以其为准。

**相关配置**：`configs/train_online0324.yaml` 中 `hidden_dim`、`pos_scale`、`schemes` 等。

**分阶段执行记录**：[`.cursor/plans/tier123_network_upgrade_840bb206.plan.md`](../.cursor/plans/tier123_network_upgrade_840bb206.plan.md)

**图例**：✅ 已完成 · ⚠️ 部分完成 · ❌ 未开始

---

## 实施摘要（Tier 0–2 + C/D 对齐，2026-06）

> **说明**：下列改动已在工作区落地；历史改动曾通过 `pytest tests/test_phase3_models.py`（13 项全绿）。本轮新增修改已完成 `read_lints` 与 `python3 -m py_compile` 语法检查；**尚未 git commit**。`wandb/run-20260529_001423-lxs7dmoh` 启动于 commit `d0f285f`，**不含**本次任何改动（无 `pos_scale`、无相对坐标、旧 `CentralCritic`），不能用于验证新网络。

### 已完成

| 阶段 | 内容 | 主要文件 | 适用范围 |
|------|------|----------|----------|
| **阶段 0** | `PermInvariantCritic` 静态构建，替换懒加载 `CentralCritic`；修复 critic 参数未被优化器跟踪的 bug | `critic.py`, `network.py` | A/B/C/D 共用 |
| **Tier 1** | Ego-centric 相对坐标：位姿 3→5、点 2→3；候选内部编码输入从 8 扩展为 几何5+质量6（11 维） | `embeddings.py` | A/B/C/D 共用 |
| **Tier 1** | `StructuredCandidateEncoder`：几何 5 维 + 质量 6 维双分支 MLP → 投影 D（方案 2） | `embeddings.py` | A/B/C/D 共用 |
| **Tier 1** | 共享 `enc_cand` + `role_self_pts` / `role_ally_pts` 区分自/友候选（建议 2 部分） | `embeddings.py` | A/B/C/D 共用 |
| **Tier 1** | Embedding 内 `_mlp` 激活 ReLU→GELU（方案 13 部分） | `embeddings.py` | A/B/C/D 共用 |
| **Tier 1** | 质量特征无参尺度改造：`asinh/log1p/signed_log1p/sin-cos` + 自动度弧兼容，减少长尾与方向不一致 | `embeddings.py` | A/B/C/D 共用 |
| **Tier 1** | `pos_scale` 透传：YAML → `TrainConfig` → `build_actor_critic_schemes` → `UAVInterceptionNetwork` | `online_train.py`, `models.py`, `train_online0324.yaml` | 训练全局 |
| **Tier 2** | **设计 D**（Two-Stage Residual Network）：`SelfStageScorer` + `AllyCoordCrossAttention` + `CoordResidualHead`（zero-init） | `actor.py`, `attention.py`, `network.py`, `schemes.py` | 仅 scheme D |
| **Tier 2** | D 的 critic 与 C 对齐：统一走 `attn_ab` + `ConcatMLPFusion8`，移除廉价 mean-pool 路径 | `network.py` | 设计 C/D |
| **Tier 2** | 前向缓存优化：同一 `obs` 下复用 embedding/shared_ctx，减少 actor→critic 重复计算 | `network.py` | 设计 C/D |
| **训练配置** | 默认 `schemes` 改为 C + D（A/B 保留注释兼容） | `train_online0324.yaml` | 训练全局 |
| **验证** | 历史模型测试 `test_design_d_*`、`test_critic_params_tracked_by_optimizer`；本轮执行 lints + py_compile | `tests/test_phase3_models.py`, `network.py`, `embeddings.py` | — |

### 未完成（下一批建议）

| 优先级 | 内容 | 对应编号 |
|--------|------|----------|
| **高** | 用新代码重训并对比旧 run 的 `grad_norm` / `explained_variance` / `episode_entropy` | — |
| **中** | 统一 token + 单层 Transformer encoder，替换 7 路 star cross-attention | 建议 2（完整）、方案 10 |
| **中** | Attention 块 Pre-LN + 残差 FFN；全局 ReLU→GELU | 建议 5、方案 13（完整） |
| **中** | `hidden_dim` 128→256 消融 | 方案 12 |
| **低** | LearnableFeatureScale 或 RunningMeanStd（仅在无参压缩不足时再引入） | 方案 3（完整）、建议 5（完整） |
| **低** | 相对位置编码 / ALiBi / FiLM / Enhanced Embedding | 方案 6–11 |
| **低** | Critic 全局候选统计、attention 全局聚合 | 方案 14（完整） |
| **低** | 在 `generator.py` 侧统一 ego-centric（当前仅在 embedding 前向变换） | 建议 1（完整） |

### 架构现状（一句话）

主体仍是 **分散 MLP 编码 + 7 路 star cross-attention + A/B/C/D 四套 actor 头**；**Embedding 已完成无参尺度改造（质量 6 维）**；**D 的 critic 已与 C 对齐并加入前向缓存优化**；统一 Transformer / Pre-LN block 仍未动。

---

## 一、Embedding 层改进（方案 1–6）

### 方案 1：候选点分组结构化 Embedding ❌

> **落地状态**：未实施。当前采用方案 2 的双分支（几何/质量），非四组（空间/时间/几何/路径）。

将 `self_pts` 的 8 维特征按语义分组，独立编码后融合：

| 子空间 | 特征 | 输出维度 |
|--------|------|----------|
| 空间 | `c_x`, `c_y`, `theta` | D//2 |
| 时间 | `delta_t` | D//8 或 D//4 |
| 几何 | `delta_d`, `delta_theta` | D//4 |
| 路径 | `path_l`, `distance_V` | D//4 |

最后 fuse 拼接投影到 D 维。

**优势**：每个子网络只处理 1–3 维输入，不需要同时学习不同尺度/语义的映射。

---

### 方案 2：位置-质量双分支 Embedding（修正版） ✅

> **落地状态**：已完成并增强。`marl/model_blocks/embeddings.py` → `StructuredCandidateEncoder`：几何 5 维 `[rel_x,rel_y,r,sinΔθ,cosΔθ]` + 质量 6 维 `[asinh(Δt),-log1p(Δd),sin(Δθ),cos(Δθ),-log1p(path_L),signed_log1p(ΔV)]`，各过子 MLP 到 `D//2`，concat 投影到 D。对 A/B/C/D 全部生效。

基于特征物理含义，按「位置 vs 质量」分组：

| 分支 | 特征 | 输出维度 | 语义 |
|------|------|----------|------|
| 位置 | `c_x`, `c_y`, `theta` | D//2 | 候选点在哪 |
| 质量 | `asinh(delta_t)`, `-log1p(delta_d)`, `sin/cos(delta_theta)`, `-log1p(path_l)`, `signed_log1p(distance_V)` | D//2 | 拦截计划有多好 |

两分支输出在 D 维上融合。

**优势**：位置特征（绝对坐标 O(10³)）不会淹没质量特征（相对值 O(0–π)）。

---

### 方案 3：可学习特征尺度归一化（LearnableFeatureScale） ⚠️

> **落地状态**：部分完成。未实现 `LearnableFeatureScale` 参数化缩放；当前采用 `pos_scale` + 无参压缩（`asinh/log1p/signed_log1p/sin-cos`）处理长尾与量纲差异。

```python
class LearnableFeatureScale(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.scale + self.bias
```

- 对不同尺度特征做可学习的 per-feature 缩放和偏移
- 建议用在 quality 分支入口，让网络自己学习 `distance_V` 的方向（通过 scale 的正负）

---

### 方案 4：Quality Feature 方向归一化 ⚠️

> **落地状态**：部分完成。`Δd/path_L` 已通过 `-log1p` 统一为“越大越好”；`Δθ` 改为 `sin/cos`；`ΔV` 用 `signed_log1p` 压缩并保留方向。尚未引入显式可学习方向参数。

质量特征「好」的方向不一致，需统一为「越大越好」：

| 特征 | `delta_t` | `delta_d` | `delta_theta` | `path_l` | `distance_V` |
|------|-----------|-----------|---------------|----------|--------------|
| 原始好方向 | 大 | 小 | 小 | 小 | 大 |
| 翻转后 | 大 | 大 | 大 | 大 | 大 |

```python
# 固定可学习初值示例
self.scale = nn.Parameter(torch.tensor([1.0, -1.0, -1.0, -1.0, 1.0]))
```

---

### 方案 5：Embedding 后统一 RMSNorm ⚠️

> **落地状态**：部分完成。各 token 流独立 `LayerNorm`（`norm_self`…`norm_asset`），非统一 `RMSNorm` 模块。

```python
class UnifiedEmbeddingNorm(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, *embeddings):
        return tuple(self.norm(e) for e in embeddings)
```

确保所有 embedding 输出方差一致，使 attention 的 Q·K dot product 数值稳定。

---

### 方案 6：Enhanced Embedding（更深 + 残差） ❌

> **落地状态**：未实施。Embedding 仍为 2 层 MLP + GELU，无残差连接。

```python
class EnhancedEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.final_norm(self.net(x) + self.residual(x))
```

3 层 MLP + 残差连接 + GELU + LayerNorm，替代当前的 2 层 MLP + ReLU。

---

## 二、Attention 结构改进（方案 7–10）

### 方案 7：相对位置编码注入（RelativeSpatialEncoding） ❌

> **落地状态**：未实施。

```python
class RelativeSpatialEncoding(nn.Module):
    def __init__(self, hidden_dim):
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, pos_query, pos_key):
        dx, dy = pos_key - pos_query
        dist = sqrt(dx**2 + dy**2)
        angle = atan2(dy, dx)
        return self.net([dx, dy, dist, angle])
```

在 attention 的 key embedding 上加相对位置编码，让 attention 直接感知实体间的空间关系，而不是从绝对坐标中推导。

---

### 方案 8：相对位置偏置注入 Attention Score（ALiBi 风格） ❌

> **落地状态**：未实施。

```python
class RelativeSpatialBias(nn.Module):
    def __init__(self, hidden_dim):
        self.bias_net = nn.Sequential(
            nn.Linear(4, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, query_pos, key_pos):
        # 返回 (B, 1, L) 的 attention bias
        ...
```

参考 ALiBi (Press et al., 2022)，在 softmax 前加空间偏置。

---

### 方案 9：候选点 Embedding 注入追捕者相对上下文 ❌

> **落地状态**：未实施。相对几何已在 ego 变换中部分覆盖，但未单独注入 delta_heading 等相对上下文分支。

```python
class CandidateEmbeddingWithRelativeContext(nn.Module):
    def forward(self, pts, self_uav):
        cand = self.cand_enc(pts)
        # 追捕者→候选点：(dx, dy, dist, angle, delta_heading)
        rel = self.rel_enc([dx, dy, dist, angle, delta_heading])
        return self.fuse(cat([cand, rel], dim=-1))
```

在 embedding 阶段注入追捕者到候选点的几何关系，包括 `delta_heading`（要飞向候选点需要转多少弯）。

---

### 方案 10：Hierarchical Attention（组内→组间） ❌

> **落地状态**：未实施。仍为 7 路 star cross-attention（`HeterogeneousAttentionAB`）。

替代当前 7 路并行 cross-attention（star topology）：

| 阶段 | 内容 |
|------|------|
| 阶段 1 | 组内 cross-attention：enemy 组、asset 组、ally 组、candidate 组 |
| 阶段 2 | 组间 cross-attention：`e_self` 为 query，聚合 4 个组摘要 |

**优势**：参数量从 7 个 MHA 降到 5 个（4 组内 + 1 组间），减少约 30%；先局部后全局的信息处理更合理。

---

## 三、Fusion 层改进（方案 11）

### 方案 11：FiLM Conditioning 替代 Dot-Product Gating ❌

> **落地状态**：未实施。设计 B 仍用 `SoftGatingFusion7` 标量门控。

```python
class FiLMFusion(nn.Module):
    def forward(self, ctx):
        params = self.film_gen(e_self).view(-1, 7, 2, D)  # gamma, beta
        modulated = [gamma[:, i] * h_i + beta[:, i] for i, h_i in enumerate(hs)]
        return LayerNorm(e_self + mean(modulated))
```

用 Feature-wise Linear Modulation 替代 `SoftGatingFusion7` 的标量门控，对每个分支做 feature-level 的缩放和平移。

---

## 四、容量与架构改进（方案 12–13）

### 方案 12：hidden_dim 128 → 256 ❌

> **落地状态**：未实施。`configs/train_online0324.yaml` 仍为 `hidden_dim: 128`。

增大表示空间，配合结构化 embedding 使用。**改进 embedding 后再扩容量才有意义。**

### 方案 13：ReLU → GELU ⚠️

> **落地状态**：部分完成。`embeddings.py` 的 `_mlp`、设计 D 的 `self_ctx_fuse`/`CoordResidualHead`、`PermInvariantCritic.value_head` 已用 GELU；`fusion.py`、A/B actor、`actor.py` 中 `PointerActor`/`PointwiseScoringActor` 仍 ReLU。

当前所有 MLP 使用 ReLU，改为 GELU（Transformer 标准激活函数），与 attention 架构更兼容。

---

### 设计 D：两段式残差决策头（Tier 2 新增） ✅

> **落地状态**：已完成。新增 scheme **D**（别名 `d` / `tsr` / `two-stage`），与 A/B/C 并列；YAML `schemes` 需显式启用。

| 组件 | 文件 | 作用 |
|------|------|------|
| `SelfStageScorer` | `actor.py` | 阶段 1：自身上下文对 `e_self_pts` 逐点打分 → `logits_self` |
| `AllyCoordCrossAttention` | `attention.py` | 阶段 2：候选点 query 友军上下文 cross-attn |
| `CoordResidualHead` | `actor.py` | 阶段 2：输出残差 `delta`（最后一层 zero-init） |
| `_embed()` 拆分 | `network.py` | D 的 actor 只跑 embedding；critic 可复用 shared_ctx 缓存 |
| critic 与 C 对齐 | `network.py` | D 的 critic 走 `attn_ab + ConcatMLPFusion8 + PermInvariantCritic` |
| 前向缓存 | `network.py` | 同一 `obs` 下复用 embedding/shared_ctx，减少 actor→critic 重复计算 |

**决策流程**：`logits = logits_self + delta` → mask + softmax → 选候选点索引。

**消融建议**：与 **设计 C**（Point-Wise Scoring）对比，代表「通用逐点打分」vs「自决策 + 友军微调」两种范式。

---

## 五、Critic 改进（方案 14）

### 方案 14：Critic 输入扩展 ⚠️

> **落地状态**：部分完成。`PermInvariantCritic`（attention pooling + 置换不变）✅；D 的 critic 表征已与 C 对齐（同源 `attn_ab+fusion`）✅；`_GlobalStateEncoder` 仍为 13 维 masked-mean，未加入候选点全局统计。

当前 `_GlobalStateEncoder` 只做 mean-pooling（enemies/targets/assets → 13 维），可考虑：

- 加入全局候选点统计信息
- 使用 attention 而非 mean-pooling 做全局聚合

---

## 六、外部建议（建议 1–5）

### 建议 1：Ego-centric 相对坐标（投入小、收益大） ⚠️

> **落地状态**：部分完成。在 `embeddings.py` 前向中对 allies/enemies/assets/candidates 做本机系变换；本机用 `ego_token`；**未**改 `generator.py` 输出；位姿编码为 `[rel_x,rel_y,r,sinΔθ,cosΔθ]`，无独立 bearing sin/cos 五维格式。

在 `generator.py` 出 obs 前，把所有位置平移到本机、按本机航向 θ 旋转。位置编码成 `[Δx_rel, Δy_rel, dist, sin(bearing), cos(bearing)]`，角度一律用 `(sin, cos)`。候选点的 `c_x`, `c_y` 同理换相对。

**参考**：VectorNet / Scene Transformer 的 agent-centric 表征思想。

**直接给注意力提供平移/旋转近似不变性。**

---

### 建议 2：统一 token 化 + 类型 embedding，替换 6 个分散 MLP + 7 路 cross-attn ⚠️

> **落地状态**：部分完成。共享 `enc_cand` + `role_embedding` + `ego_token` ✅；仍为 6 路分散 MLP + 7 路 `HeterogeneousAttentionAB`，**未**替换为单层 Transformer encoder。

把所有实体编成同一 D 维 token 序列：本机、各友机、各被分配敌机、资产、候选点。

- 按语义类型共享编码器（敌机一套、资产一套、pursuer 一套、候选点一套）
- `self` / `ally` 用一个可学习 `role_embedding` 相加区分（取代 `enc_self_pts` / `enc_ally_pts` 两套权重）
- 编码后加 `token_norm = LayerNorm`
- 用**一个标准 Transformer encoder**（Pre-LN：LN→MHSA→残差→LN→FFN→残差）对整个集合做自注意力，`key_padding_mask` 复用现有 mask

**参考**：UPDeT (ICLR 2021)、Multi-Agent Transformer/MAT (NeurIPS 2022)、Relational Deep RL (Zambaldi et al., ICLR 2019)、AlphaStar 实体编码器、Set Transformer (ICML 2019)。

---

### 建议 3：动作头保留「候选点作 query 的指针/逐点打分」（设计 C 方向） ⚠️

> **落地状态**：部分完成。设计 C（`PointwiseScoringActor`）与设计 D（两段式残差）✅；训练配置默认 `schemes` 已切到 C+D；A/B 仍保留兼容实现。

让候选点 token 经过 encoder（或对 encoder 输出做 cross-attn），再逐点打分输出 logits。天然处理变长候选集、对候选顺序置换不变，比「先压成 `h_env` 再点积」更贴合动作选择。

**参考**：Pointer Networks (Vinyals et al., NeurIPS 2015)。

A/B 的「先融合成单一 `h_env`」是信息瓶颈；**C 范式最契合顶会做法**。

---

### 建议 4：Critic 改成置换不变的注意力池化（PIC） ✅

> **落地状态**：已完成。`PermInvariantCritic` 替代旧 `CentralCritic`；静态 `__init__` 构建，attention pooling 跨 agent 聚合；**同时修复** lazy init 导致 critic 参数未被优化器跟踪的 bug。A/B/C/D 共用。

不要 concat-over-P。改为：每个 agent 一个 token → 跨 agent 做注意力/均值池化得到全局上下文 → 每 agent 输出标量 value。P 变化不再重建网络、且置换不变。

**参考**：PIC (Permutation Invariant Critic, Liu et al., CoRL 2019)。

**注意**：需确认训练中 P 是否跨地图变化；若会变，旧 critic 的随机重建会严重破坏训练。

---

### 建议 5：归一化与训练稳定性 ⚠️

> **落地状态**：部分完成。`pos_scale` 常数归一化 + 各 token `LayerNorm` + 质量分支无参压缩（`asinh/log1p/signed_log1p/sin-cos`）✅；**未**实现 running mean/std、Pre-LN Transformer block、全局 FFN 残差块。

- 输入特征做 running mean/std 标准化（坐标、Δt、path_L 尺度差异巨大）
- 注意力块统一 Pre-LN + 残差 + FFN（dim ≈ 4D）
- 配合后续再调 `hidden_dim` / 层数

---

## 七、优先级路线图

| 优先级 | 内容 | 预期收益 | 落地 |
|--------|------|----------|------|
| **1（高/低成本）** | 相对坐标 + 输入标准化 + token 后 LayerNorm | 高 | ⚠️ 部分（running stats 未做，但已加无参压缩） |
| **2（高）** | 统一 token + 类型/角色 embedding + 共享候选点编码器 | 高 | ⚠️ 部分（缺 Transformer） |
| **3（高）** | 裸 MHA 升级为完整 Transformer block，动作头收敛到 C 范式 | 高 | ⚠️ 部分（训练默认 C+D，A/B 仍兼容保留） |
| **4（高）** | Critic 改置换不变池化（PIC） | 高 | ✅ 已完成 |
| **5（中）** | 容量缩放（hidden_dim、层数）与消融 | 中 | ❌ 未开始 |
| **—（Tier 2）** | 设计 D 两段式残差决策头 + critic 对齐 C + 前向缓存 | 中–高 | ✅ 已完成 |

---

## 八、落地状态索引

完整摘要见文档开头“实施摘要（Tier 0–2 + C/D 对齐，2026-06）”小节。下表与各方案章节标题后的 ✅/⚠️/❌ 标记一致。

### 状态总览

| 状态 | 数量 | 编号 |
|------|------|------|
| ✅ 已完成 | 3 | 方案 2、建议 4、设计 D |
| ⚠️ 部分完成 | 9 | 方案 3/4/5/13/14、建议 1/2/3/5 |
| ❌ 未开始 | 9 | 方案 1、6–12 |

### 已应用（详细）

| 编号 | 状态 | 落地位置 / 说明 |
|------|------|-----------------|
| 方案 2 | ✅ | `StructuredCandidateEncoder`：几何 5 维 + 质量 6 维双分支（含无参压缩） |
| 方案 3 | ⚠️ | 无 `LearnableFeatureScale`；采用 `pos_scale` + 无参压缩替代 |
| 方案 4 | ⚠️ | `Δd/path_L` 用 `-log1p`，`Δθ` 用 `sin/cos`，`ΔV` 用 `signed_log1p` |
| 方案 5 | ⚠️ | 各 token 独立 `LayerNorm`，非 `RMSNorm` 统一 norm |
| 方案 13 | ⚠️ | `embeddings.py`、设计 D、critic value head 用 GELU；fusion/A/B actor 仍 ReLU |
| 方案 14 | ⚠️ | `PermInvariantCritic` ✅；D critic 已与 C 对齐；`_GlobalStateEncoder` 仍 13 维 mean-pool |
| 设计 D | ✅ | `SelfStageScorer` + `AllyCoordCrossAttention` + `CoordResidualHead` + critic C 对齐 + 前向缓存 |
| 建议 1 | ⚠️ | 在 `embeddings.py` 前向做 ego 变换，**未**改 `generator.py` |
| 建议 2 | ⚠️ | 共享 `enc_cand` + `role_embedding` + `ego_token` ✅；统一 Transformer ❌ |
| 建议 3 | ⚠️ | 设计 C/D 为逐点打分；训练默认 C+D；A/B 保留兼容 |
| 建议 4 | ✅ | `PermInvariantCritic` 替代旧 `CentralCritic`（并修复优化器未跟踪 bug） |
| 建议 5 | ⚠️ | `pos_scale` + token LayerNorm + 无参压缩；无 running stats、无 Pre-LN Transformer block |

**配置与透传**：`pos_scale` 默认值 1000 → `configs/train_online0324.yaml` → `TrainConfig` → `build_actor_critic_schemes` → `UAVInterceptionNetwork` → `TODCEmbeddings`；CLI `--pos-scale` 可覆盖。

**验证**：历史阶段曾通过 `pytest tests/test_phase3_models.py` 13 项全绿（含 `test_design_d_*`、`test_critic_params_tracked_by_optimizer`）；本轮改动已执行 `read_lints` 与 `python3 -m py_compile`。

### 未应用

| 编号 | 说明 |
|------|------|
| 方案 1 | 四组结构化（空间/时间/几何/路径），当前为双分支方案 2 |
| 方案 3（完整） | `LearnableFeatureScale` |
| 方案 4（完整） | 显式可学习方向参数（当前为无参方向统一） |
| 方案 6 | `EnhancedEmbedding` 3 层残差 MLP |
| 方案 7–10 | 相对位置编码、ALiBi 偏置、候选相对上下文、分层 attention |
| 方案 11 | `FiLMFusion` |
| 方案 12 | `hidden_dim` 128→256（配置仍为 128） |
| 方案 13（完整） | 全局 ReLU→GELU |
| 方案 14（完整） | Critic 全局候选统计、attention 全局聚合 |
| 建议 1（完整） | `generator.py` 统一 ego-centric + bearing sin/cos |
| 建议 2（完整） | 统一 token + 单层 Transformer encoder |
| 建议 5（完整） | **`RunningMeanStd` running mean/std**、Pre-LN Transformer block |

### 旧训练 run 对照

| 项目 | `wandb/run-20260529_001423-lxs7dmoh` | 当前工作区 |
|------|--------------------------------------|------------|
| commit | `d0f285f`（改动前） | 含 Tier 0–2 未提交改动 |
| `pos_scale` | 无 | 默认 1000 |
| critic | 懒加载 `CentralCritic`（未训练） | `PermInvariantCritic` |
| schemes | A + B | 默认 C + D（A/B 保留可选） |
| 典型症状 | `grad_norm` 1787–4666，`explained_variance` B≈0.05 | 需重训验证 |

---

## 九、参考文献

| 简称 | 文献 |
|------|------|
| ALiBi | Press et al., Train Short, Test Long: Attention with Linear Biases, 2022 |
| UPDeT | Hu et al., UPDeT: Universal Multi-agent RL via Policy Decoupling with Transformers, ICLR 2021 |
| MAT | Wen et al., Multi-Agent Reinforcement Learning is a Sequence Modeling Problem, NeurIPS 2022 |
| Relational RL | Zambaldi et al., Deep Reinforcement Learning with Relational Inductive Biases, ICLR 2019 |
| Set Transformer | Lee et al., Set Transformer, ICML 2019 |
| Pointer Networks | Vinyals et al., Pointer Networks, NeurIPS 2015 |
| PIC | Liu et al., PIC: Permutation Invariant Critic, CoRL 2019 |
| VectorNet | Gao et al., VectorNet: Encoding HD Maps and Agent Dynamics, CVPR 2020 |

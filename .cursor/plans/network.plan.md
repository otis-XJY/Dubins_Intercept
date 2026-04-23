---
name: hetero-attn-marl-refactor
overview: 将现有 `marl/models.py` 的 A/B/C 三种策略网络按“异构并行注意力”文档拆分为独立模块，并将仓库目录切换到 `env/ models/ algorithms/ configs/ train.py` 结构；保持观测键/shape 与训练/环境逻辑不变，仅做结构与网络模块边界重构。
todos:
  - id: scan-deps
    content: 梳理 `marl/train_online0325.py`、`marl/obs_generator.py`、tests 对 `marl.models` 的依赖点，列出需要改动的 import 清单（仅路径变化）
    status: pending
  - id: split-models
    content: 按 embeddings/attention/fusion/actor/critic 拆分 `UAVInterceptionNetwork`，保持输出字典键与 mask 行为完全一致
    status: pending
  - id: move-modules
    content: 迁移目录到 `env/ models/ algorithms/` 并调整入口脚本 import；保留（或临时保留）`marl/` 兼容层避免断裂
    status: pending
  - id: update-config-entrypoints
    content: 新增/迁移 `configs/default.yaml`、`train.py`、`evaluate.py`，保持原训练参数语义不变
    status: pending
  - id: verify-tests
    content: 运行现有 tests（尤其 `test_phase3_models.py` 与 train/env smoke）确保重构只影响结构不影响行为
    status: pending
isProject: false
---

# 异构并行注意力网络：结构重构设计与实施计划

## 目标与硬约束
- **目标**：参考你给的“异构并行注意力多智能体拦截网络”文档，把当前网络实现从单文件 [`marl/models.py`](/home/xujunyi/Dubins_Intercept/marl/models.py) 拆成清晰的模块边界，并把工程目录重构为 `env/ models/ algorithms/ configs/ train.py evaluate.py`。
- **硬约束（按你选择）**：
  - **obs 输入契约不变**：继续使用现有键与 shape（如 `self_uav/allies_local/self_pts/.../self_pts_mask`），由 [`marl/obs_generator.py`](/home/xujunyi/Dubins_Intercept/marl/obs_generator.py) 产出；不改语义。
  - **训练/环境实现不改语义**：[`marl/train_online0325.py`](/home/xujunyi/Dubins_Intercept/marl/train_online0325.py)、[`marl/MARL_env.py`](/home/xujunyi/Dubins_Intercept/marl/MARL_env.py)、[`marl/mappo.py`](/home/xujunyi/Dubins_Intercept/marl/mappo.py) 的算法逻辑不改，只允许“搬家/改 import/薄包装”。
  - **输出契约不变**：保持 `actor_forward` 输出键 `action_logits/action_probs/best_candidate_idx` 与 `critic_forward` 输出 shape，确保现有测试（如 [`tests/test_phase3_models.py`](/home/xujunyi/Dubins_Intercept/tests/test_phase3_models.py)）仍通过。

## 现状与文档的对齐关系（关键差异）
- 文档假设的实体为 4 类（self/allies/enemies/points），而现网在 [`UAVInterceptionNetwork`](/home/xujunyi/Dubins_Intercept/marl/models.py) 实际是 **8 路分支**（`N_BRANCHES=8`）：
  - `self_uav`、`allies_local`、`self_pts`、`ally_pts`、`enemy_assigned_self`、`enemy_assigned_per_ally`、`asset_target_self`、`asset_target_per_ally`
- 因为你要求 obs 不变，所以“按文档模块化”的落地方式是：**保留 8 路分支的数据流与数值逻辑**，但把它们拆进“embedding/attention/fusion/actor/critic”的模块边界中（命名与职责对齐文档，实体类别数不强行降到 4）。

## 两种可行结构方案（你已选 C：完全切换到文档目录）
- **方案 C（推荐且已选）**：新增顶层 `env/ models/ algorithms/` 等目录，并把旧 `marl/` 迁移/缩减为兼容层或直接替换入口与 import。
- **备选（不采用）**：仅在 `marl/` 内部拆分。

## 网络结构设计（模块边界与类职责）
下面设计保证：**数值等价**（同一权重初始化与同一输入下输出一致），仅重排代码与参数归属。

### 1) `models/embeddings.py`
- **职责**：把 8 路原始输入分别编码到同一 `hidden_dim=D`。
- **落地映射（来自现网）**：
  - `enc_self : MLP(3→D)` 对 `obs['self_uav']`
  - `enc_ally : MLP(3→D)` 对 `obs['allies_local']`
  - `enc_self_pts : MLP(8→D)` 对 `obs['self_pts']`
  - `enc_ally_pts : MLP(8→D)` 对 `obs['ally_pts']`
  - `enc_enemy : MLP(3→D)` 对 `enemy_assigned_*`
  - `enc_asset : MLP(2→D)` 对 `asset_target_*`
- **输出**：保持当前 `_encode_obs` 的 8 个张量输出顺序与 shape。

### 2) `models/attention.py`
- **职责**：封装注意力调用与 mask 处理策略（你现网的 `_apply_mha`），并实现“异构分支注意力”。
- **落地映射**：
  - 复用 `nn.MultiheadAttention(batch_first=True)`，每个分支独立实例（与你现网一致：`mha_ally/mha_enemy_self/...` 都是独立模块）。
  - 统一 `key_padding_mask` 逻辑：现网把 `mask` 取反后喂给 `key_padding_mask`，并对 all-masked 行做输出归零；该逻辑保持不变。
- **输出**：
  - 设计 A/B：产生 7 个上下文（`h_ally/h_spts/h_apts/h_eself/h_eally/h_ast_s/h_ast_a`）+ `e_self`，并负责 squeeze。
  - 设计 C：以 `e_self_pts` 为 Query，对多类 Key/Value 做 cross-attn（与你现网一致）。

### 3) `models/fusion.py`
- **职责**：把 8 路特征融合为 `h_env`。
- **落地映射**：
  - 设计 A：`Concat + MLP`（对应现网 `fusion_mlp`）。
  - 设计 B：`Soft-Gating`（对应现网 `gate_*` + softmax 权重 `alphas`）；**注意：现网 gate 目前是 7 路（不含 e_self 本体），输出 `alphas` shape 为 `(B,7)`**，这点与文档示例 `(B,3)` 不同，但会保留以保证等价。
- **输出**：保持现网 `h_env` 计算方式。

### 4) `models/actor.py`
- **职责**：实现 A/B 的指针式打分、以及 C 的逐点 scoring。
- **落地映射**：
  - 设计 A/B：
    - `q_opt: Linear(D→D)`、`k_opt: Linear(D→D)`
    - `logits = sum(q_action * k_action) / sqrt(D)`（现网第 316-333 行附近逻辑）
  - 设计 C：
    - `scoring_mlp: MLP(8D→1)` 逐点输出 `logits`
  - `mask` 后处理：保持现网对 `self_pts_mask` 的 `masked_fill(~mask, -1e9)` 与 “全无可用点” 的退化策略（输出 probs 全 0、把第 0 位设 1、best_candidate_idx=-1）。
- **对外接口**：提供 `actor_forward(obs)->dict`，返回键保持不变。

### 5) `models/critic.py`
- **职责**：集中式 critic（现网 `_ensure_critic` + `critic_forward`）。
- **落地映射**：
  - 保持“按 batch 首维 `P` 动态创建 critic MLP（输出 shape `(P,)`）”的策略，避免影响训练脚本。
  - A/B：直接拼接每机 8 路特征；C：对点级特征做 masked mean pool，再拼接。

### 6) `models/__init__.py` 与顶层工厂
- 保留现有外部语义：`resolve_design_mode / build_actor_critic_schemes / TODCHeteroActorCritic / UAVInterceptionNetwork`。
- 但它们将成为“组装层”：内部组合 embeddings+attention+fusion/actor+critic。

## 工程目录重构（完全切换到文档树，但不改语义）
在仓库根目录新增并迁移到：
- [`env/intercept_env.py`](env/intercept_env.py)：承载当前 `TODCMARLEnv`（从 [`marl/MARL_env.py`](/home/xujunyi/Dubins_Intercept/marl/MARL_env.py) 迁移/改名）
- [`env/utils.py`](env/utils.py)：若需要，从现有工具函数中“原样搬运”
- [`models/`](models/)：按上面的网络拆分
- [`algorithms/mappo.py`](algorithms/mappo.py)：从 [`marl/mappo.py`](/home/xujunyi/Dubins_Intercept/marl/mappo.py) 原样迁移
- [`configs/default.yaml`](configs/default.yaml)：由现有 [`configs/train_online0324.yaml`](/home/xujunyi/Dubins_Intercept/configs/train_online0324.yaml) 精简/重命名而来（字段可保持兼容，尽量不改含义）
- [`train.py`](train.py)：把 [`marl/train_online0325.py`](/home/xujunyi/Dubins_Intercept/marl/train_online0325.py) 入口迁移为顶层脚本；内部逻辑不改，只调整 import 路径
- [`evaluate.py`](evaluate.py)：从现有评估逻辑抽出（若当前训练脚本内置评估则“原样搬运”）

为避免一次性改动过大，目录切换采用“两阶段迁移”更稳：
1. **先复制新目录并改 import**，保持旧 `marl/` 暂时可运行
2. 测试通过后，再决定是否删除/瘦身 `marl/`

## 验证策略（只用现有测试证明“结构等价”）
- 重点跑：
  - [`tests/test_phase3_models.py`](/home/xujunyi/Dubins_Intercept/tests/test_phase3_models.py)（forward/masking/别名解析/兼容 use_soft_gating）
  - 训练 smoke（如 [`tests/test_train_online_smoke.py`](/home/xujunyi/Dubins_Intercept/tests/test_train_online_smoke.py)）
- 验证点：
  - 输出 key/shape 全部一致
  - mask 退化路径（全 False）仍然不会 nan，且 idx=-1

## 风险与回滚
- **最大风险**：目录硬迁移导致 import 路径断裂。
- **回滚策略**：保留一层 `marl/` 兼容转发（例如 `marl/models.py` 仅 `from models import ...`），在所有入口替换完成后再移除。


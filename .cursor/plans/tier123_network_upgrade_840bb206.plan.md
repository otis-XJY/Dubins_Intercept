---
name: tier123 network upgrade
overview: 拦截策略网络分阶段升级。执行顺序经修正：先修复 critic 未被优化器跟踪的 bug 并改置换不变 critic（原 Tier 3，提到最前），再做 Tier 1 候选点编码改进，最后做 Tier 2 两段式残差决策头（设计 D）。全程保持 obs 键/shape 与 actor/critic 输出契约不变。
todos:
  - id: s0-critic-fix
    content: "阶段0(原Tier3,最高优先): critic.py 用静态 PermInvariantCritic 替换懒加载 concat-over-P，修复优化器未跟踪 bug；network.py critic_forward 传 h_env(P,D)+h_global(D)"
    status: completed
  - id: s0-verify
    content: "阶段0验证: 断言 critic 参数在 list(model.parameters()) 中；pytest tests/test_phase3_models.py 全绿；确认 DDP 下无 unused param"
    status: completed
  - id: t1-cand-encoder
    content: "Tier1: embeddings.py 合并 self/ally 候选编码器为共享 enc_cand + role_embedding(self/ally)，删除 enc_ally_pts"
    status: completed
  - id: t1-structured
    content: "Tier1: 候选编码结构化为 几何5维/质量5维 双子MLP -> D//2 -> concat 投影 D"
    status: completed
  - id: t1-gelu-scale
    content: "Tier1: _mlp 激活改 GELU；位置+距离量级通道(rel_x,rel_y,r,Δd,path_L)按可配置 pos_scale 归一化"
    status: completed
  - id: t1-plumb
    content: "Tier1: pos_scale 从 YAML -> TrainConfig -> build_actor_critic_schemes -> UAVInterceptionNetwork -> TODCEmbeddings 透传"
    status: completed
  - id: t2-embed-split
    content: "Tier2: network.py 拆出 _embed(只编码) 与 attn_ab；设计 D 的 actor 仅用 _embed，不重算 7 路注意力"
    status: completed
  - id: t2-schemes
    content: "Tier2: schemes.py 注册设计 D (Two-Stage Residual Network) 及别名 tsr/two-stage"
    status: completed
  - id: t2-attn
    content: "Tier2: attention.py 新增 AllyCoordCrossAttention(候选点 query 对友军上下文 cross-attn + 拼接 mask)"
    status: completed
  - id: t2-heads
    content: "Tier2: actor.py 新增 SelfStageScorer 与 CoordResidualHead(最后层 zero-init)"
    status: completed
  - id: t2-network
    content: "Tier2: network.py 增加 design_mode==D 的构建与 actor_forward 两段式残差 logits=logits_self+delta；D 的 critic 用廉价池化表征"
    status: completed
  - id: verify
    content: "验证: 加设计D前向用例(断言 logits/probs/idx/value 形状)；pytest tests/ 全量回归；可选 scripts/smoke_verify_env.py"
    status: completed
isProject: false
---

## 背景与关键发现

- 真正生效的实现是 `marl/model_blocks/`（`nn/blocks/network.py` 仅转发）。`nn/blocks/` 下同名文件是无人导入的死副本，本方案只改 `model_blocks/`。
- 相对坐标（建议1）已在 `marl/model_blocks/embeddings.py` 落地：位姿类 3→5、纯位置类 2→3、候选点 8→10，本机用可学习 `ego_token`。
- P（追捕者数）当前所有 map 恒为 3，单回合内不变。

### 关键 bug（决定执行顺序）

- 模型在 `marl/runners/online_train.py:548` 构建，优化器在 `:613` 用 `model.parameters()` 创建，**早于任何前向**。
- `marl/model_blocks/critic.py` 的 `CentralCritic._critic_net` 懒加载（首次 `critic_forward` 才建），因此 critic 参数**不在优化器 param_groups 里**。
- `marl/rl/mappo.py:151` 的 `optimizer.step()` 只更新锁定的 param_groups → **critic 永远停在随机初始化**，PPO 的 value/GAE 基于随机固定 value 函数，可能已严重拖累当前训练。
- `distributed: true`，DDP wrap 时也看不到 lazy critic 参数。
- 结论：critic 修复必须**最先做**（下方“阶段0”），独立于 Tier 1/2。

## 阶段 0（原 Tier 3，最高优先）：置换不变 critic + 修 bug

改 `marl/model_blocks/critic.py`，把 `CentralCritic` 换成 `PermInvariantCritic`（`__init__` 静态构建，维度与 P 无关）：

- 输入：每机 `h_env (P,D)` 与全局 `h_global (D)`（沿用 `network.py` 的 `_GlobalStateEncoder`，本就是 masked-mean 置换不变）。
- 跨机聚合：attention pooling（可学习 query 对 P 个 agent token softmax 加权和）得 `g (D)`，替代 concat-over-P。
- 每机 value：`value_head(cat(h_env_i, g, h_global)) -> 1`，逐 token 应用 → `(P,)`。
- 全部参数在 `__init__` 构建 → 被优化器跟踪（修 bug）、P 变化不重建、置换不变、DDP 友好。
- `critic_forward` 改为直接传 `h_env (P,D)` 与 `h_global (D)`，不再 concat+reshape；A/B/C/D 共用。
- 注意：critic 结构变化使旧 checkpoint 不兼容（属预期，需重训）。

验证：断言构造后 critic 参数出现在 `list(model.parameters())`；`pytest tests/test_phase3_models.py -q` 全绿；确认 `ddp_find_unused_parameters=false` 下无 unused 参数。

## Tier 1：候选点编码改进（改 `marl/model_blocks/embeddings.py`）

对 A/B/C/D 全部生效，obs 接口不变。

- T1.1 合并 `enc_self_pts`/`enc_ally_pts` 为共享 `enc_cand`，加可学习 `role_embedding`（self/ally）区分来源，删除 `enc_ally_pts`。
- T1.2 结构化候选编码：候选 10 维拆为几何 5 维 `[rel_x,rel_y,r,sinθ,cosθ]` 与质量 5 维 `[Δt,Δd,Δθ,path_L,ΔV]`，各过子 MLP 到 `D//2`（第二支用 `D-D//2` 保证维度对齐），concat 投影到 D。
- T1.3 `_mlp` 激活 ReLU→GELU（仅 embeddings 内）。
- T1.4 尺度归一化（**修正**：覆盖所有距离量级通道）：`rel_x,rel_y,r`（几何）与 `Δd,path_L`（质量）进 MLP 前除以可配置 `pos_scale`；角度 sin/cos、`Δt`、`Δθ`、`ΔV` 不缩放；后置 LayerNorm 保留。
- T1.5 透传 `pos_scale`：YAML → `TrainConfig` → `build_actor_critic_schemes` → `UAVInterceptionNetwork.__init__` → `TODCEmbeddings.__init__`，给默认值。
- 不变：`forward` 仍返回 8 个 `e_*`（含 `e_self`=ego token），下游 `HeterogeneousAttentionAB` 形状不变。

## Tier 2：两段式残差决策头 = 设计 D

新增设计模式 D（别名 `tsr`/`two-stage`），与 A/B/C 并列，YAML `schemes` 显式启用做消融。

- `marl/model_blocks/network.py`：**拆分 `_embed(obs)`（只跑 `TODCEmbeddings` 得 8 个 `e_*`）与 attn_ab**。A/B/C 仍用 `_shared_ctx`(=`_embed`+attn_ab)，**设计 D 的 actor 只调 `_embed`，不重算 7 路注意力**。
- `marl/model_blocks/schemes.py`：注册 D = `Two-Stage Residual Network` 及别名。
- `marl/model_blocks/attention.py`：新增 `AllyCoordCrossAttention`，query=候选点，KV=友军上下文 concat(`e_ally,e_ally_pts,e_eally,e_ast_a`)，padding mask=concat(`ally_mask,ally_pts_mask,ally_enemy_mask,ally_enemy_mask`)，复用 `PointsContextCrossAttention` 的全屏蔽兜底。
- `marl/model_blocks/actor.py`：新增 `SelfStageScorer`（self 上下文对 `e_self_pts` 逐点打分 → `logits_self`）与 `CoordResidualHead`（输入 `cat(e_self_pts, coord_ctx)` → 逐点 `delta`，最后一层 zero-init，初始 delta≈0 但幅度不受约束可强力翻转）。
- 设计 D 的 critic 表征：**不走 ConcatMLPFusion8 / attn_ab**，改用对全部实体 token 的廉价池化（如对 `e_*` 做 masked mean / 一层 attention pool）得到 `h_env (P,D)`，再进阶段0的 `PermInvariantCritic`。

设计 D 的 `actor_forward` 流程：
1. `e_self,e_ally,e_self_pts,e_ally_pts,e_eself,e_eally,e_ast_s,e_ast_a = self._embed(obs)`
2. `self_ctx = self_ctx_fuse(cat(e_self, e_eself, e_ast_s))`（阶段1，自身上下文）
3. `logits_self = self.stage1_scorer(self_ctx, e_self_pts)`（B,N）
4. `coord_ctx, w_coord = self.coord_attn(q=e_self_pts, KV=友军, masks)`
5. `delta = self.coord_head(cat(e_self_pts, coord_ctx))`（B,N，zero-init）
6. `logits = logits_self + delta`
7. `postprocess_mask_and_sample(logits, obs["self_pts_mask"])` → probs/idx
8. 输出含 `attn_weights`(可为空)、`coord_attn_weights=w_coord`

## 验证

- 给 `tests/test_phase3_models.py` 的 `_fake_obs` 加 `design_mode="D"` 前向用例，断言 `logits/probs/idx/value` 形状。
- `conda activate dubins && pytest tests/test_phase3_models.py -q` 全绿。
- `pytest tests/` 全量回归；可选 `python scripts/smoke_verify_env.py` 端到端烟雾验证。

## 默认决策（可调整）

- 阶段0 新 critic 对 A/B/C/D 全部生效（必须修 bug）。
- 设计 D 为新增模式，A/B/C 保留做消融；默认 `schemes` 不含 D。
- 输入标准化用常数 `pos_scale`（可配置），不引入对小 batch 不稳的 BatchNorm/running-stats。
- 执行顺序：阶段0 → Tier 1 → Tier 2，每段独立验证后再进入下一段。

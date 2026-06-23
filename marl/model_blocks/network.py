from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .actor import (
    CoordResidualHead,
    CounterfactualQHead,
    PointerActor,
    PointwiseScoringActor,
    SelfStageScorer,
    TemporalSelfCtxEncoder,
    postprocess_mask_and_sample,
)
from .attention import AllyCoordCrossAttention, HeterogeneousAttentionAB, PointsContextCrossAttention
from .critic import PermInvariantCritic
from .embeddings import TODCEmbeddings
from .fusion import ConcatMLPFusion8, SoftGatingFusion7
from .schemes import SCHEME_KEY_TO_NAME, resolve_design_mode


def _masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 feat 按 mask 做均值池化；mask 全零时返回零向量。"""
    m = mask.float().unsqueeze(-1)  # (..., 1)
    s = (feat * m).sum(dim=0)
    c = m.sum(dim=0).clamp(min=1.0)
    return (s / c).squeeze(0)


class _GlobalStateEncoder(nn.Module):
    """从 obs 中提取固定维度的全局状态向量（masked mean-pooling）。"""

    RAW_DIM = 13  # pursuer(3) + enemy(3) + asset(2) + target(2) + ratios(3)

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.RAW_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pursuer 位置均值
        p_feat = obs["self_uav"][:, 0, :]          # (P, 3)
        p_mask = obs["pursuer_active"].float()      # (P,)
        p_mean = _masked_mean(p_feat, p_mask)       # (3,)

        # enemy 位置均值（取 pursuer 0 的视图，数据对所有 pursuer 重复）
        e_feat = obs["enemies"][0]                   # (E, 3)
        e_mask = obs["enemy_mask"][0].float()        # (E,)
        e_mean = _masked_mean(e_feat, e_mask)        # (3,)

        # asset 位置均值
        a_feat = obs["assets"][0]                    # (V, 2)
        a_mask = obs["asset_mask"][0].float()        # (V,)
        a_mean = _masked_mean(a_feat, a_mask)        # (2,)

        # target 位置均值
        t_feat = obs["targets"][0]                   # (E, 2)
        t_mask = obs["target_mask"][0].float()       # (E,)
        t_mean = _masked_mean(t_feat, t_mask)        # (2,)

        # 存活比例
        p_ratio = p_mask.mean().unsqueeze(0)         # (1,)
        e_ratio = e_mask.mean().unsqueeze(0)         # (1,)
        a_ratio = a_mask.mean().unsqueeze(0)         # (1,)

        raw = torch.cat([p_mean, e_mean, a_mean, t_mean, p_ratio, e_ratio, a_ratio])  # (13,)
        return self.net(raw)  # (D,)


class UAVInterceptionNetwork(nn.Module):
    """异构并行注意力网络（A/B/C）。

    - obs 键/shape 与旧实现一致
    - actor/critic 输出契约与旧实现一致（tests 依赖）
    """

    N_BRANCHES = 8

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        design_mode: Optional[str] = None,
        use_soft_gating: Optional[bool] = None,
        pos_scale: float = 1000.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.pos_scale = float(pos_scale)
        self.design_mode = resolve_design_mode(design_mode, use_soft_gating=use_soft_gating)
        self.design_name = SCHEME_KEY_TO_NAME[self.design_mode]

        self.emb = TODCEmbeddings(self.hidden_dim, pos_scale=self.pos_scale)
        self.attn_ab = HeterogeneousAttentionAB(self.hidden_dim, self.num_heads)
        self.global_encoder = _GlobalStateEncoder(self.hidden_dim)

        # 各设计专属子模块（未用到的置 None）
        self.fusion_a = None
        self.fusion_b = None
        self.ptr = None
        self.points_ctx = None
        self.score_c = None
        self._c_fusion_for_critic = None
        # 设计 D（两段式残差）专属
        self.self_ctx_fuse = None
        self.coord_attn = None
        self.stage1_scorer = None
        self.coord_head = None
        self.cf_q_head = None  # 反事实 Q_i 头（近似 COMA 信用分配）
        self.temporal_encoder = None  # 多帧时序编码器（仅 Design D 使用）

        if self.design_mode == "A":
            self.fusion_a = ConcatMLPFusion8(self.hidden_dim)
            self.ptr = PointerActor(self.hidden_dim)
        elif self.design_mode == "B":
            self.fusion_b = SoftGatingFusion7(self.hidden_dim)
            self.ptr = PointerActor(self.hidden_dim)
        elif self.design_mode == "C":
            self.points_ctx = PointsContextCrossAttention(self.hidden_dim, self.num_heads)
            self.score_c = PointwiseScoringActor(self.hidden_dim, score_hidden_dim=self.hidden_dim * 2)
            # Critic for scheme C uses concat fusion over 8 branches.
            self._c_fusion_for_critic = ConcatMLPFusion8(self.hidden_dim)
        else:  # design_mode == "D"：两段式残差决策头
            self.self_ctx_fuse = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            )
            self.coord_attn = AllyCoordCrossAttention(self.hidden_dim, self.num_heads)
            self.stage1_scorer = SelfStageScorer(self.hidden_dim)
            self.coord_head = CoordResidualHead(self.hidden_dim)
            self.cf_q_head = CounterfactualQHead(self.hidden_dim)
            # D 的 critic 与 C 对齐：走同源 attn_ab + concat fusion，避免廉价 mean-pool 的表征失配。
            self._c_fusion_for_critic = ConcatMLPFusion8(self.hidden_dim)
            # 多帧时序编码器：对 self_ctx 多帧序列做 Transformer 聚合
            # temporal_window 等参数由构造器 kwargs 传入
            self.temporal_encoder = TemporalSelfCtxEncoder(
                hidden_dim=self.hidden_dim,
                num_heads=int(kwargs.get("temporal_heads", 2)),
                num_layers=int(kwargs.get("temporal_layers", 1)),
                max_window=int(kwargs.get("temporal_window", 4)),
            )

        self.critic = PermInvariantCritic(embed_dim=self.hidden_dim, hidden_dim=self.hidden_dim)

        # 前向缓存：同一 obs 对象下复用 embedding/shared_ctx，减少 actor->critic 重复计算。
        self._cache_obs_ref = None
        self._cache_embs = None
        self._cache_ctx = None
        self._cache_attn_w = None

    def _ensure_cache_obs(self, obs: Dict[str, torch.Tensor]) -> None:
        if self._cache_obs_ref is obs:
            return
        self._cache_obs_ref = obs
        self._cache_embs = None
        self._cache_ctx = None
        self._cache_attn_w = None

    def _embed(self, obs: Dict[str, torch.Tensor]):
        """仅做实体编码，返回 8 个 e_*（不跑异构注意力）。"""
        self._ensure_cache_obs(obs)
        if self._cache_embs is None:
            self._cache_embs = self.emb(obs)
        return self._cache_embs

    def _shared_ctx(self, obs: Dict[str, torch.Tensor], embs=None):
        self._ensure_cache_obs(obs)
        if embs is None:
            embs = self._embed(obs)
        else:
            self._cache_embs = embs

        if self._cache_ctx is None or self._cache_attn_w is None:
            e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = embs
            self._cache_ctx, self._cache_attn_w = self.attn_ab(
                e_self=e_self,
                e_ally=e_ally,
                e_self_pts=e_self_pts,
                e_ally_pts=e_ally_pts,
                e_eself=e_eself,
                e_eally=e_eally,
                e_ast_s=e_ast_s,
                e_ast_a=e_ast_a,
                ally_mask=obs["ally_mask"],
                self_pts_mask=obs["self_pts_mask"],
                ally_pts_mask=obs["ally_pts_mask"],
                enemy_self_mask=obs["enemy_self_mask"].bool(),
                ally_enemy_mask=obs["ally_enemy_mask"],
            )
        return embs, self._cache_ctx, self._cache_attn_w

    def _h_env(self, obs: Dict[str, torch.Tensor], ctx) -> torch.Tensor:
        if self.design_mode == "A":
            assert self.fusion_a is not None
            return self.fusion_a(ctx)
        if self.design_mode == "B":
            assert self.fusion_b is not None
            h_env, _gate = self.fusion_b(ctx)
            return h_env
        # C/D 统一使用同源 concat fusion（来自 attn_ab 的 8 路上下文）
        assert self._c_fusion_for_critic is not None
        return self._c_fusion_for_critic(ctx)

    def _actor_forward_d(
        self,
        obs: Dict[str, torch.Tensor],
        self_ctx_history: Optional[torch.Tensor] = None,
        self_ctx_history_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """设计 D：阶段1 自身打分 + 阶段2 友军协同残差 + 反事实信用分配 + 可选多帧时序聚合。

        self_ctx_history: (P, T, D) 最近 T 帧的 self_ctx 序列（含当前帧在最后一帧）
        self_ctx_history_mask: (P, T) True=有效帧
        若不提供历史帧，退化为单帧前馈（T=1，时序编码器恒等）。
        """
        embs = self._embed(obs)
        e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = embs
        assert self.self_ctx_fuse is not None and self.coord_attn is not None
        assert self.stage1_scorer is not None and self.coord_head is not None
        assert self.cf_q_head is not None

        self_ctx = self.self_ctx_fuse(
            torch.cat([e_self.squeeze(1), e_eself.squeeze(1), e_ast_s.squeeze(1)], dim=-1)
        )  # (P, D)

        # 多帧时序聚合：若提供了历史 self_ctx 序列，用 TemporalSelfCtxEncoder 聚合
        if self_ctx_history is not None and self.temporal_encoder is not None:
            # 将当前步 self_ctx 拼接到历史序列最后一帧
            # self_ctx_history 形状 (P, T-1, D)，当前帧 self_ctx (P, D)
            P = self_ctx.shape[0]
            T_prev = self_ctx_history.shape[1]
            current_frame = self_ctx.unsqueeze(1)  # (P, 1, D)
            full_seq = torch.cat([self_ctx_history, current_frame], dim=1)  # (P, T, D)
            # 构建有效帧 mask：历史帧 mask + 当前帧（始终有效）
            current_valid = torch.ones(P, 1, dtype=torch.bool, device=self_ctx.device)
            if self_ctx_history_mask is not None:
                full_mask = torch.cat([self_ctx_history_mask, current_valid], dim=1)  # (P, T)
            else:
                # 无 mask 时假定所有历史帧有效
                full_mask = torch.ones(P, T_prev + 1, dtype=torch.bool, device=self_ctx.device)
            self_ctx = self.temporal_encoder(full_seq, full_mask)  # (P, D)
        logits_self = self.stage1_scorer(self_ctx, e_self_pts)  # (P, N)

        coord_ctx, w_coord = self.coord_attn(
            q=e_self_pts,
            e_ally=e_ally,
            e_ally_pts=e_ally_pts,
            e_eally=e_eally,
            e_ast_a=e_ast_a,
            ally_mask=obs["ally_mask"],
            ally_pts_mask=obs["ally_pts_mask"],
            ally_enemy_mask=obs["ally_enemy_mask"],
        )
        delta = self.coord_head(e_self_pts, coord_ctx)  # (P, N)
        logits = logits_self + delta

        # 反事实信用分配：Q_i(s, a_i) - Σ_a π_i(a|o_i) Q_i(s, a)
        q_values = self.cf_q_head(self_ctx, e_self_pts)  # (P, N)
        masked_q = q_values.masked_fill(~obs["self_pts_mask"], 0.0)
        # 策略概率（与主 logits 同步 softmax）
        masked_logits_for_cf = logits.masked_fill(~obs["self_pts_mask"], -1e9)
        cf_probs = torch.softmax(masked_logits_for_cf, dim=-1)  # (P, N)
        # 反事实基线 b_i = Σ_a π_i(a) * Q_i(s, a)
        cf_baseline = (cf_probs * masked_q).sum(dim=-1)  # (P,)
        # 反事实优势 A_cf_i(a_i) = Q_i(s, a_i) - b_i
        # 在 actor 输出中不直接使用，留给 PPO 更新时与 GAE advantage 融合
        # 选取实际动作对应的 Q 值
        best_idx = torch.argmax(cf_probs, dim=-1)  # (P,)
        q_selected = q_values.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)  # (P,)
        cf_adv = q_selected - cf_baseline  # (P,)

        masked_logits, probs, best_candidate_idx = postprocess_mask_and_sample(
            logits=logits, self_pts_mask=obs["self_pts_mask"]
        )
        return {
            "action_logits": masked_logits,
            "action_probs": probs,
            "best_candidate_idx": best_candidate_idx,
            "coord_attn_weights": w_coord,
            "cf_adv": cf_adv,  # 反事实优势，PPO 更新时可用
            "q_values": q_values,  # 全部候选的 Q 值，PPO 更新时可用
            "cf_baseline": cf_baseline,  # 反事实基线
            "self_ctx": self_ctx.detach(),  # 当前步 self_ctx（供时序缓存收集）
        }

    def actor_forward(
        self,
        obs: Dict[str, torch.Tensor],
        self_ctx_history: Optional[torch.Tensor] = None,
        self_ctx_history_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.design_mode == "D":
            return self._actor_forward_d(obs, self_ctx_history=self_ctx_history, self_ctx_history_mask=self_ctx_history_mask)
        (e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a), ctx, attn_w = self._shared_ctx(obs)

        w_ctx = None
        if self.design_mode in ["A", "B"]:
            if self.design_mode == "A":
                assert self.fusion_a is not None
                h_env = self.fusion_a(ctx)
                gate_weights = None
            else:
                assert self.fusion_b is not None
                h_env, gate_weights = self.fusion_b(ctx)
            assert self.ptr is not None
            logits = self.ptr(h_env=h_env, e_self_pts=e_self_pts)
        else:
            assert self.points_ctx is not None and self.score_c is not None
            fused, w_ctx = self.points_ctx(
                q=e_self_pts,
                e_self=e_self,
                e_ally=e_ally,
                e_ally_pts=e_ally_pts,
                e_eself=e_eself,
                e_eally=e_eally,
                e_ast_s=e_ast_s,
                e_ast_a=e_ast_a,
                ally_mask=obs["ally_mask"],
                ally_pts_mask=obs["ally_pts_mask"],
                enemy_self_mask=obs["enemy_self_mask"].bool(),
                ally_enemy_mask=obs["ally_enemy_mask"],
            )
            logits = self.score_c(fused)
            gate_weights = None

        masked_logits, probs, best_candidate_idx = postprocess_mask_and_sample(
            logits=logits, self_pts_mask=obs["self_pts_mask"]
        )
        out: Dict[str, torch.Tensor] = {
            "action_logits": masked_logits,
            "action_probs": probs,
            "best_candidate_idx": best_candidate_idx,
            "attn_weights": attn_w,
        }
        if gate_weights is not None:
            out["gate_weights"] = gate_weights
        if w_ctx is not None:
            out["points_ctx_attn_weights"] = w_ctx
        return out

    def critic_forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # C/D 对齐：critic 都基于同源 attn_ab + fusion 表征，减少 policy/value 表征失配。
        embs = self._embed(obs)
        _, ctx, _attn_w = self._shared_ctx(obs, embs=embs)
        h_env = self._h_env(obs, ctx)  # (P, D)
        h_global = self.global_encoder(obs)  # (D,)
        return self.critic(h_env, h_global)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        self_ctx_history: Optional[torch.Tensor] = None,
        self_ctx_history_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.actor_forward(obs, self_ctx_history=self_ctx_history, self_ctx_history_mask=self_ctx_history_mask)
        out["value"] = self.critic_forward(obs)
        return out

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .actor import (
    CoordResidualHead,
    PointerActor,
    PointwiseScoringActor,
    SelfStageScorer,
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
        **_: Any,
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

        if self.design_mode == "A":
            self.fusion_a = ConcatMLPFusion8(self.hidden_dim)
            self.ptr = PointerActor(self.hidden_dim)
        elif self.design_mode == "B":
            self.fusion_b = SoftGatingFusion7(self.hidden_dim)
            self.ptr = PointerActor(self.hidden_dim)
        elif self.design_mode == "C":
            self.points_ctx = PointsContextCrossAttention(self.hidden_dim, self.num_heads)
            self.score_c = PointwiseScoringActor(self.hidden_dim, score_hidden_dim=self.hidden_dim * 2)
            # Critic for scheme C uses the same fused representation as A (concat over 8 branches).
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

        self.critic = PermInvariantCritic(embed_dim=self.hidden_dim, hidden_dim=self.hidden_dim)

    def _embed(self, obs: Dict[str, torch.Tensor]):
        """仅做实体编码，返回 8 个 e_*（不跑异构注意力）。"""
        return self.emb(obs)

    def _shared_ctx(self, obs: Dict[str, torch.Tensor]):
        e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = self._embed(obs)
        ctx, attn_w = self.attn_ab(
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
        return (e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a), ctx, attn_w

    def _h_env(self, obs: Dict[str, torch.Tensor], ctx) -> torch.Tensor:
        if self.design_mode == "A":
            assert self.fusion_a is not None
            return self.fusion_a(ctx)
        if self.design_mode == "B":
            assert self.fusion_b is not None
            h_env, _gate = self.fusion_b(ctx)
            return h_env
        assert self._c_fusion_for_critic is not None
        return self._c_fusion_for_critic(ctx)

    def _pool_embeddings(self, obs: Dict[str, torch.Tensor], embs) -> torch.Tensor:
        """对全部实体 token 做 masked mean，得每机 h_env (P, D)。设计 D 的 critic 用此廉价表征，避免重算异构注意力。"""
        e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = embs
        p = e_self.shape[0]
        ones1 = torch.ones((p, 1), device=e_self.device, dtype=torch.bool)
        toks = torch.cat([e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a], dim=1)
        masks = torch.cat(
            [
                ones1,
                obs["ally_mask"].bool(),
                obs["self_pts_mask"].bool(),
                obs["ally_pts_mask"].bool(),
                obs["enemy_self_mask"].bool(),
                obs["ally_enemy_mask"].bool(),
                obs["enemy_self_mask"].bool(),
                obs["ally_enemy_mask"].bool(),
            ],
            dim=1,
        ).float().unsqueeze(-1)
        s = (toks * masks).sum(dim=1)
        c = masks.sum(dim=1).clamp(min=1.0)
        return s / c

    def _actor_forward_d(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """设计 D：阶段1 自身打分 + 阶段2 友军协同残差。"""
        e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = self._embed(obs)
        assert self.self_ctx_fuse is not None and self.coord_attn is not None
        assert self.stage1_scorer is not None and self.coord_head is not None

        self_ctx = self.self_ctx_fuse(
            torch.cat([e_self.squeeze(1), e_eself.squeeze(1), e_ast_s.squeeze(1)], dim=-1)
        )  # (P, D)
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

        masked_logits, probs, best_candidate_idx = postprocess_mask_and_sample(
            logits=logits, self_pts_mask=obs["self_pts_mask"]
        )
        return {
            "action_logits": masked_logits,
            "action_probs": probs,
            "best_candidate_idx": best_candidate_idx,
            "coord_attn_weights": w_coord,
        }

    def actor_forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.design_mode == "D":
            return self._actor_forward_d(obs)
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
        if self.design_mode == "D":
            # 设计 D：用廉价池化表征作 h_env，不重算异构注意力
            h_env = self._pool_embeddings(obs, self._embed(obs))  # (P, D)
        else:
            _, ctx, _attn_w = self._shared_ctx(obs)
            h_env = self._h_env(obs, ctx)  # (P, D)
        h_global = self.global_encoder(obs)  # (D,)
        return self.critic(h_env, h_global)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.actor_forward(obs)
        out["value"] = self.critic_forward(obs)
        return out

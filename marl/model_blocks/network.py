from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .actor import PointerActor, PointwiseScoringActor, postprocess_mask_and_sample
from .attention import HeterogeneousAttentionAB, PointsContextCrossAttention
from .critic import CentralCritic
from .embeddings import TODCEmbeddings
from .fusion import ConcatMLPFusion8, SoftGatingFusion7
from .schemes import SCHEME_KEY_TO_NAME, resolve_design_mode


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
        **_: Any,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.design_mode = resolve_design_mode(design_mode, use_soft_gating=use_soft_gating)
        self.design_name = SCHEME_KEY_TO_NAME[self.design_mode]

        self.emb = TODCEmbeddings(self.hidden_dim)
        self.attn_ab = HeterogeneousAttentionAB(self.hidden_dim, self.num_heads)

        if self.design_mode == "A":
            self.fusion_a = ConcatMLPFusion8(self.hidden_dim)
            self.fusion_b = None
            self.ptr = PointerActor(self.hidden_dim)
            self.points_ctx = None
            self.score_c = None
        elif self.design_mode == "B":
            self.fusion_a = None
            self.fusion_b = SoftGatingFusion7(self.hidden_dim)
            self.ptr = PointerActor(self.hidden_dim)
            self.points_ctx = None
            self.score_c = None
        else:
            self.fusion_a = None
            self.fusion_b = None
            self.ptr = None
            self.points_ctx = PointsContextCrossAttention(self.hidden_dim, self.num_heads)
            self.score_c = PointwiseScoringActor(self.hidden_dim, score_hidden_dim=self.hidden_dim * 2)
            # Critic for scheme C uses the same fused representation as A (concat over 8 branches).
            self._c_fusion_for_critic = ConcatMLPFusion8(self.hidden_dim)

        self.critic = CentralCritic(embed_dim=self.hidden_dim, hidden_dim=self.hidden_dim)

    def _shared_ctx(self, obs: Dict[str, torch.Tensor]):
        e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = self.emb(obs)
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

    def actor_forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        _, ctx, _attn_w = self._shared_ctx(obs)
        o = self._h_env(obs, ctx)  # (B,D) but B is actually P in this codebase
        return self.critic(o)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.actor_forward(obs)
        out["value"] = self.critic_forward(obs)
        return out

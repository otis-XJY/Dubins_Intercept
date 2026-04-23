from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _key_padding_mask_from_valid_mask(valid_mask: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert 'valid_mask=True means valid' to MultiheadAttention key_padding_mask=True means padding.

    Returns:
      key_padding_mask: (B, L) with True meaning padding
      all_masked: (B,) True where all positions are masked (for safe handling)
    """
    if valid_mask is None:
        return None, None
    key_padding_mask = ~valid_mask.bool()
    all_masked = key_padding_mask.all(dim=-1)
    if all_masked.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked, :] = False
    return key_padding_mask, all_masked


class SingleBranchAttention(nn.Module):
    """Self(1) -> branch cross-attn with independent parameters."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        *,
        query: torch.Tensor,  # (B, 1, D)
        key: torch.Tensor,  # (B, L, D)
        value: torch.Tensor,  # (B, L, D)
        valid_mask: Optional[torch.Tensor] = None,  # (B, L) True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask, all_masked = _key_padding_mask_from_valid_mask(valid_mask)
        out, w = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        if all_masked is not None and all_masked.any():
            out = out.clone()
            out[all_masked] = 0.0
            w = w.clone()
            w[all_masked] = 0.0
        return out, w


@dataclass(frozen=True)
class HeteroContexts:
    e_self: torch.Tensor  # (B, D)
    h_ally: torch.Tensor  # (B, D)
    h_spts: torch.Tensor  # (B, D)
    h_apts: torch.Tensor  # (B, D)
    h_eself: torch.Tensor  # (B, D)
    h_eally: torch.Tensor  # (B, D)
    h_ast_s: torch.Tensor  # (B, D)
    h_ast_a: torch.Tensor  # (B, D)


class HeterogeneousAttentionAB(nn.Module):
    """A/B：e_self 对 7 路上下文并行 cross-attn。"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ally = SingleBranchAttention(embed_dim, num_heads)
        self.self_pts = SingleBranchAttention(embed_dim, num_heads)
        self.ally_pts = SingleBranchAttention(embed_dim, num_heads)
        self.enemy_self = SingleBranchAttention(embed_dim, num_heads)
        self.enemy_per_ally = SingleBranchAttention(embed_dim, num_heads)
        self.asset_self = SingleBranchAttention(embed_dim, num_heads)
        self.asset_per_ally = SingleBranchAttention(embed_dim, num_heads)

    def forward(
        self,
        *,
        e_self: torch.Tensor,  # (B,1,D)
        e_ally: torch.Tensor,  # (B,A,D)
        e_self_pts: torch.Tensor,  # (B,N,D)
        e_ally_pts: torch.Tensor,  # (B,Ma,D)
        e_eself: torch.Tensor,  # (B,1,D)
        e_eally: torch.Tensor,  # (B,A,D)
        e_ast_s: torch.Tensor,  # (B,1,D)
        e_ast_a: torch.Tensor,  # (B,A,D)
        ally_mask: torch.Tensor,  # (B,A) True=valid
        self_pts_mask: torch.Tensor,  # (B,N) True=valid
        ally_pts_mask: torch.Tensor,  # (B,Ma) True=valid
        enemy_self_mask: torch.Tensor,  # (B,1) True=valid
        ally_enemy_mask: torch.Tensor,  # (B,A) True=valid
    ) -> Tuple[HeteroContexts, Dict[str, torch.Tensor]]:
        h_ally, w_ally = self.ally(query=e_self, key=e_ally, value=e_ally, valid_mask=ally_mask)
        h_spts, w_spts = self.self_pts(query=e_self, key=e_self_pts, value=e_self_pts, valid_mask=self_pts_mask)
        h_apts, w_apts = self.ally_pts(query=e_self, key=e_ally_pts, value=e_ally_pts, valid_mask=ally_pts_mask)
        h_eself, w_eself = self.enemy_self(query=e_self, key=e_eself, value=e_eself, valid_mask=enemy_self_mask)
        h_eally, w_eally = self.enemy_per_ally(query=e_self, key=e_eally, value=e_eally, valid_mask=ally_enemy_mask)
        h_ast_s, w_ast_s = self.asset_self(query=e_self, key=e_ast_s, value=e_ast_s, valid_mask=enemy_self_mask)
        h_ast_a, w_ast_a = self.asset_per_ally(query=e_self, key=e_ast_a, value=e_ast_a, valid_mask=ally_enemy_mask)

        ctx = HeteroContexts(
            e_self=e_self.squeeze(1),
            h_ally=h_ally.squeeze(1),
            h_spts=h_spts.squeeze(1),
            h_apts=h_apts.squeeze(1),
            h_eself=h_eself.squeeze(1),
            h_eally=h_eally.squeeze(1),
            h_ast_s=h_ast_s.squeeze(1),
            h_ast_a=h_ast_a.squeeze(1),
        )
        weights = {
            "allies": w_ally,
            "self_points": w_spts,
            "ally_points": w_apts,
            "enemy_self": w_eself,
            "enemy_per_ally": w_eally,
            "asset_self": w_ast_s,
            "asset_per_ally": w_ast_a,
        }
        return ctx, weights


class PointsContextCrossAttention(nn.Module):
    """Design C: Q=E_points, K/V=E_context (concat of 8 types, excluding points from Q side)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        *,
        q: torch.Tensor,  # (B,N,D)
        e_self: torch.Tensor,  # (B,1,D)
        e_ally: torch.Tensor,  # (B,A,D)
        e_ally_pts: torch.Tensor,  # (B,Ma,D)
        e_eself: torch.Tensor,  # (B,1,D)
        e_eally: torch.Tensor,  # (B,A,D)
        e_ast_s: torch.Tensor,  # (B,1,D)
        e_ast_a: torch.Tensor,  # (B,A,D)
        ally_mask: torch.Tensor,
        ally_pts_mask: torch.Tensor,
        enemy_self_mask: torch.Tensor,
        ally_enemy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.cat([e_self, e_ally, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a], dim=1)

        pad_ally = ~ally_mask.bool()
        pad_ally_pts = ~ally_pts_mask.bool()
        pad_eself = ~enemy_self_mask.bool()
        pad_eally = ~ally_enemy_mask.bool()
        pad_ast_s = ~enemy_self_mask.bool()
        pad_ast_a = ~ally_enemy_mask.bool()

        key_padding_mask = torch.cat(
            [
                torch.zeros((ctx.shape[0], 1), device=ctx.device, dtype=torch.bool),  # e_self slot always present
                pad_ally,
                pad_ally_pts,
                pad_eself,
                pad_eally,
                pad_ast_s,
                pad_ast_a,
            ],
            dim=1,
        )

        all_masked = key_padding_mask.all(dim=-1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, :] = False

        out, w = self.attn(
            query=q,
            key=ctx,
            value=ctx,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        if all_masked.any():
            out = out.clone()
            out[all_masked] = 0.0
            w = w.clone()
            w[all_masked] = 0.0
        return out, w

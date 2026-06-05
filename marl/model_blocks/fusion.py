from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import HeteroContexts


def concat_ab_contexts(ctx: HeteroContexts) -> torch.Tensor:
    return torch.cat(
        [
            ctx.e_self,
            ctx.h_ally,
            ctx.h_spts,
            ctx.h_apts,
            ctx.h_eself,
            ctx.h_eally,
            ctx.h_ast_s,
            ctx.h_ast_a,
        ],
        dim=-1,
    )


class ConcatMLPFusion8(nn.Module):
    """Design A: Concat([e_self, h_1..h_7]) -> MLP -> LayerNorm."""

    def __init__(self, embed_dim: int):
        super().__init__()
        d = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(d * 8, d * 4),
            nn.ReLU(),
            nn.Linear(d * 4, d),
            nn.LayerNorm(d),
        )

    def forward(self, ctx: HeteroContexts) -> torch.Tensor:
        x = concat_ab_contexts(ctx)
        return self.net(x)


class SoftGatingFusion7(nn.Module):
    """Design B: dot-product gates over 7 branches + residual + LayerNorm."""

    def __init__(self, embed_dim: int):
        super().__init__()
        d = int(embed_dim)
        self.ln = nn.LayerNorm(d)
        self.scale = 1.0 / math.sqrt(float(d))

    def forward(self, ctx: HeteroContexts) -> tuple[torch.Tensor, torch.Tensor]:
        e = ctx.e_self
        hs = [ctx.h_ally, ctx.h_spts, ctx.h_apts, ctx.h_eself, ctx.h_eally, ctx.h_ast_s, ctx.h_ast_a]
        scores = torch.stack([(e * hi).sum(dim=-1) * self.scale for hi in hs], dim=-1)  # (B,7)
        gate = torch.softmax(scores, dim=-1)  # (B,7)
        mix = torch.zeros_like(e)
        for i, hi in enumerate(hs):
            mix = mix + gate[:, i : i + 1] * hi
        h_env = self.ln(e + mix)
        return h_env, gate

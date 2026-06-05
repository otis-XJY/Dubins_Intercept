from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class MLPActor(nn.Module):
    """Linear(D,D) -> ReLU -> Linear(D,D) (no LayerNorm on output)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        d = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointerActor(nn.Module):
    """Design A/B: q(h_env) dot k(W_k * points)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        d = int(embed_dim)
        self.mlp_actor = MLPActor(d)
        self.W_K_action = nn.Linear(d, d, bias=False)

    def forward(self, *, h_env: torch.Tensor, e_self_pts: torch.Tensor) -> torch.Tensor:
        q_action = self.mlp_actor(h_env).unsqueeze(1)  # (B,1,D)
        keys = self.W_K_action(e_self_pts)  # (B,N,D)
        logits = torch.bmm(q_action, keys.transpose(1, 2)).squeeze(1) / math.sqrt(float(keys.shape[-1]))
        return logits


class PointwiseScoringActor(nn.Module):
    """Design C: per-point MLP over fused point representation."""

    def __init__(self, embed_dim: int, score_hidden_dim: int):
        super().__init__()
        d = int(embed_dim)
        h = int(score_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class SelfStageScorer(nn.Module):
    """Design D 阶段1：由自身上下文 self_ctx 对每个自候选点逐点打分，得 logits_self。"""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        d = int(embed_dim)
        h = int(hidden_dim) if hidden_dim is not None else d
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, self_ctx: torch.Tensor, e_self_pts: torch.Tensor) -> torch.Tensor:
        # self_ctx (B,D)；e_self_pts (B,N,D) -> logits (B,N)
        n = e_self_pts.shape[1]
        ctx = self_ctx.unsqueeze(1).expand(-1, n, -1)
        x = torch.cat([e_self_pts, ctx], dim=-1)
        return self.mlp(x).squeeze(-1)


class CoordResidualHead(nn.Module):
    """Design D 阶段2：由候选点的协同上下文产生残差 delta（最后一层 zero-init，初始 delta≈0）。"""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        d = int(embed_dim)
        h = int(hidden_dim) if hidden_dim is not None else d
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        # 残差头 zero-init：训练起点为纯自身策略，协同修正从 0 学起（幅度不受约束）
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, e_self_pts: torch.Tensor, coord_ctx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e_self_pts, coord_ctx], dim=-1)
        return self.mlp(x).squeeze(-1)


def postprocess_mask_and_sample(
    *,
    logits: torch.Tensor,  # (B,N)
    self_pts_mask: torch.Tensor,  # (B,N) True=valid
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """保持现网 masking + 全无可用点退化策略。"""
    masked_logits = logits.masked_fill(~self_pts_mask, -1e9)
    active = self_pts_mask.any(dim=-1)
    inactive = ~active
    if inactive.any():
        masked_logits = masked_logits.clone()
        masked_logits[inactive] = 0.0
    probs = torch.softmax(masked_logits, dim=-1)
    if inactive.any():
        probs = probs.clone()
        probs[inactive] = 0.0
        probs[inactive, 0] = 1.0
    best_candidate_idx = torch.argmax(probs, dim=-1).to(dtype=torch.int64)
    best_candidate_idx = best_candidate_idx.masked_fill(inactive, -1)
    return masked_logits, probs, best_candidate_idx

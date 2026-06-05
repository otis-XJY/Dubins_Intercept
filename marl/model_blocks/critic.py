from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PermInvariantCritic(nn.Module):
    """置换不变集中 critic（CTDE）。

    输入每机 h_env(P,D) 与全局 h_global(D)：
      - 跨机用可学习 query 的 attention pooling 聚合成全局摘要 g(D)，对 agent 顺序不变；
      - 每机 value = value_head(cat(h_env_i, g, h_global))，输出 (P,)。

    所有参数在 __init__ 静态构建，维度与 P 无关：保证被优化器跟踪、P 变化不重建、DDP 友好。
    （取代旧 CentralCritic 的 concat-over-P 懒加载：那种写法因 critic 网络在首次前向才创建、
    晚于优化器绑定 model.parameters()，导致 critic 参数从不被更新。）
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        d = int(embed_dim)
        h = int(hidden_dim) if hidden_dim is not None else d
        self.embed_dim = d
        self.scale = 1.0 / math.sqrt(float(d))

        # 跨机 attention pooling：可学习 query 对 P 个 agent token 打分
        self.pool_query = nn.Parameter(torch.randn(d) * 0.02)
        self.pool_key = nn.Linear(d, d)
        self.pool_val = nn.Linear(d, d)

        # 每机 value 头：输入 cat(h_env_i, g, h_global) = 3D
        self.value_head = nn.Sequential(
            nn.Linear(d * 3, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, h_env: torch.Tensor, h_global: torch.Tensor) -> torch.Tensor:
        """h_env: (P, D)；h_global: (D,)。返回 (P,)。"""
        keys = self.pool_key(h_env)                       # (P, D)
        vals = self.pool_val(h_env)                       # (P, D)
        scores = (keys @ self.pool_query) * self.scale    # (P,)
        attn = torch.softmax(scores, dim=0)               # (P,)
        g = attn @ vals                                   # (D,)

        p = h_env.shape[0]
        g_exp = g.unsqueeze(0).expand(p, -1)              # (P, D)
        hg_exp = h_global.unsqueeze(0).expand(p, -1)      # (P, D)
        combined = torch.cat([h_env, g_exp, hg_exp], dim=-1)  # (P, 3D)
        return self.value_head(combined).squeeze(-1)      # (P,)

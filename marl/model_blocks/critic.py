from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CentralCritic(nn.Module):
    """集中 critic：输入所有 agent 的 o(=h_env) 拼接，输出每机 value。

    保持现有“按 P 动态重建”的策略：P 变化时重建 MLP 的输入/输出维度。
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else int(embed_dim)
        self._critic_net: Optional[nn.Module] = None
        self._critic_in_dim: Optional[int] = None

    def _ensure(self, p: int, per_agent_dim: int, device: torch.device) -> None:
        in_dim = p * per_agent_dim
        if self._critic_net is None or self._critic_in_dim != in_dim:
            self._critic_in_dim = in_dim
            self._critic_net = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, p),
            ).to(device)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """Args:
        o: (P, D) or (P, 2D) per-agent fused representations (optionally with global state).
        """
        device = o.device
        p = int(o.shape[0])
        d = int(o.shape[1])
        self._ensure(p, d, device)
        central = o.reshape(1, -1)
        assert self._critic_net is not None
        values = self._critic_net(central).squeeze(0)
        return values

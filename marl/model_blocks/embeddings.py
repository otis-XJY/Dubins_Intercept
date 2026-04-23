from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(max(1, num_layers - 1)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TODCEmbeddings(nn.Module):
    """8 路观测编码（保持现有 obs 键与 shape 不变）。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.enc_self = _mlp(3, hidden_dim, hidden_dim)
        self.enc_ally = _mlp(3, hidden_dim, hidden_dim)
        self.enc_self_pts = _mlp(8, hidden_dim, hidden_dim)
        self.enc_ally_pts = _mlp(8, hidden_dim, hidden_dim)
        self.enc_enemy = _mlp(3, hidden_dim, hidden_dim)
        self.enc_asset = _mlp(2, hidden_dim, hidden_dim)

    def forward(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e_self = self.enc_self(obs["self_uav"])
        e_ally = self.enc_ally(obs["allies_local"])
        e_self_pts = self.enc_self_pts(obs["self_pts"])
        e_ally_pts = self.enc_ally_pts(obs["ally_pts"])
        e_eself = self.enc_enemy(obs["enemy_assigned_self"])
        e_eally = self.enc_enemy(obs["enemy_assigned_per_ally"])
        e_ast_s = self.enc_asset(obs["asset_target_self"])
        e_ast_a = self.enc_asset(obs["asset_target_per_ally"])
        return e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from marl.nn.blocks.network import UAVInterceptionNetwork
from marl.nn.blocks.schemes import SCHEME_KEY_TO_NAME, design_mode_to_name, resolve_design_mode


def build_actor_critic_schemes(
    schemes: Optional[Union[str, Sequence[str]]] = None,
    *,
    hidden_dim: int = 128,
    num_heads: int = 4,
    pos_scale: float = 1000.0,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, "TODCHeteroActorCritic"]:
    if schemes is None:
        requested: List[str] = ["A", "B", "C"]
    elif isinstance(schemes, str):
        requested = [schemes]
    else:
        requested = list(schemes)

    unique_keys: List[str] = []
    seen = set()
    for item in requested:
        key = resolve_design_mode(item)
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)

    models: Dict[str, TODCHeteroActorCritic] = {}
    for key in unique_keys:
        name = SCHEME_KEY_TO_NAME[key]
        model = TODCHeteroActorCritic(
            hidden_dim=hidden_dim,
            design_mode=key,
            num_heads=num_heads,
            pos_scale=pos_scale,
        )
        if device is not None:
            model = model.to(device)
        models[name] = model
    return models


class TODCHeteroActorCritic(nn.Module):
    """Wrapper: actor head + MAPPO centralized critic."""

    def __init__(
        self,
        hidden_dim: int = 128,
        design_mode: Optional[str] = None,
        num_heads: int = 4,
        use_soft_gating: Optional[bool] = None,
        pos_scale: float = 1000.0,
    ):
        super().__init__()
        self.model = UAVInterceptionNetwork(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            design_mode=design_mode,
            use_soft_gating=use_soft_gating,
            pos_scale=pos_scale,
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(obs)

    def actor_forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model.actor_forward(obs)

    def critic_forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.critic_forward(obs)


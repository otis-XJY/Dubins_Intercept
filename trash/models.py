"""兼容入口：请优先使用 ``marl.nn.models``。"""
from marl.nn.models import (
    SCHEME_KEY_TO_NAME,
    TODCHeteroActorCritic,
    UAVInterceptionNetwork,
    build_actor_critic_schemes,
    design_mode_to_name,
    resolve_design_mode,
)

__all__ = [
    "SCHEME_KEY_TO_NAME",
    "TODCHeteroActorCritic",
    "UAVInterceptionNetwork",
    "build_actor_critic_schemes",
    "design_mode_to_name",
    "resolve_design_mode",
]

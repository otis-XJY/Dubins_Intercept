from .envs import TODCMARLEnv
from .obs import TODCObservationGenerator, alive_pid_eid_sets
from .rewards import RewardConfig, TODCRewardFunction

# 注意：marl.nn 依赖 torch。为了支持“只使用环境/观测/奖励”的场景（无需 torch），
# 这里对 nn 相关导入做可选处理：未安装 torch 时仍可 import marl 并创建 TODCMARLEnv。
try:
    from .nn import (
        SCHEME_KEY_TO_NAME,
        TODCHeteroActorCritic,
        UAVInterceptionNetwork,
        build_actor_critic_schemes,
        design_mode_to_name,
        resolve_design_mode,
    )
except ModuleNotFoundError as e:
    if getattr(e, "name", None) != "torch":
        raise

    SCHEME_KEY_TO_NAME = None  # type: ignore
    TODCHeteroActorCritic = None  # type: ignore
    UAVInterceptionNetwork = None  # type: ignore

    def _torch_required(*_args, **_kwargs):
        raise ModuleNotFoundError("torch 未安装：nn/训练相关接口不可用，但环境可正常使用。")

    build_actor_critic_schemes = _torch_required  # type: ignore
    design_mode_to_name = _torch_required  # type: ignore
    resolve_design_mode = _torch_required  # type: ignore

__all__ = [
    "TODCMARLEnv",
    "TODCObservationGenerator",
    "alive_pid_eid_sets",
    "TODCHeteroActorCritic",
    "UAVInterceptionNetwork",
    "SCHEME_KEY_TO_NAME",
    "resolve_design_mode",
    "design_mode_to_name",
    "build_actor_critic_schemes",
    "RewardConfig",
    "TODCRewardFunction",
]

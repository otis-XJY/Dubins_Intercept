"""Dubins 拦截多智能体强化学习包（marl0327）。"""

from marl0327.config import MARLConfig
from marl0327.env import DubinsInterceptEnvConfig, DubinsInterceptMARLEnv

__all__ = ["MARLConfig", "DubinsInterceptEnvConfig", "DubinsInterceptMARLEnv", "__version__"]

__version__ = "0.1.0"

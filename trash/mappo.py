"""兼容入口：请优先使用 ``marl.rl.mappo``。"""
from marl.rl.mappo import compute_gae, ppo_minibatch_update

__all__ = ["compute_gae", "ppo_minibatch_update"]

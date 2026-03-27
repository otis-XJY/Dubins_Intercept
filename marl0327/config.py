"""训练与仿真占位配置（可按实验修改）。"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MARLConfig:
    # 网络（与 models.UAVInterceptionNetwork 一致）
    hidden_dim: int = 128
    num_heads: int = 4
    design_mode: str = "A"

    # 序列长度上界（padding）
    max_allies: int = 8
    max_enemies: int = 8
    max_targets: int = 32
    max_self_pts: int = 64
    max_ally_pts: int = 64

    # PPO / 优化
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    train_epochs: int = 4
    minibatch_size: int = 256

    # 分布式
    seed: int = 0
    local_rank: Optional[int] = None

#!/usr/bin/env python3
"""
MARL 训练入口占位：多卡 DDP 自检 + 与 models 的观测维度假 forward。

运行示例（单机多卡）:
  conda activate dubins
  torchrun --nproc_per_node=2 marl0327/train_marl.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from marl0327.config import MARLConfig
from marl0327.models import UAVInterceptionNetwork
from marl0327.observation import numpy_obs_to_torch


def _setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def _fake_batch(B: int, cfg: MARLConfig, device: torch.device) -> dict:
    """构造与网络一致的随机观测，用于形状检查。"""
    return {
        "self_uav": torch.randn(B, 1, 3, device=device),
        "ally_uavs": torch.randn(B, cfg.max_allies, 3, device=device),
        "self_pts": torch.randn(B, cfg.max_self_pts, 8, device=device),
        "ally_pts": torch.randn(B, cfg.max_ally_pts, 8, device=device),
        "enemies": torch.randn(B, cfg.max_enemies, 3, device=device),
        "targets": torch.randn(B, cfg.max_targets, 2, device=device),
        "ally_mask": torch.ones(B, cfg.max_allies, dtype=torch.bool, device=device),
        "self_pts_mask": torch.ones(B, cfg.max_self_pts, dtype=torch.bool, device=device),
        "ally_pts_mask": torch.ones(B, cfg.max_ally_pts, dtype=torch.bool, device=device),
        "enemy_mask": torch.ones(B, cfg.max_enemies, dtype=torch.bool, device=device),
        "target_mask": torch.ones(B, cfg.max_targets, dtype=torch.bool, device=device),
    }


def main() -> None:
    rank, world_size, device = _setup_distributed()
    cfg = MARLConfig()

    model = UAVInterceptionNetwork(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        design_mode=cfg.design_mode,
    ).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )

    B = 4
    batch = _fake_batch(B, cfg, device)
    mod = model.module if hasattr(model, "module") else model
    out = mod(batch)
    assert out["action_logits"].shape == (B, cfg.max_self_pts)
    assert out["value"].shape == (B,)

    if rank == 0:
        print("MARL 模型前向形状检查通过:", {k: tuple(v.shape) for k, v in out.items()})
        # 演示 numpy_obs_to_torch 路径（单样本复制 B 份）
        import numpy as np

        from marl0327.observation import encode_single_pursuer_obs

        fake_decision = {
            "output_P_state": {0: np.zeros(3, dtype=np.float32)},
            "output_P_alley_state": {0: {"states": np.zeros((0, 3), dtype=np.float32)}},
            "output_enemy_state": {0: {"states": np.zeros((1, 3), dtype=np.float32)}},
            "output_V_state": {0: {"states": np.zeros((2, 2), dtype=np.float32)}},
        }
        ic = np.zeros((0, 20), dtype=np.float32)
        iso = np.zeros((0, 8), dtype=np.float32)
        obs0 = encode_single_pursuer_obs(
            0,
            fake_decision,
            ic,
            iso,
            (cfg.max_allies, cfg.max_enemies, cfg.max_targets, cfg.max_self_pts, cfg.max_ally_pts),
        )
        stacked = numpy_obs_to_torch([obs0] * B, device)
        out2 = mod(stacked)
        print("占位观测编码路径 OK, value mean:", float(out2["value"].mean().item()))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""时序编码器：多帧滑窗 + Transformer 聚合，用于 P2-1 时序输入。

将每步的 per-agent 表征缓存 T 帧，用可学习时间位置编码 + 1-2 层 Transformer encoder
聚合成当前步的时序感知表示，再送入 D 主干。

T=1 时退化为恒等映射（不破坏旧调用）。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """多帧时序注意力编码器。

    输入：(B, T, D) 的时序 token 序列（T 帧滑窗）
    输出：(B, D) 的当前步聚合表示
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        max_temporal_window: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_temporal_window = int(max_temporal_window)

        # 可学习时间位置编码（最远 max_temporal_window 帧）
        self.time_pos = nn.Parameter(torch.randn(1, self.max_temporal_window, self.embed_dim) * 0.02)

        # Transformer encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 取最后一帧（当前步）的输出作为聚合表示
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """输入 x: (B, T, D)；temporal_mask: (B, T) True=valid。

        返回 (B, D)：当前步的时序聚合表示。
        """
        t = x.shape[1]
        if t > self.max_temporal_window:
            # 超过最大窗口时截断到最近 max_temporal_window 帧
            x = x[:, -self.max_temporal_window:, :]
            if temporal_mask is not None:
                temporal_mask = temporal_mask[:, -self.max_temporal_window:]
            t = self.max_temporal_window

        # 加时间位置编码
        pos = self.time_pos[:, :t, :]
        x = x + pos

        # 构建 key_padding_mask（True = padding/无效）
        if temporal_mask is not None:
            key_padding_mask = ~temporal_mask.bool()  # (B, T)
        else:
            key_padding_mask = None

        # Transformer encoder
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, T, D)

        # 取最后一帧（当前步）作为聚合表示
        result = self.norm(out[:, -1, :])  # (B, D)
        return result

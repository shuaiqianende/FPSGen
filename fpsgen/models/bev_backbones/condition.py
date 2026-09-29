"""Zero-preserving FPSGen spatial-condition adapters for dense backbones."""

from __future__ import annotations

import torch
import torch.nn as nn


def _require_bchw(raw_pc: torch.Tensor, layout: torch.Tensor, channels: int) -> torch.Tensor:
    if raw_pc.ndim != 4 or layout.ndim != 4:
        raise ValueError("raw_pc and layout must both have BCHW layout")
    if raw_pc.shape[0] != layout.shape[0] or raw_pc.shape[-2:] != layout.shape[-2:]:
        raise ValueError("raw_pc/layout batch and spatial dimensions must match")
    condition = torch.cat((raw_pc, layout), dim=1)
    if condition.shape[1] != channels:
        raise ValueError(f"Expected {channels} condition channels, got {condition.shape[1]}")
    return condition


class DiCConditionEncoder(nn.Module):
    """A bias-free three-level condition pyramid for DiC stage injection.

    Every operation is linear and bias-free, so exact zero conditions remain
    exact zero before the learned stage gates are applied.
    """

    def __init__(self, lidar_channels: int = 32, layout_channels: int = 2,
                 hidden_size: int = 96):
        super().__init__()
        self.in_channels = int(lidar_channels + layout_channels)
        self.hidden_size = int(hidden_size)
        self.level0 = nn.Conv2d(self.in_channels, hidden_size, 3, padding=1, bias=False)
        self.level1 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2 // 4, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size * 4 // 4, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, raw_pc: torch.Tensor, layout: torch.Tensor):
        x = _require_bchw(raw_pc, layout, self.in_channels)
        c0 = self.level0(x)
        c1 = self.level1(c0)
        c2 = self.level2(c1)
        return c0, c1, c2


class PixelUConditionEncoder(nn.Module):
    """Bias-free patch and in-context condition-token construction for PixelU."""

    def __init__(self, lidar_channels: int = 32, layout_channels: int = 2,
                 patch_size: int = 16, bottleneck_dim: int = 64,
                 hidden_size: int = 384, context_tokens: int = 32):
        super().__init__()
        self.in_channels = int(lidar_channels + layout_channels)
        self.patch_size = int(patch_size)
        self.context_tokens = int(context_tokens)
        self.patch_proj1 = nn.Conv2d(
            self.in_channels, int(bottleneck_dim), kernel_size=patch_size,
            stride=patch_size, bias=False,
        )
        self.patch_proj2 = nn.Conv2d(int(bottleneck_dim), int(hidden_size), 1, bias=False)
        self.context_proj = nn.Linear(int(hidden_size), int(hidden_size), bias=False)

    def forward(self, raw_pc: torch.Tensor, layout: torch.Tensor):
        x = _require_bchw(raw_pc, layout, self.in_channels)
        patch = self.patch_proj2(self.patch_proj1(x)).flatten(2).transpose(1, 2)
        # Adaptive average pooling changes only token cardinality; no bias or
        # learned constant is introduced when every condition is inactive.
        context = torch.nn.functional.adaptive_avg_pool1d(
            patch.transpose(1, 2), self.context_tokens
        ).transpose(1, 2)
        return patch, self.context_proj(context)

"""PixelControl-style nonlinear patch controller with zero-initialized paths."""
from __future__ import annotations

import torch
import torch.nn as nn


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class PixelControlEncoder(nn.Module):
    """34ch map -> nonlinear 16x16 patch-aligned control tokens."""
    def __init__(self, hidden_size: int, base_channels: int = 16, max_channels: int = 128):
        super().__init__()
        widths = [base_channels, min(base_channels * 2, max_channels),
                  min(base_channels * 4, max_channels), max_channels]
        layers = []
        previous = 34
        for width in widths:
            layers += [nn.Conv2d(previous, width, 3, stride=2, padding=1, bias=False),
                       nn.GroupNorm(_groups(width), width, affine=False), nn.SiLU()]
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.out = nn.Conv2d(previous, hidden_size, 1, bias=False)

    def forward(self, condition_map: torch.Tensor) -> torch.Tensor:
        tokens = self.out(self.encoder(condition_map))
        if tokens.shape[-2:] != (16, 16):
            raise ValueError("PixelControl encoder requires a 256x256 condition map")
        return tokens.flatten(2).transpose(1, 2)


class PixelControlAdapter(nn.Module):
    """Per-Patch-DiT residual adapter; projection starts at exact zero."""
    def __init__(self, hidden_size: int, depth: int, gate_init: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.projections = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(depth))
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.gates = nn.Parameter(torch.full((depth,), float(gate_init)))
        self.last_ratios = {}

    def residual(self, layer: int, condition_tokens: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        residual = self.gates[layer].to(feature.dtype) * self.projections[layer](self.norm(condition_tokens))
        self.last_ratios[f"adapter_ratio_{layer}"] = (
            residual.detach().float().square().mean().sqrt() /
            (feature.detach().float().square().mean().sqrt() + 1e-8)
        )
        return residual

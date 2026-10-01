"""Bias-free lightweight SPADE for exact condition-off independence."""
from __future__ import annotations

import torch
import torch.nn as nn


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class LiteSPADE(nn.Module):
    """GN(h) * (1 + gamma(C)) + beta(C), with zero-preserving projections."""
    def __init__(self, channels: int, condition_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, affine=True)
        self.gamma = nn.Conv2d(condition_channels, channels, 3, padding=1, bias=False)
        self.beta = nn.Conv2d(condition_channels, channels, 3, padding=1, bias=False)
        self.last_gamma_rms = torch.tensor(0.)
        self.last_beta_rms = torch.tensor(0.)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.gamma(condition), self.beta(condition)
        self.last_gamma_rms = gamma.detach().float().square().mean().sqrt()
        self.last_beta_rms = beta.detach().float().square().mean().sqrt()
        return self.norm(x) * (1.0 + gamma) + beta

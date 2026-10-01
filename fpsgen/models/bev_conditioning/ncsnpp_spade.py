"""Neutral, lightweight SPADE adapters for existing NCSN++ GroupNorm sites."""
from __future__ import annotations

import torch
import torch.nn as nn


class LiteSPADEGroupNorm(nn.Module):
    """Use a supplied NCSN++ GroupNorm, then identity-plus spatial modulation.

    The adapter owns no normalization weights: its ``base_norm`` is the
    original BigGAN residual-block norm.  Zero gamma/beta heads consequently
    reproduce that norm exactly at initialization.
    """
    def __init__(self, condition_channels: int, output_channels: int, cond_dim: int = 8):
        super().__init__()
        self.project = nn.Conv2d(condition_channels, cond_dim, 1, bias=False)
        self.depthwise = nn.Conv2d(cond_dim, cond_dim, 3, padding=1, groups=cond_dim, bias=False)
        self.gamma = nn.Conv2d(cond_dim, output_channels, 1)
        self.beta = nn.Conv2d(cond_dim, output_channels, 1)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
        self.last_gamma_rms = torch.tensor(0.)
        self.last_beta_rms = torch.tensor(0.)
        self.last_effect_ratio = torch.tensor(0.)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, base_norm: nn.Module) -> torch.Tensor:
        normalized = base_norm(x)
        hidden = self.depthwise(torch.nn.functional.silu(self.project(condition)))
        gamma, beta = self.gamma(hidden), self.beta(hidden)
        effect = gamma * normalized + beta
        self.last_gamma_rms = gamma.detach().float().square().mean().sqrt()
        self.last_beta_rms = beta.detach().float().square().mean().sqrt()
        self.last_effect_ratio = effect.detach().float().square().mean().sqrt() / (normalized.detach().float().square().mean().sqrt() + 1e-8)
        return normalized + effect


class NCSNppSPADEPair(nn.Module):
    """One adapter for each of the two GroupNorm sites in a BigGAN block."""
    def __init__(self, condition_channels: int, norm1_channels: int, norm2_channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = LiteSPADEGroupNorm(condition_channels, norm1_channels, cond_dim)
        self.norm2 = LiteSPADEGroupNorm(condition_channels, norm2_channels, cond_dim)

    def diagnostics(self, prefix: str):
        return {
            f"gamma_{prefix}_1": self.norm1.last_gamma_rms.detach(),
            f"beta_{prefix}_1": self.norm1.last_beta_rms.detach(),
            f"ratio_{prefix}_1": self.norm1.last_effect_ratio.detach(),
            f"gamma_{prefix}_2": self.norm2.last_gamma_rms.detach(),
            f"beta_{prefix}_2": self.norm2.last_beta_rms.detach(),
            f"ratio_{prefix}_2": self.norm2.last_effect_ratio.detach(),
        }

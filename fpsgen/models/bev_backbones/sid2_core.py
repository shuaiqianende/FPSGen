"""Paper-based SiD2-style Residual U-ViT core for raw BEV velocity fields.

This is deliberately a pixel-space Residual U-ViT, not a concatenative U-Net:
every scale has one level-wise residual skip ``up(low - down) + high``.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return max(group for group in (32, 16, 8, 4, 2, 1) if channels % group == 0)


class ContinuousTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU(), nn.Linear(dim, dim, bias=False))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        embedding = torch.cat((torch.sin(t[:, None] * freq), torch.cos(t[:, None] * freq)), dim=-1)
        return self.mlp(F.pad(embedding, (0, self.dim - embedding.shape[-1])))


class AdaResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.mod = nn.Linear(time_dim, channels * 2, bias=False)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(time).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(F.silu(h))
        return x + self.conv2(F.silu(self.norm2(h)))


class AdaTransformerBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, head_dim: int):
        super().__init__()
        if channels % head_dim:
            raise ValueError("SiD2 transformer width must divide head_dim")
        self.channels, self.heads = int(channels), channels // head_dim
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 4, bias=False), nn.SiLU(), nn.Linear(channels * 4, channels, bias=False))
        self.mod = nn.Linear(time_dim, channels * 4, bias=False)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.ff[-1].weight)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape; tokens = x.flatten(2).transpose(1, 2)
        s1, b1, s2, b2 = self.mod(time).chunk(4, dim=-1)
        qkv = self.qkv(self.norm1(tokens) * (1 + s1[:, None]) + b1[:, None])
        q, k, v = qkv.chunk(3, dim=-1)
        def heads(value): return value.view(b, h * w, self.heads, c // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v)).transpose(1, 2).reshape(b, h * w, c)
        tokens = tokens + self.proj(attended)
        tokens = tokens + self.ff(self.norm2(tokens) * (1 + s2[:, None]) + b2[:, None])
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, source: int, target: int):
        super().__init__(); self.proj = nn.Conv2d(source, target, 1, bias=False)
    def forward(self, x): return self.proj(F.avg_pool2d(x, 2))


class Upsample(nn.Module):
    def __init__(self, source: int, target: int):
        super().__init__(); self.proj = nn.Conv2d(source, target, 1, bias=False)
    def forward(self, x): return F.interpolate(self.proj(x), scale_factor=2, mode="nearest")


class SiD2Core(nn.Module):
    """Small four-level E3-D3 Residual U-ViT with level-wise skips only."""
    def __init__(self, input_size=256, patch_size=2, in_channels=3, channels=(64, 128, 256, 384),
                 num_updown_blocks=(3, 3, 3), num_mid_blocks=16, head_dim=64, time_dim=256):
        super().__init__()
        self.input_size, self.patch_size = int(input_size), int(patch_size)
        self.channels = tuple(map(int, channels)); self.num_updown_blocks = tuple(map(int, num_updown_blocks))
        self.num_mid_blocks, self.head_dim, self.time_dim = int(num_mid_blocks), int(head_dim), int(time_dim)
        if self.patch_size != 2 or len(self.channels) != 4 or len(self.num_updown_blocks) != 3:
            raise ValueError("SiD2-S-BEV requires patch2, four levels, and E3-D3 depths")
        self.time = ContinuousTime(time_dim)
        self.input_proj = nn.Conv2d(in_channels * patch_size * patch_size, self.channels[0], 1, bias=False)
        self.enc0 = nn.ModuleList(AdaResBlock(self.channels[0], time_dim) for _ in range(self.num_updown_blocks[0]))
        self.enc1 = nn.ModuleList(AdaResBlock(self.channels[1], time_dim) for _ in range(self.num_updown_blocks[1]))
        self.enc2 = nn.ModuleList(AdaTransformerBlock(self.channels[2], time_dim, head_dim) for _ in range(self.num_updown_blocks[2]))
        self.mid = nn.ModuleList(AdaTransformerBlock(self.channels[3], time_dim, head_dim) for _ in range(self.num_mid_blocks))
        self.down0, self.down1, self.down2 = Downsample(self.channels[0], self.channels[1]), Downsample(self.channels[1], self.channels[2]), Downsample(self.channels[2], self.channels[3])
        self.up2, self.up1, self.up0 = Upsample(self.channels[3], self.channels[2]), Upsample(self.channels[2], self.channels[1]), Upsample(self.channels[1], self.channels[0])
        self.dec2 = nn.ModuleList(AdaTransformerBlock(self.channels[2], time_dim, head_dim) for _ in range(self.num_updown_blocks[2]))
        self.dec1 = nn.ModuleList(AdaResBlock(self.channels[1], time_dim) for _ in range(self.num_updown_blocks[1]))
        self.dec0 = nn.ModuleList(AdaResBlock(self.channels[0], time_dim) for _ in range(self.num_updown_blocks[0]))
        self.output_proj = nn.Conv2d(self.channels[0], in_channels * patch_size * patch_size, 1)
        nn.init.zeros_(self.output_proj.weight); nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _run(blocks, value, time):
        for block in blocks: value = block(value, time)
        return value

    def forward(self, x, t, spatial_conditions=None, level_conditions=None):
        if x.shape[-2:] != (self.input_size, self.input_size): raise ValueError("SiD2 input size mismatch")
        time = self.time(t)
        levels = [time if level_conditions is None else time + value for value in level_conditions] if level_conditions is not None else [time] * 4
        cond = spatial_conditions if spatial_conditions is not None else (None,) * 4
        def add(value, condition): return value if condition is None else value + condition.to(value.dtype)
        h0 = self._run(self.enc0, add(self.input_proj(F.pixel_unshuffle(x, 2)), cond[0]), levels[0])
        d0 = self.down0(h0)
        h1 = self._run(self.enc1, add(d0, cond[1]), levels[1])
        d1 = self.down1(h1)
        h2 = self._run(self.enc2, add(d1, cond[2]), levels[2])
        d2 = self.down2(h2)
        low = self._run(self.mid, add(d2, cond[3]), levels[3])
        # The decoder receives the same resolution-specific condition as its
        # paired encoder.  This remains a *level-wise residual* U-ViT skip:
        # no encoder-block activations are concatenated or retained.
        y2 = self._run(self.dec2, add(h2 + self.up2(low - d2), cond[2]), levels[2])
        y1 = self._run(self.dec1, add(h1 + self.up1(y2 - d1), cond[1]), levels[1])
        y0 = self._run(self.dec0, add(h0 + self.up0(y1 - d0), cond[0]), levels[0])
        return F.pixel_shuffle(self.output_proj(y0), 2)

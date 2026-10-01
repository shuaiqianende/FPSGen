"""HDiT-S pixel-space core for FPSGen BEV Flow Matching.

Architecture follows the shifted-window hierarchy published in
``crowsonkb/k-diffusion`` commit ``4601bf085320592473f681a62808ed873d17fad5``,
``k_diffusion/models/image_transformer_v2.py``.  This is a compact
architectural reimplementation: FPSGen changes only the image channels,
time-input API and adds its condition adapter in :mod:`hdit_bev`.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition import _require_bchw, modality_norms_and_ratios


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight.to(x.dtype)


class FourierFeatures(nn.Module):
    """Fixed Fourier features for FPSGen's continuous FM time ``t in [0, 1]``."""

    def __init__(self, dim: int = 128):
        super().__init__()
        if dim % 2:
            raise ValueError("Fourier feature dimension must be even")
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), dim // 2))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.float().reshape(-1, 1) * self.freqs[None].to(t.device)
        return torch.cat((x.sin(), x.cos()), dim=-1)


class MappingNetwork(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int, d_ff: int):
        super().__init__()
        layers = [nn.Linear(d_in, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, d_ff), nn.SiLU(), nn.Linear(d_ff, width), nn.SiLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class FeedForward(nn.Module):
    def __init__(self, dim: int, d_ff: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * d_ff)
        self.out_proj = nn.Linear(d_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.dropout(self.out_proj(value * F.gelu(gate)))


class Attention(nn.Module):
    def __init__(self, dim: int, d_head: int, dropout: float):
        super().__init__()
        if dim % d_head:
            raise ValueError(f"{dim=} must divide {d_head=}")
        self.num_heads = dim // d_head
        self.d_head = d_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, d_ff: int, cond_dim: int, d_head: int, dropout: float):
        super().__init__()
        self.norm1 = AdaRMSNorm(dim, cond_dim)
        self.attn = Attention(dim, d_head, dropout)
        self.norm2 = AdaRMSNorm(dim, cond_dim)
        self.mlp = FeedForward(dim, d_ff, dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, cond))
        return x + self.mlp(self.norm2(x, cond))


def _partition_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    b, h, w, c = x.shape
    if h % window or w % window:
        raise ValueError(f"Window {window} does not divide spatial shape {(h, w)}")
    return x.reshape(b, h // window, window, w // window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def _reverse_windows(x: torch.Tensor, batch: int, h: int, w: int, window: int) -> torch.Tensor:
    c = x.shape[-1]
    return x.reshape(batch, h // window, w // window, window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(batch, h, w, c)


class ShiftedWindowBlock(nn.Module):
    """Official HDiT-style alternating shifted-window transformer block."""

    def __init__(self, dim: int, d_ff: int, cond_dim: int, d_head: int, window_size: int,
                 shift: bool, dropout: float):
        super().__init__()
        self.window_size = int(window_size)
        self.shift = bool(shift)
        self.norm1 = AdaRMSNorm(dim, cond_dim)
        self.attn = Attention(dim, d_head, dropout)
        self.norm2 = AdaRMSNorm(dim, cond_dim)
        self.mlp = FeedForward(dim, d_ff, dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        y = self.norm1(x.reshape(b, h * w, -1), cond).reshape_as(x)
        shift = self.window_size // 2 if self.shift else 0
        if shift:
            y = torch.roll(y, shifts=(-shift, -shift), dims=(1, 2))
        y = _reverse_windows(self.attn(_partition_windows(y, self.window_size)), b, h, w, self.window_size)
        if shift:
            y = torch.roll(y, shifts=(shift, shift), dims=(1, 2))
        x = x + y
        z = self.norm2(x.reshape(b, h * w, -1), cond).reshape_as(x)
        return x + self.mlp(z.reshape(b, h * w, -1)).reshape_as(x)


class TokenMerge(nn.Module):
    """2x2 token merge used by the public HDiT hierarchy."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(4 * dim_in, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        if h % 2 or w % 2:
            raise ValueError("TokenMerge requires even spatial dimensions")
        y = x.reshape(b, h // 2, 2, w // 2, 2, c).permute(0, 1, 3, 2, 4, 5)
        return self.proj(y.reshape(b, h // 2, w // 2, 4 * c))


class TokenSplit(nn.Module):
    """2x2 token split with HDiT's learned lerp skip merge."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, 4 * dim_out, bias=False)
        self.lerp = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        c = skip.shape[-1]
        y = self.proj(x).reshape(b, h, w, 2, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h * 2, w * 2, c)
        return torch.lerp(skip, y, self.lerp.clamp(0.0, 1.0))


class HDiTConditionEncoder(nn.Module):
    """Bias-free HDiT-aligned spatial and global condition pyramid.

    It lives beside the core so CPU topology tests do not depend on the
    PyKeOps-backed PointPillar runtime.  Every learned projection is
    bias-free; therefore an all-zero inactive condition is exactly zero.
    """

    def __init__(self, lidar_channels=32, layout_channels=2, widths=(128, 256, 512), patch_size=4,
                 mapping_width=256, fusion="shared", native=False):
        super().__init__()
        self.lidar_channels, self.layout_channels = int(lidar_channels), int(layout_channels)
        self.in_channels = self.lidar_channels + self.layout_channels
        self.widths = tuple(map(int, widths))
        self.fusion, self.native = str(fusion), bool(native)
        if self.fusion == "shared":
            self.patch = nn.Conv2d(self.in_channels, self.widths[0], patch_size, stride=patch_size, bias=False)
        elif self.fusion == "separate":
            self.lidar_patch = nn.Conv2d(self.lidar_channels, self.widths[0], patch_size, stride=patch_size, bias=False)
            self.vehicle_patch = nn.Conv2d(1, self.widths[0], patch_size, stride=patch_size, bias=False)
            self.road_patch = nn.Conv2d(1, self.widths[0], patch_size, stride=patch_size, bias=False)
        else:
            raise ValueError("HDiT condition fusion must be shared or separate")
        self.merge0 = TokenMerge(self.widths[0], self.widths[1])
        self.merge1 = TokenMerge(self.widths[1], self.widths[2])
        self.global_proj = nn.Linear(self.widths[0], mapping_width, bias=False)
        if self.native:
            self.level_global_proj = nn.ModuleList(
                nn.Linear(width, mapping_width, bias=False) for width in self.widths
            )
        self._modality_norms = {}

    def forward(self, raw_pc: torch.Tensor, layout: torch.Tensor):
        x = _require_bchw(raw_pc, layout, self.in_channels)
        if self.fusion == "shared":
            c0 = self.patch(x)
            self._modality_norms = {}
        else:
            components = {
                "lidar": self.lidar_patch(raw_pc),
                "vehicle": self.vehicle_patch(layout[:, 0:1]),
                "road": self.road_patch(layout[:, 1:2]),
            }
            c0 = sum(components.values())
            self._modality_norms = modality_norms_and_ratios(components, channel_dim=1)
        c0 = c0.permute(0, 2, 3, 1)
        c1 = self.merge0(c0)
        c2 = self.merge1(c1)
        global_condition = self.global_proj(c0.mean(dim=(1, 2)))
        return c0, c1, c2, global_condition

    def modality_norms(self):
        """Detached C3 component norms for low-frequency diagnostic logging."""
        return self._modality_norms

    def native_global_conditions(self, maps):
        if not self.native:
            raise RuntimeError("Native HDiT condition projections were not enabled")
        return tuple(
            projection(value.mean(dim=(1, 2)))
            for projection, value in zip(self.level_global_proj, maps)
        )


class HDiTCore(nn.Module):
    """HDiT-S 64->32->16 hierarchy with stage-boundary injection hooks."""

    def __init__(self, input_size: int = 256, in_channels: int = 3, patch_size: int = 4,
                 widths: Sequence[int] = (128, 256, 512), depths: Sequence[int] = (2, 2, 4),
                 d_ffs: Sequence[int] = (384, 768, 1536), d_head: int = 64,
                 window_size: int = 8, mapping_width: int = 256, mapping_depth: int = 2,
                 mapping_d_ff: int = 768, dropout: Sequence[float] = (0.0, 0.0, 0.1)):
        super().__init__()
        self.input_size, self.in_channels, self.patch_size = int(input_size), int(in_channels), int(patch_size)
        self.widths, self.depths, self.d_ffs = tuple(map(int, widths)), tuple(map(int, depths)), tuple(map(int, d_ffs))
        self.window_size, self.d_head = int(window_size), int(d_head)
        if len(self.widths) != 3 or len(self.depths) != 3 or len(self.d_ffs) != 3:
            raise ValueError("HDiT-S requires three hierarchy levels")
        if self.input_size % (self.patch_size * 4):
            raise ValueError("input_size must remain divisible through two token merges")
        self.patch_embed = nn.Conv2d(self.in_channels, self.widths[0], self.patch_size, stride=self.patch_size)
        # Time and adapter-derived global condition share the mapping-width
        # space before the official-style mapping MLP.
        self.time_features = FourierFeatures(mapping_width)
        self.mapping = MappingNetwork(mapping_width, mapping_width, mapping_depth, mapping_d_ff)
        self.to_stage_cond = nn.ModuleList(nn.Linear(mapping_width, width) for width in self.widths)
        self.enc0 = nn.ModuleList(ShiftedWindowBlock(self.widths[0], self.d_ffs[0], self.widths[0], d_head, window_size, i % 2 == 1, dropout[0]) for i in range(self.depths[0]))
        self.merge0 = TokenMerge(self.widths[0], self.widths[1])
        self.enc1 = nn.ModuleList(ShiftedWindowBlock(self.widths[1], self.d_ffs[1], self.widths[1], d_head, window_size, i % 2 == 1, dropout[1]) for i in range(self.depths[1]))
        self.merge1 = TokenMerge(self.widths[1], self.widths[2])
        self.middle = nn.ModuleList(TransformerBlock(self.widths[2], self.d_ffs[2], self.widths[2], d_head, dropout[2]) for _ in range(self.depths[2]))
        self.split1 = TokenSplit(self.widths[2], self.widths[1])
        self.dec1 = nn.ModuleList(ShiftedWindowBlock(self.widths[1], self.d_ffs[1], self.widths[1], d_head, window_size, i % 2 == 1, dropout[1]) for i in range(self.depths[1]))
        self.split0 = TokenSplit(self.widths[1], self.widths[0])
        self.dec0 = nn.ModuleList(ShiftedWindowBlock(self.widths[0], self.d_ffs[0], self.widths[0], d_head, window_size, i % 2 == 1, dropout[0]) for i in range(self.depths[0]))
        self.unpatchify = nn.ConvTranspose2d(self.widths[0], self.in_channels, self.patch_size, stride=self.patch_size)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, global_condition: Optional[torch.Tensor] = None,
                stage_add: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
                stage_global_conditions: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        if xt.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size}x{self.input_size}, got {tuple(xt.shape[-2:])}")
        mapping_input = self.time_features(t)
        if global_condition is not None:
            mapping_input = mapping_input + global_condition
        mapped = self.mapping(mapping_input)
        if stage_global_conditions is None:
            conds = [layer(mapped) for layer in self.to_stage_cond]
        else:
            if len(stage_global_conditions) != 3:
                raise ValueError("HDiT native condition requires three level-global vectors")
            conds = [
                layer(mapped + stage_global_conditions[level])
                for level, layer in enumerate(self.to_stage_cond)
            ]
        inject = stage_add if stage_add is not None else (lambda _stage, x: x)
        x = self.patch_embed(xt).permute(0, 2, 3, 1)
        x = inject(0, x)
        for block in self.enc0:
            x = block(x, conds[0])
        skip0 = x
        x = inject(1, self.merge0(x))
        for block in self.enc1:
            x = block(x, conds[1])
        skip1 = x
        x = inject(2, self.merge1(x))
        for block in self.middle:
            b, h, w, c = x.shape
            x = block(x.reshape(b, h * w, c), conds[2]).reshape(b, h, w, c)
        x = inject(3, self.split1(x, skip1))
        for block in self.dec1:
            x = block(x, conds[1])
        x = inject(4, self.split0(x, skip0))
        for block in self.dec0:
            x = block(x, conds[0])
        return self.unpatchify(x.permute(0, 3, 1, 2))

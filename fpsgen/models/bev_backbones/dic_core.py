"""DiC-S convolutional core, architecturally reimplemented for FPSGen.

Architecture follows ``YuchuanTian/DiC`` at
``bfa541b5e3a968919f3a399fc0e223e877439b3b``, source ``dic_models.py``.
The public DiC-S block/depth/down-up/skip topology is retained.  FPSGen uses
three BEV channels, predicts velocity (not variance), and contributes its
spatial conditions only at stage boundaries in ``dic_bev.py``.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    """DiC's dynamic group-count GroupNorm implementation."""

    def __init__(self, channels: int, groups: int = 32, min_channels: int = 4,
                 eps: float = 1e-5):
        super().__init__()
        self.groups = min(groups, max(1, channels // min_channels))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(x, self.groups, self.weight.to(x.dtype),
                            self.bias.to(x.dtype), self.eps)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class TimestepEmbedder(nn.Module):
    """Official DiC sinusoidal timestep MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(
            half, dtype=torch.float32, device=t.device
        ) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=-1) if dim % 2 else emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class FinalLayer(nn.Module):
    """Official DiC final GroupNorm + adaptive modulation + 3x3 projection."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = GroupNorm(hidden_size, eps=1e-6)
        self.out_proj = nn.Conv2d(hidden_size, out_channels, 3, padding=1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.out_proj(modulate(self.norm_final(x), shift, scale))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Official 3x3 conv followed by PixelUnshuffle(2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels % 4:
            raise ValueError("PixelUnshuffle output channels must be divisible by four")
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Official 3x3 conv followed by PixelShuffle(2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UNetBlock(nn.Module):
    """Official DiC GroupNorm/GELU/adaptive-scale-shift-gate residual block."""

    def __init__(self, in_channels: int, out_channels: int, emb_channels: int,
                 affinef: int = 3):
        super().__init__()
        self.norm0 = GroupNorm(in_channels)
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.affine = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels * affinef))
        self.norm1 = GroupNorm(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        original = x
        x = self.conv0(F.gelu(self.norm0(x)))
        gate, scale, shift = self.affine(emb).chunk(3, dim=1)
        x = F.gelu(modulate(self.norm1(x), shift, scale))
        x = self.conv1(x)
        return gate[:, :, None, None] * x + (self.skip(original) if self.skip else original)


class UBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, emb_channels: int):
        super().__init__()
        self.conv = UNetBlock(in_channels, hidden_size, emb_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.conv(x, c)


class DiCCore(nn.Module):
    """DiC-S core with an explicit stage-boundary injection callback.

    Labels are intentionally absent: the official label contribution is fixed
    to zero, while its timestep embedding and all convolutional topology remain
    unchanged.
    """

    def __init__(self, input_size: int = 256, in_channels: int = 3,
                 hidden_size: int = 96, depth=(6, 6, 5, 6, 6),
                 mult_channels=(1, 2, 4, 2, 1), skip_stride: int = 3):
        super().__init__()
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.depth = tuple(int(v) for v in depth)
        self.mult_channels = tuple(int(v) for v in mult_channels)
        self.skip_stride = int(skip_stride)
        if self.depth != (6, 6, 5, 6, 6) or self.mult_channels != (1, 2, 4, 2, 1):
            raise ValueError("DiC-S requires official depth=[6,6,5,6,6] and mult_channels=[1,2,4,2,1]")
        if self.skip_stride != 3:
            raise ValueError("DiC-S requires official skip_stride=3")
        self.x_embedder = OverlapPatchEmbed(in_channels, hidden_size)
        stage_channels = [hidden_size * mult for mult in self.mult_channels]
        self.t_embedder_ls = nn.ModuleList(TimestepEmbedder(c) for c in stage_channels[:3])
        self.enc_blocks = nn.ModuleList([
            nn.ModuleList(UBlock(stage_channels[level], stage_channels[level], stage_channels[level])
                          for _ in range(self.depth[level]))
            for level in range(2)
        ])
        self.downs = nn.ModuleList([
            Downsample(stage_channels[0], stage_channels[1]),
            Downsample(stage_channels[1], stage_channels[2]),
        ])
        self.lat_blocks = nn.ModuleList([
            UBlock(stage_channels[2], stage_channels[2], stage_channels[2])
            for _ in range(self.depth[2])
        ])
        self.ups = nn.ModuleList([
            Upsample(stage_channels[2], stage_channels[3]),
            Upsample(stage_channels[3], stage_channels[4]),
        ])
        self.dec_blocks = nn.ModuleList([
            nn.ModuleList(
                UBlock(stage_channels[3] + stage_channels[1] if i % skip_stride == 0 else stage_channels[3],
                       stage_channels[3], stage_channels[3])
                for i in range(self.depth[3])
            ),
            nn.ModuleList(
                UBlock(stage_channels[4] + stage_channels[0] if i % skip_stride == 0 else stage_channels[4],
                       stage_channels[4], stage_channels[4])
                for i in range(self.depth[4])
            ),
        ])
        self.output = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.final_layer = FinalLayer(hidden_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(basic_init)
        for embedder in self.t_embedder_ls:
            nn.init.normal_(embedder.mlp[0].weight, std=.02)
            nn.init.normal_(embedder.mlp[2].weight, std=.02)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.out_proj.weight)
        nn.init.zeros_(self.final_layer.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, stage_add=None) -> torch.Tensor:
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size}x{self.input_size} input, got {tuple(x.shape[-2:])}")
        c0, c1, c2 = (embed(t) for embed in self.t_embedder_ls)
        conditions = (c0, c1, c2, c1, c0)
        inject = stage_add if stage_add is not None else (lambda stage, tensor: tensor)
        x = inject(0, self.x_embedder(x))
        skips = []
        for block_index, block in enumerate(self.enc_blocks[0]):
            x = block(x, conditions[0])
            if (len(self.enc_blocks[0]) - 1 - block_index) % self.skip_stride == 0:
                skips.append(x)
        x = inject(1, self.downs[0](x))
        for block_index, block in enumerate(self.enc_blocks[1]):
            x = block(x, conditions[1])
            if (len(self.enc_blocks[1]) - 1 - block_index) % self.skip_stride == 0:
                skips.append(x)
        x = inject(2, self.downs[1](x))
        for block in self.lat_blocks:
            x = block(x, conditions[2])
        x = inject(3, self.ups[0](x))
        for block_index, block in enumerate(self.dec_blocks[0]):
            x = block(torch.cat((x, skips.pop()), dim=1), conditions[3]) if block_index % self.skip_stride == 0 else block(x, conditions[3])
        x = inject(4, self.ups[1](x))
        for block_index, block in enumerate(self.dec_blocks[1]):
            x = block(torch.cat((x, skips.pop()), dim=1), conditions[4]) if block_index % self.skip_stride == 0 else block(x, conditions[4])
        return self.final_layer(self.output(x), conditions[4])

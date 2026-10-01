"""Small source-based SynFlow U-Net used for controlled BEV conditioning.

This is intentionally a compact U-Net adaptation, not a claim of an official
SynFlow/FMS2 FPSGen implementation.  U0, SynFlow and CrackSegFlow-style share
this exact core; only the spatial conditioning placement/type changes.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fpsgen.models.bev_conditioning.lite_spade import LiteSPADE


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequency = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        embedding = torch.cat((torch.sin(t[:, None] * frequency), torch.cos(t[:, None] * frequency)), dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, head_channels: int):
        super().__init__()
        self.heads = max(1, channels // head_channels)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(tokens).chunk(3, dim=1)
        width = c // self.heads
        # Torch 2.0 CUDA SDPA requires contiguous head-width dimensions.
        q = q.view(b, self.heads, width, h * w).transpose(-2, -1).contiguous()
        k = k.view(b, self.heads, width, h * w).transpose(-2, -1).contiguous()
        v = v.view(b, self.heads, width, h * w).transpose(-2, -1).contiguous()
        attended = F.scaled_dot_product_attention(q, k, v).transpose(-2, -1).reshape(b, c, h * w)
        return x + self.out(attended).reshape(b, c, h, w)


class SynResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, condition_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.spade = LiteSPADE(channels, condition_channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels * 2))
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.last_ratio = torch.tensor(0.)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, condition: torch.Tensor | None,
                injection: str, gate: torch.Tensor | None) -> torch.Tensor:
        if condition is not None and injection == "spade":
            h = self.spade(x, condition)
            effect = self.spade.beta(condition) + self.spade.gamma(condition) * self.norm1(x)
        else:
            h = self.norm1(x)
            effect = None
            if condition is not None and injection == "generic_add":
                effect = gate.to(x.dtype) * condition
                h = h + effect
        if effect is not None:
            self.last_ratio = effect.detach().float().square().mean().sqrt() / (x.detach().float().square().mean().sqrt() + 1e-8)
        else:
            self.last_ratio = x.new_zeros((), dtype=torch.float32)
        h = self.conv1(F.silu(h))
        scale, shift = self.time(temb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.conv2(F.silu(h))


class SynConditionEncoder(nn.Module):
    """Shared nonlinear 34ch pyramid. No modality-specific learned branch."""
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        self.channels = tuple(int(x) for x in channels)
        self.stem = nn.Sequential(nn.Conv2d(34, self.channels[0], 3, padding=1, bias=False),
                                  nn.GroupNorm(_groups(self.channels[0]), self.channels[0], affine=False), nn.SiLU())
        self.down = nn.ModuleList()
        for source, target in zip(self.channels[:-1], self.channels[1:]):
            self.down.append(nn.Sequential(nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
                                           nn.GroupNorm(_groups(target), target, affine=False), nn.SiLU()))

    def forward(self, condition_map: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = [self.stem(condition_map)]
        for layer in self.down:
            values.append(layer(values[-1]))
        return tuple(values)


class SynFlowCore(nn.Module):
    """Shared 256→8 U-Net core with spatial conditioning selectable by config."""
    def __init__(self, input_size=256, in_channels=3, model_channels=64,
                 channel_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2,
                 attention_spatial_resolutions=(32, 16, 8), num_head_channels=64,
                 time_scale=1000.0, injection="spade", encoder_conditioned=True,
                 middle_conditioned=True, decoder_conditioned=True, boundary_gate=False):
        super().__init__()
        self.input_size, self.time_scale = int(input_size), float(time_scale)
        self.channels = tuple(int(model_channels * multiplier) for multiplier in channel_mult)
        self.num_res_blocks, self.injection = int(num_res_blocks), str(injection)
        self.encoder_conditioned, self.middle_conditioned = bool(encoder_conditioned), bool(middle_conditioned)
        self.decoder_conditioned, self.boundary_gate = bool(decoder_conditioned), bool(boundary_gate)
        if self.input_size % (2 ** (len(self.channels) - 1)):
            raise ValueError("input size must be divisible by the U-Net downsample factor")
        self.time = TimeEmbedding(model_channels)
        time_dim = model_channels * 4
        self.input = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)
        self.enc = nn.ModuleList(nn.ModuleList(SynResBlock(width, time_dim, width) for _ in range(self.num_res_blocks)) for width in self.channels)
        self.enc_attn = nn.ModuleList(nn.ModuleList(
            SpatialAttention(width, num_head_channels) if self.input_size // (2 ** level) in attention_spatial_resolutions else nn.Identity()
            for _ in range(self.num_res_blocks)) for level, width in enumerate(self.channels))
        self.down = nn.ModuleList(nn.Conv2d(source, target, 3, stride=2, padding=1) for source, target in zip(self.channels[:-1], self.channels[1:]))
        self.mid = nn.ModuleList(SynResBlock(self.channels[-1], time_dim, self.channels[-1]) for _ in range(self.num_res_blocks))
        self.mid_attn = nn.ModuleList(SpatialAttention(self.channels[-1], num_head_channels) for _ in range(self.num_res_blocks))
        self.up = nn.ModuleList(nn.Conv2d(self.channels[level + 1], self.channels[level], 3, padding=1) for level in reversed(range(len(self.channels) - 1)))
        self.merge = nn.ModuleList(nn.Conv2d(self.channels[level] * 2, self.channels[level], 1) for level in reversed(range(len(self.channels) - 1)))
        self.dec = nn.ModuleList(nn.ModuleList(SynResBlock(self.channels[level], time_dim, self.channels[level]) for _ in range(self.num_res_blocks)) for level in reversed(range(len(self.channels) - 1)))
        self.dec_attn = nn.ModuleList(nn.ModuleList(
            SpatialAttention(self.channels[level], num_head_channels) if self.input_size // (2 ** level) in attention_spatial_resolutions else nn.Identity()
            for _ in range(self.num_res_blocks)) for level in reversed(range(len(self.channels) - 1)))
        self.gates = nn.Parameter(torch.full((len(self.channels),), 0.1))
        self.boundary_alpha = nn.Parameter(torch.zeros(1))
        self.out_norm = nn.GroupNorm(_groups(self.channels[0]), self.channels[0])
        self.out = nn.Conv2d(self.channels[0], in_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self._diagnostics = {}

    def _run_stage(self, blocks, attention, x, temb, condition, enabled, level):
        active = condition if enabled else None
        for block, attn in zip(blocks, attention):
            x = block(x, temb, active, self.injection, self.gates[level])
            x = attn(x)
        return x

    def forward(self, xt: torch.Tensor, t: torch.Tensor, condition_pyramid: tuple[torch.Tensor, ...],
                boundary: torch.Tensor | None = None) -> torch.Tensor:
        if len(condition_pyramid) != len(self.channels):
            raise ValueError("condition pyramid level mismatch")
        temb = self.time(t * self.time_scale)
        x, skips = self.input(xt), []
        for level, (blocks, attn) in enumerate(zip(self.enc, self.enc_attn)):
            x = self._run_stage(blocks, attn, x, temb, condition_pyramid[level], self.encoder_conditioned, level)
            skips.append(x)
            if level < len(self.down): x = self.down[level](x)
        x = self._run_stage(self.mid, self.mid_attn, x, temb, condition_pyramid[-1], self.middle_conditioned, len(self.channels) - 1)
        for index, level in enumerate(reversed(range(len(self.channels) - 1))):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.up[index](x)
            x = self.merge[index](torch.cat((x, skips[level]), dim=1))
            if self.boundary_gate and boundary is not None:
                edge = F.interpolate(boundary, size=x.shape[-2:], mode="nearest")
                x = x * (1 + self.boundary_alpha.to(x.dtype) * edge)
            x = self._run_stage(self.dec[index], self.dec_attn[index], x, temb, condition_pyramid[level], self.decoder_conditioned, level)
        values = {}
        for family, stages in (("enc", self.enc), ("mid", [self.mid]), ("dec", self.dec)):
            for sidx, blocks in enumerate(stages):
                for bidx, block in enumerate(blocks):
                    values[f"ratio_{family}_{sidx}_{bidx}"] = block.last_ratio.detach()
                    if self.injection == "spade":
                        values[f"gamma_{family}_{sidx}_{bidx}"] = block.spade.last_gamma_rms.detach()
                        values[f"beta_{family}_{sidx}_{bidx}"] = block.spade.last_beta_rms.detach()
        self._diagnostics = values
        return self.out(F.silu(self.out_norm(x)))

    def condition_diagnostics(self):
        return {**{f"gate_{idx}": value.detach() for idx, value in enumerate(self.gates)},
                "boundary_alpha": self.boundary_alpha.detach(), **self._diagnostics}

"""NCSN++-Small pixel-space velocity backbone for FPSGen.

Based on the Apache-2.0 NCSN++ architecture in ``yang-song/score_sde_pytorch``
commit ``cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44``, source
``models/ncsnpp.py``.  The 256 hierarchy, BigGAN residual blocks, FIR
resampling, input/output skip paths and DDPM attention are retained.  FPSGen
uses continuous FM time and predicts velocity rather than an SDE score.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition import _require_bchw
from .pixelu_core import TimestepEmbedder


def _groups(channels: int) -> int:
    # Keep at least two channels per group so CPU tiny-shape guards reaching
    # 1x1 remain well-defined at batch size one.
    for groups in range(min(32, max(1, channels // 2)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FIRResample(nn.Module):
    """Lightweight explicit [1,3,3,1] FIR filter around 2x resampling."""

    def __init__(self, fir_kernel=(1, 3, 3, 1), up: bool = False):
        super().__init__()
        kernel = torch.tensor(fir_kernel, dtype=torch.float32)
        kernel2d = torch.outer(kernel, kernel)
        self.register_buffer("kernel", kernel2d / kernel2d.sum(), persistent=False)
        self.up = bool(up)

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        k = self.kernel.to(x.dtype).expand(c, 1, -1, -1)
        # Asymmetric same padding preserves dimensions for the even 4x4 FIR.
        return F.conv2d(F.pad(x, (1, 2, 1, 2)), k, groups=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up:
            return self._filter(F.interpolate(x, scale_factor=2, mode="nearest"))
        return F.avg_pool2d(self._filter(x), 2)


class BigGANResBlock(nn.Module):
    """NCSN++ BigGAN residual block with time embedding and skip rescaling."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float = 0.0,
                 skip_rescale: bool = True):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.skip_rescale = bool(skip_rescale)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        out = self.skip(x) + h
        return out * (1.0 / math.sqrt(2.0) if self.skip_rescale else 1.0)


class AttnBlockpp(nn.Module):
    """DDPM attention block retained at the official 16x16 resolution."""

    def __init__(self, channels: int, skip_rescale: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.skip_rescale = bool(skip_rescale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        q = self.q(y).reshape(b, c, h * w).transpose(1, 2)
        k = self.k(y).reshape(b, c, h * w)
        weight = torch.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        v = self.v(y).reshape(b, c, h * w).transpose(1, 2)
        y = torch.bmm(weight, v).transpose(1, 2).reshape(b, c, h, w)
        out = x + self.proj(y)
        return out * (1.0 / math.sqrt(2.0) if self.skip_rescale else 1.0)


class NCSNConditionEncoder(nn.Module):
    """Bias-free seven-resolution spatial/global FPSGen condition pyramid."""

    def __init__(self, lidar_channels=32, layout_channels=2, nf=64,
                 ch_mult=(1, 1, 2, 2, 2, 2, 2), temb_dim=None, fusion="shared",
                 native=False, condition_nf=None):
        super().__init__()
        self.lidar_channels, self.layout_channels = int(lidar_channels), int(layout_channels)
        self.in_channels = self.lidar_channels + self.layout_channels
        self.channels = tuple(int(nf * m) for m in ch_mult)
        self.hidden_channels = self.channels if condition_nf is None else tuple(int(condition_nf * m) for m in ch_mult)
        self.fusion, self.native = str(fusion), bool(native)
        if self.fusion == "shared":
            self.level0 = nn.Conv2d(self.in_channels, self.hidden_channels[0], 3, padding=1, bias=False)
        elif self.fusion == "separate":
            self.lidar_level0 = nn.Conv2d(self.lidar_channels, self.hidden_channels[0], 3, padding=1, bias=False)
            self.vehicle_level0 = nn.Conv2d(1, self.hidden_channels[0], 3, padding=1, bias=False)
            self.road_level0 = nn.Conv2d(1, self.hidden_channels[0], 3, padding=1, bias=False)
        else:
            raise ValueError("NCSN++ condition fusion must be shared or separate")
        self.transitions = nn.ModuleList(nn.Conv2d(self.hidden_channels[i], self.hidden_channels[i + 1], 3, padding=1, bias=False) for i in range(len(self.channels) - 1))
        self.global_proj = nn.Linear(self.hidden_channels[0], int(temb_dim or nf * 4), bias=False)
        if self.hidden_channels != self.channels:
            self.spatial_out = nn.ModuleList(
                nn.Conv2d(hidden, output, 1, bias=False)
                for hidden, output in zip(self.hidden_channels, self.channels)
            )
        if self.native:
            self.level_global_proj = nn.ModuleList(
                nn.Linear(channels, int(temb_dim or nf * 4), bias=False)
                for channels in self.channels
            )

    def forward(self, raw_pc: torch.Tensor, layout: torch.Tensor):
        x = _require_bchw(raw_pc, layout, self.in_channels)
        if self.fusion == "shared":
            hidden_maps = [self.level0(x)]
        else:
            hidden_maps = [
                self.lidar_level0(raw_pc)
                + self.vehicle_level0(layout[:, 0:1])
                + self.road_level0(layout[:, 1:2])
            ]
        for transition in self.transitions:
            hidden_maps.append(transition(F.avg_pool2d(hidden_maps[-1], 2)))
        maps = hidden_maps if not hasattr(self, "spatial_out") else [
            projection(value) for projection, value in zip(self.spatial_out, hidden_maps)
        ]
        global_condition = self.global_proj(hidden_maps[0].mean(dim=(2, 3)))
        return tuple(maps), global_condition

    def native_global_conditions(self, maps):
        if not self.native:
            raise RuntimeError("Native NCSN++ condition projections were not enabled")
        return tuple(
            projection(value.mean(dim=(2, 3)))
            for projection, value in zip(self.level_global_proj, maps)
        )


class NCSNppCore(nn.Module):
    """Seven-resolution NCSN++ 256 hierarchy adapted to direct FM velocity."""

    def __init__(self, input_size=256, in_channels=3, nf=96,
                 ch_mult=(1, 1, 2, 2, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(16,), resblock_type="biggan", resamp_with_conv=True,
                 fir=True, fir_kernel=(1, 3, 3, 1), skip_rescale=True,
                 progressive="output_skip", progressive_input="input_skip", progressive_combine="sum",
                 attention_type="ddpm", init_scale=0.0, conv_size=3):
        super().__init__()
        self.input_size, self.in_channels, self.nf = int(input_size), int(in_channels), int(nf)
        self.ch_mult, self.num_res_blocks = tuple(map(int, ch_mult)), int(num_res_blocks)
        self.attn_resolutions = tuple(map(int, attn_resolutions))
        self.resblock_type, self.resamp_with_conv, self.fir = resblock_type, bool(resamp_with_conv), bool(fir)
        self.fir_kernel, self.skip_rescale = tuple(fir_kernel), bool(skip_rescale)
        self.progressive, self.progressive_input, self.progressive_combine = progressive, progressive_input, progressive_combine
        self.attention_type, self.init_scale, self.conv_size = attention_type, float(init_scale), int(conv_size)
        if self.ch_mult != (1, 1, 2, 2, 2, 2, 2) or self.num_res_blocks != 2:
            raise ValueError("NCSNpp-S-BEV retains ch_mult=[1,1,2,2,2,2,2] and two residual blocks")
        if self.resblock_type != "biggan" or not self.fir or self.progressive != "output_skip" or self.progressive_input != "input_skip":
            raise ValueError("NCSNpp-S-BEV retains BigGAN/FIR/input_skip/output_skip topology")
        self.channels = tuple(self.nf * mult for mult in self.ch_mult)
        self.temb_dim = self.nf * 4
        self.t_embedder = TimestepEmbedder(self.temb_dim)
        self.input_conv = nn.Conv2d(self.in_channels, self.channels[0], self.conv_size, padding=self.conv_size // 2)
        self.global_condition_proj = nn.Linear(self.temb_dim, self.temb_dim, bias=False)
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        self.input_pyramid_proj = nn.ModuleList()
        prev = self.channels[0]
        for level, channels in enumerate(self.channels):
            blocks = nn.ModuleList()
            attn = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(BigGANResBlock(prev, channels, self.temb_dim, skip_rescale=self.skip_rescale))
                prev = channels
                attn.append(AttnBlockpp(channels, self.skip_rescale) if self._resolution(level) in self.attn_resolutions else nn.Identity())
            self.enc_blocks.append(blocks)
            self.enc_attn.append(attn)
            if level < len(self.channels) - 1:
                self.downsamplers.append(FIRResample(self.fir_kernel, up=False))
                self.down_projs.append(nn.Conv2d(self.channels[level], self.channels[level + 1], 3, padding=1))
                self.input_pyramid_proj.append(nn.Conv2d(self.in_channels, self.channels[level + 1], 1))
                # The FIR/down projection changes the next level's feature
                # width before its first BigGAN block.
                prev = self.channels[level + 1]
        self.mid1 = BigGANResBlock(self.channels[-1], self.channels[-1], self.temb_dim, skip_rescale=self.skip_rescale)
        self.mid_attn = AttnBlockpp(self.channels[-1], self.skip_rescale) if self._resolution(len(self.channels) - 1) in self.attn_resolutions else nn.Identity()
        self.mid2 = BigGANResBlock(self.channels[-1], self.channels[-1], self.temb_dim, skip_rescale=self.skip_rescale)
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.output_heads = nn.ModuleList()
        prev = self.channels[-1]
        for level in reversed(range(len(self.channels))):
            channels = self.channels[level]
            blocks = nn.ModuleList([BigGANResBlock(prev + channels, channels, self.temb_dim, skip_rescale=self.skip_rescale),
                                    BigGANResBlock(channels, channels, self.temb_dim, skip_rescale=self.skip_rescale)])
            self.dec_blocks.append(blocks)
            self.dec_attn.append(nn.ModuleList([AttnBlockpp(channels, self.skip_rescale) if self._resolution(level) in self.attn_resolutions else nn.Identity(),
                                                AttnBlockpp(channels, self.skip_rescale) if self._resolution(level) in self.attn_resolutions else nn.Identity()]))
            self.output_heads.append(nn.Sequential(nn.GroupNorm(_groups(channels), channels, eps=1e-6), nn.SiLU(), nn.Conv2d(channels, self.in_channels, 3, padding=1)))
            prev = channels
            if level > 0:
                self.upsamplers.append(FIRResample(self.fir_kernel, up=True))
        self._init_weights()

    def _resolution(self, level: int) -> int:
        return self.input_size // (2 ** level)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Official NCSN++ uses zero-ish output scale.  The first output head is
        # intentionally small, while subsequent real optimizer steps open all paths.
        for head in self.output_heads:
            nn.init.constant_(head[-1].weight, self.init_scale)
            nn.init.zeros_(head[-1].bias)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, condition_maps=None, global_condition=None,
                level_global_conditions=None) -> torch.Tensor:
        if xt.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size}x{self.input_size}, got {tuple(xt.shape[-2:])}")
        if condition_maps is not None and len(condition_maps) != len(self.channels):
            raise ValueError("NCSN++ condition must contain seven resolution maps")
        if level_global_conditions is not None and len(level_global_conditions) != len(self.channels):
            raise ValueError("NCSN++ native condition must contain seven level-global vectors")
        temb = self.t_embedder(t)
        if global_condition is not None:
            temb = temb + self.global_condition_proj(global_condition)
        h = self.input_conv(xt)
        input_pyramid = xt
        skips = []
        for level, (blocks, attns) in enumerate(zip(self.enc_blocks, self.enc_attn)):
            if condition_maps is not None:
                condition = condition_maps[level]
                if h.shape != condition.shape:
                    raise RuntimeError(f"NCSN++ condition level {level} mismatch: {tuple(h.shape)} vs {tuple(condition.shape)}")
                h = h + condition
            level_temb = temb if level_global_conditions is None else temb + level_global_conditions[level]
            for block, attn in zip(blocks, attns):
                h = attn(block(h, level_temb))
            skips.append(h)
            if level < len(self.channels) - 1:
                h = self.down_projs[level](self.downsamplers[level](h))
                input_pyramid = self.downsamplers[level](input_pyramid)
                h = (h + self.input_pyramid_proj[level](input_pyramid)) * (1.0 / math.sqrt(2.0))
        middle_temb = temb if level_global_conditions is None else temb + level_global_conditions[-1]
        h = self.mid2(self.mid_attn(self.mid1(h, middle_temb)), middle_temb)
        output_pyramid = None
        up_index = 0
        for index, level in enumerate(reversed(range(len(self.channels)))):
            skip = skips[level]
            level_temb = temb if level_global_conditions is None else temb + level_global_conditions[level]
            h = self.dec_attn[index][0](self.dec_blocks[index][0](torch.cat((h, skip), dim=1), level_temb))
            h = self.dec_attn[index][1](self.dec_blocks[index][1](h, level_temb))
            current = self.output_heads[index](h)
            output_pyramid = current if output_pyramid is None else F.interpolate(output_pyramid, scale_factor=2, mode="nearest") + current
            if level > 0:
                h = self.upsamplers[up_index](h)
                up_index += 1
        return output_pyramid

"""FPSGen adapter for PixelDiT-S-BEV with three-level condition injection."""
from __future__ import annotations

import torch
import torch.nn as nn

from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
from .condition import _require_bchw, modality_norms_and_ratios
from .pixeldit_core import PixelDiTCore


class PixelDiTConditionEncoder(nn.Module):
    """Bias-free separate-first cell, patch, and global condition projections."""
    def __init__(self, lidar_channels=32, patch_size=16, pixel_hidden_size=8, hidden_size=384):
        super().__init__()
        self.lidar_channels, self.patch_size = int(lidar_channels), int(patch_size)
        self.lidar = nn.Conv2d(self.lidar_channels, pixel_hidden_size, 1, bias=False)
        self.vehicle = nn.Conv2d(1, pixel_hidden_size, 1, bias=False)
        self.road = nn.Conv2d(1, pixel_hidden_size, 1, bias=False)
        self.patch = nn.Linear(patch_size * patch_size * pixel_hidden_size, hidden_size, bias=False)
        self.global_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self._norms = {}

    def forward(self, raw_pc: torch.Tensor, layout_mask: torch.Tensor):
        _require_bchw(raw_pc, layout_mask, self.lidar_channels + 2)
        parts = {"lidar": self.lidar(raw_pc), "vehicle": self.vehicle(layout_mask[:, :1]), "road": self.road(layout_mask[:, 1:2])}
        cell = sum(parts.values())
        self._norms = modality_norms_and_ratios(parts, channel_dim=1)
        b, c, h, w = cell.shape
        if h % self.patch_size or w % self.patch_size: raise ValueError("Condition size must divide PixelDiT patch size")
        grid = h // self.patch_size
        pixel = cell.permute(0, 2, 3, 1).reshape(b, grid, self.patch_size, grid, self.patch_size, c).permute(0, 1, 3, 2, 4, 5).reshape(b, grid * grid, self.patch_size * self.patch_size, c)
        patch = self.patch(pixel.flatten(2))
        return pixel, patch, self.global_proj(patch.mean(dim=1))

    def modality_norms(self): return self._norms


class BEVPixelDiTS(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        cfg, cond = model_cfg.get("pixeldit", {}), model_cfg.get("condition", {})
        self.input_size = int(cfg.get("input_size", 256)); patch = int(cfg.get("patch_size", 16))
        hidden, pixel_hidden = int(cfg.get("hidden_size", 384)), int(cfg.get("pixel_hidden_size", 8))
        lidar_channels = int(cond.get("lidar_channels", 32)); gate = float(cond.get("gate_init", .1))
        self.pc_encoder = DynamicKNNPillarEncoder(3, lidar_channels, self.input_size)
        self.condition_encoder = PixelDiTConditionEncoder(lidar_channels, patch, pixel_hidden, hidden)
        self.core = PixelDiTCore(self.input_size, patch, int(cfg.get("in_channels", 3)), hidden,
                                 int(cfg.get("num_groups", 6)), int(cfg.get("patch_depth", 8)), pixel_hidden,
                                 int(cfg.get("pixel_depth", 4)), bool(cfg.get("pit_adaln_post_modulation", True)),
                                 float(cfg.get("time_scale", 1.0)), bool(cfg.get("repa", False)))
        self.g_pixel, self.g_patch, self.g_global = (nn.Parameter(torch.tensor(gate)) for _ in range(3))
        self._norms = {}

    def get_raw_pc_bev(self, points): return self.pc_encoder(points)

    def forward(self, xt, t, raw_pc, layout_mask):
        pixel, patch, global_value = self.condition_encoder(raw_pc, layout_mask)
        self._norms = {"pixel_feature_norm": pixel.detach().float().norm(dim=-1).mean(),
                       "patch_feature_norm": patch.detach().float().norm(dim=-1).mean(),
                       "global_feature_norm": global_value.detach().float().norm(dim=-1).mean()}
        return self.core(xt, t, pixel, patch, global_value, self.g_pixel, self.g_patch, self.g_global)

    def condition_diagnostics(self):
        return {"gate_pixel": self.g_pixel.detach(), "gate_patch": self.g_patch.detach(),
                "gate_global": self.g_global.detach(), **self._norms, **self.condition_encoder.modality_norms()}

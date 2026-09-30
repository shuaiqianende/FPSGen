"""FPSGen wrapper for DiP-S/16-BEV."""

from __future__ import annotations

import torch
import torch.nn as nn

from .condition import _require_bchw
from .dip_core import DiPCore


class DiPConditionEncoder(nn.Module):
    """Bias-free patch-aligned DiP condition tokens and global summary."""

    def __init__(self, lidar_channels=32, layout_channels=2, patch_size=16, hidden_size=384):
        super().__init__()
        self.in_channels = int(lidar_channels + layout_channels)
        self.patch_size, self.hidden_size = int(patch_size), int(hidden_size)
        self.patch_proj = nn.Linear(self.in_channels * self.patch_size * self.patch_size, self.hidden_size, bias=False)
        self.global_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, raw_pc: torch.Tensor, layout: torch.Tensor):
        x = _require_bchw(raw_pc, layout, self.in_channels)
        b, _, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Condition size must divide DiP patch size")
        gh, gw = h // self.patch_size, w // self.patch_size
        patches = x.reshape(b, self.in_channels, gh, self.patch_size, gw, self.patch_size).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, -1)
        patch = self.patch_proj(patches)
        return patch, self.global_proj(patch.mean(dim=1))


class BEVDiPS(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
        cfg, cond = model_cfg.get("dip", {}), model_cfg.get("condition", {})
        self.input_size = int(cfg.get("input_size", 256))
        self.pc_encoder = DynamicKNNPillarEncoder(in_channels=3, out_channels=int(cond.get("lidar_channels", 32)), grid_size=self.input_size)
        self.condition_encoder = DiPConditionEncoder(int(cond.get("lidar_channels", 32)), int(cond.get("layout_channels", 2)), int(cfg.get("patch_size", 16)), int(cfg.get("hidden_size", 384)))
        self.core = DiPCore(input_size=self.input_size, patch_size=int(cfg.get("patch_size", 16)), in_channels=int(cfg.get("in_channels", 3)), hidden_size=int(cfg.get("hidden_size", 384)), num_groups=int(cfg.get("num_groups", 6)), num_cond_blocks=int(cfg.get("num_cond_blocks", 8)), local_channels=tuple(cfg.get("local_channels", [64, 128, 256, 512])))
        self.patch_gate = nn.Parameter(torch.tensor(float(cond.get("gate_init", 0.1))))
        self.global_gate = nn.Parameter(torch.tensor(float(cond.get("gate_init", 0.1))))

    def get_raw_pc_bev(self, points: torch.Tensor) -> torch.Tensor:
        return self.pc_encoder(points)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, raw_pc: torch.Tensor, layout_mask: torch.Tensor) -> torch.Tensor:
        patch, global_condition = self.condition_encoder(raw_pc, layout_mask)
        return self.core(xt, t, self.patch_gate.to(xt.dtype) * patch, self.global_gate.to(xt.dtype) * global_condition)

"""FPSGen BEV wrapper around the official-topology DiC-S core."""

from __future__ import annotations

import torch
import torch.nn as nn

from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
from .condition import DiCConditionEncoder
from .dic_core import DiCCore


class BEVDiCS(nn.Module):
    """Direct 3x256x256 BEV velocity model with zero-gated spatial conditions."""

    def __init__(self, model_cfg):
        super().__init__()
        dic = model_cfg.get("dic", {})
        cond = model_cfg.get("condition", {})
        self.input_size = int(dic.get("input_size", 256))
        self.time_scale = float(model_cfg.get("flow", {}).get("time_scale", 1000.0))
        hidden = int(dic.get("hidden_size", 96))
        self.pc_encoder = DynamicKNNPillarEncoder(
            in_channels=3, out_channels=int(cond.get("lidar_channels", 32)), grid_size=self.input_size
        )
        self.condition_encoder = DiCConditionEncoder(
            lidar_channels=int(cond.get("lidar_channels", 32)),
            layout_channels=int(cond.get("layout_channels", 2)), hidden_size=hidden,
        )
        self.core = DiCCore(
            input_size=self.input_size, in_channels=int(dic.get("in_channels", 3)),
            hidden_size=hidden, depth=tuple(dic.get("depth", [6, 6, 5, 6, 6])),
            mult_channels=tuple(dic.get("mult_channels", [1, 2, 4, 2, 1])),
            skip_stride=int(dic.get("skip_stride", 3)),
        )
        gate_init = float(cond.get("gate_init", 0.0))
        self.condition_gates = nn.Parameter(torch.full((5,), gate_init))

    def get_raw_pc_bev(self, points: torch.Tensor) -> torch.Tensor:
        return self.pc_encoder(points)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, raw_pc: torch.Tensor,
                layout_mask: torch.Tensor) -> torch.Tensor:
        c0, c1, c2 = self.condition_encoder(raw_pc, layout_mask)
        maps = (c0, c1, c2, c1, c0)
        def stage_add(stage, x):
            condition = maps[stage]
            if x.shape != condition.shape:
                raise RuntimeError(f"DiC condition stage {stage} shape mismatch: {x.shape} vs {condition.shape}")
            return x + self.condition_gates[stage].to(x.dtype) * condition
        return self.core(xt, t * self.time_scale, stage_add)

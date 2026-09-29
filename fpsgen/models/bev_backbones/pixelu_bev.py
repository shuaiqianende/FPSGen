"""FPSGen adapter for the width-scaled PixelU-S-BEV backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
from .condition import PixelUConditionEncoder
from .pixelu_core import UiTCore


class BEVPixelUS(nn.Module):
    """PixelU-B-16 topology, scaled to 384 width and spatially conditioned."""

    def __init__(self, model_cfg):
        super().__init__()
        pixelu = model_cfg.get("pixelu", {})
        cond = model_cfg.get("condition", {})
        self.input_size = int(pixelu.get("input_size", 256))
        self.time_scale = float(model_cfg.get("flow", {}).get("time_scale", 1000.0))
        hidden = int(pixelu.get("hidden_size", 384))
        patch_size = int(pixelu.get("patch_size", 16))
        context_tokens = int(cond.get("context_tokens", pixelu.get("in_context_len", 32)))
        self.pc_encoder = DynamicKNNPillarEncoder(
            in_channels=3, out_channels=int(cond.get("lidar_channels", 32)), grid_size=self.input_size
        )
        self.condition_encoder = PixelUConditionEncoder(
            lidar_channels=int(cond.get("lidar_channels", 32)),
            layout_channels=int(cond.get("layout_channels", 2)), patch_size=patch_size,
            bottleneck_dim=int(pixelu.get("bottleneck_dim", 64)), hidden_size=hidden,
            context_tokens=context_tokens,
        )
        self.core = UiTCore(
            input_size=self.input_size, patch_size=patch_size,
            in_channels=int(pixelu.get("in_channels", 3)), hidden_size=hidden,
            num_heads=int(pixelu.get("num_heads", 6)), depth=tuple(pixelu.get("depth", [4, 4, 4])),
            bottleneck_dim=int(pixelu.get("bottleneck_dim", 64)), in_context_len=context_tokens,
            skip=bool(pixelu.get("skip", True)), skip_all_blocks=bool(pixelu.get("skip_all_blocks", True)),
        )
        gate_init = float(cond.get("gate_init", 0.0))
        self.g_patch = nn.Parameter(torch.tensor(gate_init))
        self.g_context = nn.Parameter(torch.tensor(gate_init))

    def get_raw_pc_bev(self, points: torch.Tensor) -> torch.Tensor:
        return self.pc_encoder(points)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, raw_pc: torch.Tensor,
                layout_mask: torch.Tensor) -> torch.Tensor:
        patch, context = self.condition_encoder(raw_pc, layout_mask)
        return self.core(xt, t * self.time_scale, patch, context, self.g_patch, self.g_context)

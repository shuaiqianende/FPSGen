"""FPSGen wrapper for DiP-S/16-BEV."""

from __future__ import annotations

import torch
import torch.nn as nn

from .condition import _require_bchw, modality_norms_and_ratios
from .condition_config import condition_options
from .dip_core import DiPCore


class DiPConditionEncoder(nn.Module):
    """Bias-free patch-aligned DiP condition tokens and global summary."""

    def __init__(self, lidar_channels=32, layout_channels=2, patch_size=16, hidden_size=384,
                 fusion="shared", native=False, local_channels=(64, 128, 256, 512),
                 bottleneck_dim=None, local_condition_width=16):
        super().__init__()
        self.lidar_channels, self.layout_channels = int(lidar_channels), int(layout_channels)
        self.in_channels = self.lidar_channels + self.layout_channels
        self.patch_size, self.hidden_size = int(patch_size), int(hidden_size)
        self.fusion, self.native = str(fusion), bool(native)
        self.bottleneck_dim = None if bottleneck_dim is None else int(bottleneck_dim)
        if self.fusion == "shared":
            if self.bottleneck_dim is None:
                self.patch_proj = nn.Linear(self.in_channels * self.patch_size * self.patch_size, self.hidden_size, bias=False)
            else:
                self.patch_proj1 = nn.Linear(self.in_channels * self.patch_size * self.patch_size, self.bottleneck_dim, bias=False)
                self.patch_proj2 = nn.Linear(self.bottleneck_dim, self.hidden_size, bias=False)
        elif self.fusion == "separate":
            output_dim = self.hidden_size if self.bottleneck_dim is None else self.bottleneck_dim
            self.lidar_patch_proj = nn.Linear(self.lidar_channels * self.patch_size * self.patch_size, output_dim, bias=False)
            self.vehicle_patch_proj = nn.Linear(self.patch_size * self.patch_size, output_dim, bias=False)
            self.road_patch_proj = nn.Linear(self.patch_size * self.patch_size, output_dim, bias=False)
            if self.bottleneck_dim is not None:
                self.patch_proj2 = nn.Linear(self.bottleneck_dim, self.hidden_size, bias=False)
        else:
            raise ValueError("DiP condition fusion must be shared or separate")
        self.global_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self._modality_norms = {}
        if self.native:
            c0, c1, c2, c3 = (int(value) for value in local_channels)
            local_width = int(local_condition_width)
            self.local0 = nn.Conv2d(self.in_channels, local_width, 1, bias=False)
            self.local1 = nn.Conv2d(local_width, local_width, 3, stride=2, padding=1, bias=False)
            self.local2 = nn.Conv2d(local_width, local_width, 3, stride=2, padding=1, bias=False)
            self.local3 = nn.Conv2d(local_width, local_width, 3, stride=2, padding=1, bias=False)
            self.local4 = nn.Conv2d(local_width, local_width, 3, stride=2, padding=1, bias=False)
            self.local_out = nn.ModuleList(
                nn.Conv2d(local_width, channels, 1, bias=False)
                for channels in (c0, c1, c2, c3, c3)
            )

    def forward(self, raw_pc: torch.Tensor, layout: torch.Tensor):
        x = _require_bchw(raw_pc, layout, self.in_channels)
        b, _, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Condition size must divide DiP patch size")
        gh, gw = h // self.patch_size, w // self.patch_size
        patches = x.reshape(b, self.in_channels, gh, self.patch_size, gw, self.patch_size).permute(0, 2, 4, 1, 3, 5)
        if self.fusion == "shared":
            patch_input = patches.reshape(b, gh * gw, -1)
            patch = self.patch_proj(patch_input) if self.bottleneck_dim is None else self.patch_proj2(self.patch_proj1(patch_input))
            self._modality_norms = {}
        else:
            components = {
                "lidar": self.lidar_patch_proj(patches[:, :, :, :self.lidar_channels].reshape(b, gh * gw, -1)),
                "vehicle": self.vehicle_patch_proj(patches[:, :, :, self.lidar_channels:self.lidar_channels + 1].reshape(b, gh * gw, -1)),
                "road": self.road_patch_proj(patches[:, :, :, self.lidar_channels + 1:].reshape(b, gh * gw, -1)),
            }
            patch = sum(components.values())
            self._modality_norms = modality_norms_and_ratios(components, channel_dim=-1)
            if self.bottleneck_dim is not None:
                patch = self.patch_proj2(patch)
        return patch, self.global_proj(patch.mean(dim=1))

    def modality_norms(self):
        """Detached C3 component norms for low-frequency diagnostic logging."""
        return self._modality_norms

    def local_conditions(self, raw_pc: torch.Tensor, layout: torch.Tensor):
        if not self.native:
            raise RuntimeError("Native DiP local conditions were not enabled")
        x = _require_bchw(raw_pc, layout, self.in_channels)
        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        patches = x.reshape(b, self.in_channels, gh, self.patch_size, gw, self.patch_size).permute(0, 2, 4, 1, 3, 5).reshape(b * gh * gw, self.in_channels, self.patch_size, self.patch_size)
        maps = [self.local0(patches)]
        for layer in (self.local1, self.local2, self.local3, self.local4):
            maps.append(layer(maps[-1]))
        projected = [projection(value) for projection, value in zip(self.local_out, maps)]
        return tuple(value.reshape(b, gh * gw, *value.shape[1:]) for value in projected)


class BEVDiPS(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
        cfg, cond = model_cfg.get("dip", {}), model_cfg.get("condition", {})
        self.condition_options = condition_options(model_cfg)
        self.input_size = int(cfg.get("input_size", 256))
        self.pc_encoder = DynamicKNNPillarEncoder(in_channels=3, out_channels=int(cond.get("lidar_channels", 32)), grid_size=self.input_size)
        local_channels = tuple(cfg.get("local_channels", [64, 128, 256, 512]))
        compact_dim = int(cond.get("bottleneck_dim", 96)) if self.condition_options.api_version >= 2 else None
        self.condition_encoder = DiPConditionEncoder(int(cond.get("lidar_channels", 32)), int(cond.get("layout_channels", 2)), int(cfg.get("patch_size", 16)), int(cfg.get("hidden_size", 384)), self.condition_options.fusion, self.condition_options.native, local_channels, compact_dim, int(cond.get("local_condition_width", 16)))
        self.core = DiPCore(input_size=self.input_size, patch_size=int(cfg.get("patch_size", 16)), in_channels=int(cfg.get("in_channels", 3)), hidden_size=int(cfg.get("hidden_size", 384)), num_groups=int(cfg.get("num_groups", 6)), num_cond_blocks=int(cfg.get("num_cond_blocks", 8)), local_channels=local_channels)
        self.patch_gate = nn.Parameter(torch.tensor(self.condition_options.gate_init))
        self.global_gate = nn.Parameter(torch.tensor(self.condition_options.gate_init))
        if self.condition_options.native:
            self.local_gates = nn.Parameter(torch.full((5,), self.condition_options.gate_init))
        self._feature_norms = {}

    def get_raw_pc_bev(self, points: torch.Tensor) -> torch.Tensor:
        return self.pc_encoder(points)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, raw_pc: torch.Tensor, layout_mask: torch.Tensor) -> torch.Tensor:
        patch, global_condition = self.condition_encoder(raw_pc, layout_mask)
        spatial = self.patch_gate.to(xt.dtype) * patch if self.condition_options.spatial else None
        global_value = self.global_gate.to(xt.dtype) * global_condition if self.condition_options.global_modulation else None
        local = self.condition_encoder.local_conditions(raw_pc, layout_mask) if self.condition_options.native else None
        self._feature_norms = {"patch_norm": patch.detach().float().norm(dim=-1).mean(), "global_norm": global_condition.detach().float().norm(dim=-1).mean()}
        return self.core(xt, t, spatial, global_value, local, getattr(self, "local_gates", None))

    def condition_diagnostics(self):
        values = {"gate_spatial": self.patch_gate.detach(), "gate_global": self.global_gate.detach()}
        if hasattr(self, "local_gates"):
            values.update({f"gate_local_{index}": gate.detach() for index, gate in enumerate(self.local_gates)})
        return {**values, **self._feature_norms, **self.condition_encoder.modality_norms()}

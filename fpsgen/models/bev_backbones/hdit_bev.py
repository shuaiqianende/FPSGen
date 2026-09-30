"""FPSGen condition wrapper for the HDiT-S BEV velocity core."""

from __future__ import annotations

import torch
import torch.nn as nn

from .hdit_core import HDiTConditionEncoder, HDiTCore
from .condition_config import condition_options


class BEVHDiTS(nn.Module):
    """Official-topology HDiT-S adapted only for FPSGen BEV conditions."""

    def __init__(self, model_cfg):
        super().__init__()
        # Keep PyKeOps/PointPillar a runtime-only dependency.  This lets the
        # dense HDiT core remain CPU-testable in a minimal PyTorch2 env.
        from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
        cfg, cond = model_cfg.get("hdit", {}), model_cfg.get("condition", {})
        self.condition_options = condition_options(model_cfg)
        self.input_size = int(cfg.get("input_size", 256))
        widths = tuple(cfg.get("widths", [128, 256, 512]))
        self.pc_encoder = DynamicKNNPillarEncoder(in_channels=3, out_channels=int(cond.get("lidar_channels", 32)), grid_size=self.input_size)
        self.condition_encoder = HDiTConditionEncoder(
            int(cond.get("lidar_channels", 32)), int(cond.get("layout_channels", 2)), widths,
            int(cfg.get("patch_size", [4, 4])[0] if isinstance(cfg.get("patch_size", 4), list) else cfg.get("patch_size", 4)),
            int(cfg.get("mapping_width", 256)),
            self.condition_options.fusion,
            self.condition_options.native,
        )
        self.core = HDiTCore(
            input_size=self.input_size, in_channels=int(cfg.get("in_channels", 3)),
            patch_size=int(cfg.get("patch_size", [4, 4])[0] if isinstance(cfg.get("patch_size", 4), list) else cfg.get("patch_size", 4)),
            widths=widths, depths=tuple(cfg.get("depths", [2, 2, 4])), d_ffs=tuple(cfg.get("d_ffs", [384, 768, 1536])),
            d_head=int(cfg.get("self_attns", [{"d_head": 64}])[0].get("d_head", 64)),
            window_size=int(cfg.get("self_attns", [{"window_size": 8}])[0].get("window_size", 8)),
            mapping_width=int(cfg.get("mapping_width", 256)), mapping_depth=int(cfg.get("mapping_depth", 2)),
            mapping_d_ff=int(cfg.get("mapping_d_ff", 768)), dropout=tuple(cfg.get("dropout", [0.0, 0.0, 0.1])),
        )
        self.condition_gates = nn.Parameter(torch.full((5,), self.condition_options.gate_init))
        # Only new ablation configs opt into this parameter, so C0 checkpoints
        # retain their original state-dict contract.
        if self.condition_options.api_version >= 2:
            self.global_gate = nn.Parameter(torch.tensor(self.condition_options.gate_init))
            if self.condition_options.native:
                self.native_global_gates = nn.Parameter(torch.full((3,), self.condition_options.gate_init))
        self._feature_norms = {}

    def get_raw_pc_bev(self, points: torch.Tensor) -> torch.Tensor:
        return self.pc_encoder(points)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, raw_pc: torch.Tensor, layout_mask: torch.Tensor) -> torch.Tensor:
        c0, c1, c2, global_condition = self.condition_encoder(raw_pc, layout_mask)
        maps = (c0, c1, c2, c1, c0)
        def stage_add(stage: int, x: torch.Tensor) -> torch.Tensor:
            if not self.condition_options.spatial:
                return x
            condition = maps[stage]
            if x.shape != condition.shape:
                raise RuntimeError(f"HDiT condition stage {stage} mismatch: {tuple(x.shape)} vs {tuple(condition.shape)}")
            return x + self.condition_gates[stage].to(x.dtype) * condition
        self._feature_norms = {"spatial_norm": c0.detach().float().norm(dim=-1).mean()}
        if not self.condition_options.global_modulation:
            return self.core(xt, t, stage_add=stage_add)
        if self.condition_options.native:
            native = self.condition_encoder.native_global_conditions((c0, c1, c2))
            native = tuple(self.native_global_gates[index].to(xt.dtype) * value for index, value in enumerate(native))
            self._feature_norms["global_norm"] = native[0].detach().float().norm(dim=-1).mean()
            return self.core(xt, t, stage_add=stage_add, stage_global_conditions=native)
        gate = getattr(self, "global_gate", None)
        condition = global_condition if gate is None else gate.to(xt.dtype) * global_condition
        self._feature_norms["global_norm"] = condition.detach().float().norm(dim=-1).mean()
        return self.core(xt, t, global_condition=condition, stage_add=stage_add)

    def condition_diagnostics(self):
        values = {f"gate_spatial_{index}": gate.detach() for index, gate in enumerate(self.condition_gates)}
        if hasattr(self, "global_gate"):
            values["gate_global"] = self.global_gate.detach()
        if hasattr(self, "native_global_gates"):
            values.update({f"gate_native_{index}": gate.detach() for index, gate in enumerate(self.native_global_gates)})
        return {**values, **self._feature_norms, **self.condition_encoder.modality_norms()}

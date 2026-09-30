"""FPSGen wrapper for the NCSN++-Small BEV velocity core."""

from __future__ import annotations

import torch
import torch.nn as nn

from .ncsnpp_core import NCSNConditionEncoder, NCSNppCore
from .condition_config import condition_options


class BEVNCSNppS(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
        cfg, cond = model_cfg.get("ncsnpp", {}), model_cfg.get("condition", {})
        self.condition_options = condition_options(model_cfg)
        self.input_size = int(cfg.get("input_size", 256))
        nf, ch_mult = int(cfg.get("nf", 64)), tuple(cfg.get("ch_mult", [1, 1, 2, 2, 2, 2, 2]))
        self.pc_encoder = DynamicKNNPillarEncoder(in_channels=3, out_channels=int(cond.get("lidar_channels", 32)), grid_size=self.input_size)
        condition_nf = int(cond.get("condition_nf", 32)) if self.condition_options.api_version >= 2 else None
        self.condition_encoder = NCSNConditionEncoder(int(cond.get("lidar_channels", 32)), int(cond.get("layout_channels", 2)), nf=nf, ch_mult=ch_mult, temb_dim=nf * 4, fusion=self.condition_options.fusion, native=self.condition_options.native, condition_nf=condition_nf)
        self.core = NCSNppCore(input_size=self.input_size, in_channels=int(cfg.get("in_channels", 3)), nf=nf,
                                ch_mult=ch_mult, num_res_blocks=int(cfg.get("num_res_blocks", 2)),
                                attn_resolutions=tuple(cfg.get("attn_resolutions", [16])),
                                resblock_type=cfg.get("resblock_type", "biggan"), resamp_with_conv=bool(cfg.get("resamp_with_conv", True)),
                                fir=bool(cfg.get("fir", True)), fir_kernel=tuple(cfg.get("fir_kernel", [1, 3, 3, 1])),
                                skip_rescale=bool(cfg.get("skip_rescale", True)), progressive=cfg.get("progressive", "output_skip"),
                                progressive_input=cfg.get("progressive_input", "input_skip"), progressive_combine=cfg.get("progressive_combine", "sum"),
                                attention_type=cfg.get("attention_type", "ddpm"), init_scale=float(cfg.get("init_scale", 0.0)), conv_size=int(cfg.get("conv_size", 3)))
        self.condition_gates = nn.Parameter(torch.full((7,), self.condition_options.gate_init))
        self.global_gate = nn.Parameter(torch.tensor(self.condition_options.gate_init))
        if self.condition_options.native:
            self.native_global_gates = nn.Parameter(torch.full((7,), self.condition_options.gate_init))
        self.time_scale = float(cfg.get("time_scale", 1000.0))
        self._feature_norms = {}

    def get_raw_pc_bev(self, points: torch.Tensor) -> torch.Tensor:
        return self.pc_encoder(points)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, raw_pc: torch.Tensor, layout_mask: torch.Tensor) -> torch.Tensor:
        maps, global_condition = self.condition_encoder(raw_pc, layout_mask)
        spatial = tuple(self.condition_gates[i].to(xt.dtype) * value for i, value in enumerate(maps)) if self.condition_options.spatial else None
        global_value = self.global_gate.to(xt.dtype) * global_condition if self.condition_options.global_modulation else None
        self._feature_norms = {"spatial_norm": maps[0].detach().float().norm(dim=1).mean(), "global_norm": global_condition.detach().float().norm(dim=-1).mean()}
        if self.condition_options.native and self.condition_options.global_modulation:
            native = self.condition_encoder.native_global_conditions(maps)
            native = tuple(self.native_global_gates[index].to(xt.dtype) * value for index, value in enumerate(native))
            return self.core(xt, t * self.time_scale, spatial, None, native)
        return self.core(xt, t * self.time_scale, spatial, global_value)

    def condition_diagnostics(self):
        values = {f"gate_spatial_{index}": gate.detach() for index, gate in enumerate(self.condition_gates)}
        values["gate_global"] = self.global_gate.detach()
        if hasattr(self, "native_global_gates"):
            values.update({f"gate_native_{index}": gate.detach() for index, gate in enumerate(self.native_global_gates)})
        return {**values, **self._feature_norms, **self.condition_encoder.modality_norms()}

"""SynFlow-style and generic-U-Net BEV wrappers with one concat34 frontend."""
from __future__ import annotations

import torch
import torch.nn as nn

from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
from fpsgen.models.bev_conditioning.condition_packet import make_condition_packet
from fpsgen.models.bev_conditioning.spatial_ops import layout_boundary
from .synflow_core import SynConditionEncoder, SynFlowCore


class BEVSynFlowS(nn.Module):
    supports_condition_keep = True
    def __init__(self, model_cfg):
        super().__init__()
        cfg, condition = (model_cfg.get("synflow") or model_cfg.get("cracksegflow", {})), model_cfg.get("condition", {})
        self.input_size = int(cfg.get("input_size", 256))
        model_channels = int(cfg.get("model_channels", 64))
        multiplier = tuple(cfg.get("channel_mult", [1, 1, 2, 2, 4, 4]))
        channels = tuple(model_channels * int(value) for value in multiplier)
        injection_cfg = condition.get("injection", {})
        if isinstance(injection_cfg, str): injection_cfg = {"type": injection_cfg}
        self.pc_encoder = DynamicKNNPillarEncoder(3, 32, self.input_size)
        self.condition_encoder = SynConditionEncoder(channels)
        self.core = SynFlowCore(input_size=self.input_size, in_channels=int(cfg.get("in_channels", 3)),
                                model_channels=model_channels, channel_mult=multiplier,
                                num_res_blocks=int(cfg.get("num_res_blocks", 2)),
                                attention_spatial_resolutions=tuple(cfg.get("attention_spatial_resolutions", [32, 16, 8])),
                                num_head_channels=int(cfg.get("num_head_channels", 64)),
                                time_scale=float(cfg.get("time_scale", 1000.0)),
                                injection=str(injection_cfg.get("type", "spade")),
                                encoder_conditioned=bool(injection_cfg.get("encoder", True)),
                                middle_conditioned=bool(injection_cfg.get("middle", True)),
                                decoder_conditioned=bool(injection_cfg.get("decoder", True)),
                                boundary_gate=bool(condition.get("boundary", {}).get("enabled", False)))
        self._condition_norm = torch.tensor(0.)

    def get_raw_pc_bev(self, points): return self.pc_encoder(points)

    def forward(self, xt, t, raw_pc, layout_mask, condition_keep=None):
        packet = make_condition_packet(raw_pc, layout_mask, condition_keep)
        pyramid = self.condition_encoder(packet.map)
        self._condition_norm = pyramid[0].detach().float().square().mean().sqrt()
        boundary_cfg = self.core.boundary_gate
        boundary = layout_boundary(layout_mask, packet.keep) if boundary_cfg else None
        return self.core(xt, t, pyramid, boundary)

    def condition_diagnostics(self):
        return {"feature_norm": self._condition_norm, **self.core.condition_diagnostics()}

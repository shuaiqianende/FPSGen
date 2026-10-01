"""Paired PixelDiT generic/PixControl BEV wrappers using one concat34 source."""
from __future__ import annotations

import torch
import torch.nn as nn

from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
from fpsgen.models.bev_conditioning.condition_packet import make_condition_packet
from fpsgen.models.bev_conditioning.pixelcontrol_adapter import PixelControlAdapter, PixelControlEncoder
from .pixeldit_core import PixelDiTCore


class BEVPixelControlS(nn.Module):
    """P0 uses generic token/global add; P1 uses only residual control paths."""
    supports_condition_keep = True
    def __init__(self, model_cfg):
        super().__init__()
        pixel, control, condition = model_cfg.get("pixeldit", {}), model_cfg.get("pixelcontrol", {}), model_cfg.get("condition", {})
        self.input_size = int(pixel.get("input_size", 256))
        hidden, patch_depth = int(pixel.get("hidden_size", 384)), int(pixel.get("patch_depth", 8))
        self.mode = str(control.get("mode", "pixelcontrol"))
        if self.mode not in {"generic", "pixelcontrol"}:
            raise ValueError("PixelControl mode must be 'generic' or 'pixelcontrol'")
        self.pc_encoder = DynamicKNNPillarEncoder(3, 32, self.input_size)
        self.condition_encoder = PixelControlEncoder(hidden, int(control.get("base_channels", 16)), int(control.get("max_channels", 128)))
        self.core = PixelDiTCore(self.input_size, int(pixel.get("patch_size", 16)), int(pixel.get("in_channels", 3)), hidden,
                                 int(pixel.get("num_groups", 6)), patch_depth, int(pixel.get("pixel_hidden_size", 8)),
                                 int(pixel.get("pixel_depth", 4)), bool(pixel.get("pit_adaln_post_modulation", True)),
                                 float(pixel.get("time_scale", 1.0)), bool(pixel.get("repa", False)))
        self.generic_global = nn.Linear(hidden, hidden, bias=False)
        self.generic_patch_gate = nn.Parameter(torch.tensor(float(condition.get("gate_init", 0.1))))
        self.generic_global_gate = nn.Parameter(torch.tensor(float(condition.get("gate_init", 0.1))))
        self.adapter = PixelControlAdapter(hidden, patch_depth, float(control.get("gate_init", 1.0)))
        self._norms = {}

    def get_raw_pc_bev(self, points): return self.pc_encoder(points)

    def forward(self, xt, t, raw_pc, layout_mask, condition_keep=None):
        packet = make_condition_packet(raw_pc, layout_mask, condition_keep)
        tokens = self.condition_encoder(packet.map)
        self._norms = {"control_token_norm": tokens.detach().float().square().mean().sqrt()}
        if self.mode == "generic":
            global_value = self.generic_global(tokens.mean(dim=1))
            return self.core(xt, t, patch_condition=tokens, global_condition=global_value,
                             patch_gate=self.generic_patch_gate, global_gate=self.generic_global_gate)
        return self.core(xt, t, patch_adapter=self.adapter, control_tokens=tokens)

    def condition_diagnostics(self):
        if self.mode == "generic":
            values = {"generic_patch_gate": self.generic_patch_gate.detach(),
                      "generic_global_gate": self.generic_global_gate.detach()}
        else:
            values = {f"adapter_gate_{index}": value.detach() for index, value in enumerate(self.adapter.gates)}
            values.update(self.adapter.last_ratios)
        return {**self._norms, **values}

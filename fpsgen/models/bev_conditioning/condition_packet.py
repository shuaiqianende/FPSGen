"""The unambiguous [LiDAR, vehicle, road] conditioning contract.

The packet deliberately masks each source before it is concatenated.  This
means an inactive source is independent of its tensor contents, including a
valid but empty layout frame.  Older backbones do not use this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ConditionPacket:
    map: torch.Tensor
    keep: torch.Tensor


def _validate(raw_pc: torch.Tensor, layout_mask: torch.Tensor, keep: torch.Tensor) -> None:
    if raw_pc.ndim != 4 or raw_pc.shape[1] != 32:
        raise ValueError("raw_pc must have shape [B,32,H,W]")
    if layout_mask.ndim != 4 or layout_mask.shape[1] != 2:
        raise ValueError("layout_mask must have shape [B,2,H,W]")
    if raw_pc.shape[0] != layout_mask.shape[0] or raw_pc.shape[-2:] != layout_mask.shape[-2:]:
        raise ValueError("condition sources must share batch and spatial shape")
    if keep.shape != (raw_pc.shape[0], 3):
        raise ValueError("condition_keep must have shape [B,3] in [LiDAR, vehicle, road] order")


def make_condition_packet(raw_pc: torch.Tensor, layout_mask: torch.Tensor,
                          condition_keep: torch.Tensor | None = None) -> ConditionPacket:
    """Build a masked 34-channel condition map with an explicit state.

    ``None`` is a backwards-friendly all-enabled state for direct model use.
    New training paths always pass an explicit keep mask.
    """
    if condition_keep is None:
        condition_keep = torch.ones((raw_pc.shape[0], 3), dtype=torch.bool, device=raw_pc.device)
    keep = condition_keep.to(device=raw_pc.device, dtype=torch.bool)
    _validate(raw_pc, layout_mask, keep)
    scales = keep.to(dtype=raw_pc.dtype).view(raw_pc.shape[0], 3, 1, 1)
    masked = torch.cat((raw_pc * scales[:, :1], layout_mask[:, :1] * scales[:, 1:2],
                        layout_mask[:, 1:2] * scales[:, 2:3]), dim=1)
    return ConditionPacket(map=masked, keep=keep)

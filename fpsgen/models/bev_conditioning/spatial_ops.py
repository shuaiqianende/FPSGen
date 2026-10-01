"""Small geometry-only helpers used by optional boundary modulation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_edge(mask: torch.Tensor) -> torch.Tensor:
    """One-cell morphological boundary for a binary [B,1,H,W] map."""
    binary = (mask > 0).to(mask.dtype)
    dilated = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - binary, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp_(0.0, 1.0)


def layout_boundary(layout_mask: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Vehicle/road-only boundary map respecting the explicit state."""
    if layout_mask.shape[1] != 2:
        raise ValueError("layout_mask needs vehicle and road channels")
    vehicle = binary_edge(layout_mask[:, :1]) * keep[:, 1:2, None, None].to(layout_mask.dtype)
    road = binary_edge(layout_mask[:, 1:2]) * keep[:, 2:3, None, None].to(layout_mask.dtype)
    return torch.maximum(vehicle, road)

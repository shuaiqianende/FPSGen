"""The original Gaussian-to-target linear Flow Matching path."""

from __future__ import annotations

from typing import Optional

import torch

from .base import FlowPath, PathOutput, broadcast_time


class LinearPath(FlowPath):
    """``x_t=(1-t)x0+t*x1`` with constant velocity ``x1-x0``."""

    def sample(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[dict] = None,
    ) -> PathOutput:
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must have identical shapes")
        t_view = broadcast_time(t, x0)
        return PathOutput(
            xt=(1.0 - t_view) * x0 + t_view * x1,
            velocity=x1 - x0,
        )


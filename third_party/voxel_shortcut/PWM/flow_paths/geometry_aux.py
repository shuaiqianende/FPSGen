"""Deterministic coarse-geometry auxiliary probability path."""

from __future__ import annotations

from typing import Optional

import torch

from .base import FlowPath, PathOutput, broadcast_time
from .guides.base import GeometryGuide
from .schedules.bump import BumpSchedule
from .schedules.interpolation import InterpolationSchedule


class GeometryAuxPath(FlowPath):
    """Add a soft, endpoint-preserving guide correction to an interpolant."""

    def __init__(
        self,
        interpolation_schedule: InterpolationSchedule,
        bump_schedule: BumpSchedule,
        guide: GeometryGuide,
        aux_lambda: float = 0.25,
    ):
        if aux_lambda < 0:
            raise ValueError("aux_lambda must be non-negative")
        self.interpolation_schedule = interpolation_schedule
        self.bump_schedule = bump_schedule
        self.guide = guide
        self.aux_lambda = float(aux_lambda)

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
        alpha = self.interpolation_schedule.alpha(t_view)
        beta = self.interpolation_schedule.beta(t_view)
        d_alpha = self.interpolation_schedule.d_alpha(t_view)
        d_beta = self.interpolation_schedule.d_beta(t_view)
        linear_xt = alpha * x0 + beta * x1
        base_velocity = d_alpha * x0 + d_beta * x1

        # lambda=0 is an exact baseline path and does not require guide context.
        if self.aux_lambda == 0.0:
            return PathOutput(xt=linear_xt, velocity=base_velocity)

        guide = self.guide(x0=x0, x1=x1, context=context)
        if guide.shape != x1.shape:
            raise ValueError("geometry guide must have the same shape as x1")
        eta = guide - 0.5 * (x0 + x1)
        bump = self.bump_schedule.value(t_view)
        dbump = self.bump_schedule.derivative(t_view)
        return PathOutput(
            xt=linear_xt + self.aux_lambda * bump * eta,
            velocity=base_velocity + self.aux_lambda * dbump * eta,
            diagnostics={"guide": guide, "eta": eta, "bump": bump},
        )


"""Common interfaces for probability paths used by flow-matching models.

The model and the training loop should only consume :class:`PathOutput`.
Concrete interpolants, schedules, and geometry guides live in their own
modules so that changing a training path does not change the VoxelFlow
network or its sampler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class PathOutput:
    """State, analytic velocity, and optional path diagnostics."""

    xt: torch.Tensor
    velocity: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def broadcast_time(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-sample time vector over an arbitrary state tensor."""

    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=reference.device, dtype=reference.dtype)
    else:
        t = t.to(device=reference.device, dtype=reference.dtype)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim != 1 or t.shape[0] != reference.shape[0]:
        raise ValueError(
            f"t must have shape [{reference.shape[0]}], got {tuple(t.shape)}"
        )
    return t.reshape(t.shape[0], *([1] * (reference.ndim - 1)))


class FlowPath(ABC):
    """Abstract probability path returning both state and exact velocity."""

    @abstractmethod
    def sample(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[dict] = None,
    ) -> PathOutput:
        """Evaluate the path and its analytic time derivative."""


"""Voxel geometry and exact-total point allocation utilities."""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class VoxelSpec:
    """Axis-aligned 100 x 100 x 10 grid over [-50, 50] x [-50, 50] x [-5, 5]."""

    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -5.0
    z_max: float = 5.0
    nx: int = 100
    ny: int = 100
    nz: int = 10

    def __post_init__(self):
        if not self.x_min < self.x_max:
            raise ValueError("x_min must be smaller than x_max")
        if not self.y_min < self.y_max:
            raise ValueError("y_min must be smaller than y_max")
        if not self.z_min < self.z_max:
            raise ValueError("z_min must be smaller than z_max")
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("voxel grid dimensions must be positive")

    def to_dict(self):
        return {
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "nz": int(self.nz),
        }

    @classmethod
    def from_dict(cls, values: Optional[Mapping] = None):
        """Build a grid from YAML/checkpoint values; None keeps legacy defaults."""
        if values is None:
            return cls()
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown voxel grid fields: {sorted(unknown)}")
        return cls(**dict(values))

    @property
    def num_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            (self.x_max - self.x_min) / self.nx,
            (self.y_max - self.y_min) / self.ny,
            (self.z_max - self.z_min) / self.nz,
        )

    def point_to_indices(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return flattened voxel IDs and an in-range mask for [..., 3] points."""
        sx, sy, sz = self.size
        origin = points.new_tensor([self.x_min, self.y_min, self.z_min])
        scaled = torch.floor((points[..., :3] - origin) / points.new_tensor([sx, sy, sz])).long()
        valid = (
            (scaled[..., 0] >= 0) & (scaled[..., 0] < self.nx) &
            (scaled[..., 1] >= 0) & (scaled[..., 1] < self.ny) &
            (scaled[..., 2] >= 0) & (scaled[..., 2] < self.nz)
        )
        clipped = scaled.clone()
        clipped[..., 0].clamp_(0, self.nx - 1)
        clipped[..., 1].clamp_(0, self.ny - 1)
        clipped[..., 2].clamp_(0, self.nz - 1)
        ids = clipped[..., 2] * (self.nx * self.ny) + clipped[..., 1] * self.nx + clipped[..., 0]
        return ids, valid

    def voxel_centers(self, voxel_ids: torch.Tensor) -> torch.Tensor:
        """Map flattened IDs to xyz centers."""
        ids = voxel_ids.long()
        xy = self.nx * self.ny
        iz = torch.div(ids, xy, rounding_mode="floor")
        rem = ids % xy
        iy = torch.div(rem, self.nx, rounding_mode="floor")
        ix = rem % self.nx
        sx, sy, sz = self.size
        return torch.stack([
            self.x_min + (ix.to(torch.float32) + 0.5) * sx,
            self.y_min + (iy.to(torch.float32) + 0.5) * sy,
            self.z_min + (iz.to(torch.float32) + 0.5) * sz,
        ], dim=-1).to(voxel_ids.device)

    def counts(self, points: torch.Tensor) -> torch.Tensor:
        """Count in-range points, returning a vector of length num_voxels."""
        ids, valid = self.point_to_indices(points)
        return torch.bincount(ids[valid], minlength=self.num_voxels)


def allocate_counts(density: torch.Tensor, total_points: int, generator=None) -> torch.Tensor:
    """Allocate an exact total across voxels with stochastic residual rounding."""
    if density.ndim != 1:
        raise ValueError("density must be a 1-D voxel vector")
    if int(total_points) < 0:
        raise ValueError("total_points must be non-negative")
    weights = density.clamp_min(0).float()
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        weights = torch.ones_like(weights)
    expected = weights / weights.sum() * int(total_points)
    counts = torch.floor(expected).long()
    remaining = int(total_points - counts.sum().item())
    if remaining > 0:
        residual = expected - counts
        if residual.sum() <= 0:
            residual = torch.ones_like(residual)
        extra = torch.multinomial(residual / residual.sum(), remaining, replacement=True, generator=generator)
        counts.scatter_add_(0, extra, torch.ones_like(extra, dtype=counts.dtype))
    return counts


def sample_points_in_voxels(spec: VoxelSpec, counts: torch.Tensor, generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample points in each voxel; return points and matching voxel IDs."""
    counts = counts.long()
    ids = torch.repeat_interleave(torch.arange(spec.num_voxels, device=counts.device), counts)
    centers = spec.voxel_centers(ids).to(dtype=torch.float32)
    sx, sy, sz = spec.size
    offsets = torch.rand((ids.numel(), 3), device=counts.device, generator=generator)
    offsets = (offsets - 0.5) * offsets.new_tensor([sx, sy, sz])
    return centers + offsets, ids


def clamp_points_to_voxels(
    spec: VoxelSpec,
    points: torch.Tensor,
    voxel_ids: torch.Tensor,
    margin: float = 1e-4,
) -> torch.Tensor:
    """Optionally keep a refined point inside its source voxel."""
    if points.shape != (voxel_ids.numel(), 3):
        raise ValueError("points and voxel_ids must describe the same points")
    sx, sy, sz = spec.size
    if margin < 0 or margin * 2 >= min(sx, sy, sz):
        raise ValueError("margin must be non-negative and smaller than a voxel radius")
    centers = spec.voxel_centers(voxel_ids).to(device=points.device, dtype=points.dtype)
    half_size = points.new_tensor([sx, sy, sz]) * 0.5 - margin
    return torch.maximum(torch.minimum(points, centers + half_size), centers - half_size)

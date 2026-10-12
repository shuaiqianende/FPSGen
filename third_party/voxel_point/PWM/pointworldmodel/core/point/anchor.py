"""Canonical voxel-centre Gaussian anchors for PointFlow."""

from __future__ import annotations

from typing import Tuple

import torch

from pointworldmodel.core.voxel.grid import VoxelSpec


def sample_gaussian_anchors(
    spec: VoxelSpec,
    counts: torch.Tensor,
    noise_std_ratio: float = 0.25,
    *,
    clip_to_voxel: bool = False,
    generator=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample source points at voxel centres with per-axis Gaussian noise.

    ``counts`` determines the exact number of anchors in every voxel.  The
    returned ``voxel_ids`` retain that provenance even when unclipped Gaussian
    noise moves a point across a cell boundary.
    """
    if counts.ndim != 1 or counts.numel() != spec.num_voxels:
        raise ValueError(
            f"counts must have shape [{spec.num_voxels}], got {tuple(counts.shape)}"
        )
    if noise_std_ratio < 0:
        raise ValueError("noise_std_ratio must be non-negative")
    counts = counts.to(dtype=torch.long)
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")

    voxel_ids = torch.repeat_interleave(
        torch.arange(spec.num_voxels, device=counts.device), counts
    )
    centers = spec.voxel_centers(voxel_ids).to(dtype=torch.float32)
    std = centers.new_tensor(spec.size) * float(noise_std_ratio)
    noise = torch.randn(
        centers.shape,
        device=centers.device,
        dtype=centers.dtype,
        generator=generator,
    ) * std
    anchors = centers + noise
    if clip_to_voxel:
        half = centers.new_tensor(spec.size) * 0.5 - centers.new_tensor(1e-5)
        anchors = torch.maximum(torch.minimum(anchors, centers + half), centers - half)
    return anchors, voxel_ids


def sample_pointflow_anchors(point_flow, counts, *, generator=None):
    """Sample anchors using the convention stored in a PointFlow checkpoint.

    Evaluation code should call this adapter instead of retyping anchor
    hyperparameters.  That keeps old dataset-specific visualizers aligned
    with the student checkpoint when the noise ratio or clipping policy is
    changed in a config.
    """
    hparams = point_flow.hparams
    return sample_gaussian_anchors(
        point_flow.spec,
        counts,
        noise_std_ratio=float(getattr(hparams, "anchor_noise_std_ratio", 0.25)),
        clip_to_voxel=bool(getattr(hparams, "anchor_clip_to_voxel", False)),
        generator=generator,
    )

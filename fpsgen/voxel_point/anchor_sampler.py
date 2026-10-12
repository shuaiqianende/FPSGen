"""Gaussian anchors truncated to the voxel ∩ open circular ROI; no clamping."""

import math

import torch


def sample_gaussian_anchors(spec, counts, *, generator, noise_std_ratio=.25, max_rounds=10000):
    if generator is None or noise_std_ratio <= 0:
        raise ValueError("Explicit generator and positive Gaussian std required")
    if counts.shape != (spec.num_voxels,) or counts.is_floating_point() or bool((counts < 0).any()):
        raise ValueError("Nonnegative flat integer counts required")
    valid = spec.dense_to_flat(spec.valid_circle_mask(device=counts.device))
    if bool((counts[~valid] != 0).any()):
        raise ValueError("Counts outside voxel/disk intersection")
    ids = torch.repeat_interleave(torch.arange(spec.num_voxels, device=counts.device), counts)
    indices = spec.unflatten_voxel_ids(ids).double()
    size = indices.new_tensor(spec.size)
    lower = indices.new_tensor(spec.origin) + indices * size
    upper = lower + size
    center = lower + size * .5
    # Tight axis bounds of each cell/disk intersection make corner slivers
    # practical to sample. Inverse-CDF truncation + disk rejection preserves
    # the original Gaussian conditional law rather than piling up at edges.
    if spec.circle_mask:
        closest = torch.maximum(lower[:, :2], torch.zeros_like(lower[:, :2])) + torch.minimum(upper[:, :2], torch.zeros_like(upper[:, :2]))
        axis_max = (spec.xy_radius ** 2 - closest.flip(-1).square()).clamp_min(0).sqrt()
        lower[:, :2] = torch.maximum(lower[:, :2], -axis_max)
        upper[:, :2] = torch.minimum(upper[:, :2], axis_max)
    sigma = size * noise_std_ratio
    cdf_lower = .5 * (1 + torch.erf((lower - center) / (sigma * math.sqrt(2))))
    cdf_upper = .5 * (1 + torch.erf((upper - center) / (sigma * math.sqrt(2))))
    result = center.new_empty(center.shape, dtype=torch.float32)
    pending = torch.arange(len(ids), device=ids.device)
    for _ in range(max_rounds):
        if not pending.numel():
            return result, ids
        u = torch.rand((len(pending), 3), device=ids.device, generator=generator, dtype=torch.float64)
        q = cdf_lower[pending] + u * (cdf_upper[pending] - cdf_lower[pending])
        xyz = (center[pending] + sigma * math.sqrt(2) * torch.erfinv(2 * q - 1)).float()
        candidate_ids, point_valid = spec.voxelize_points(xyz, strict=False)
        accepted = point_valid.clone()
        accepted[point_valid] &= candidate_ids == ids[pending][point_valid]
        result[pending[accepted]] = xyz[accepted]
        pending = pending[~accepted]
    raise RuntimeError(f"Truncated Gaussian sampling failed for {len(pending)} anchors; no point duplication fallback")

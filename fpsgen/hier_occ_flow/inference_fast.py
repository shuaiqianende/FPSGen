"""GPU-resident, parity-oriented HS-OccFlow inference utilities.

These functions do not change learned weights or model equations. Inputs are
validated at their boundaries; dense fine-grid tensors are never constructed.
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

_VALID_PARENT_GPU = {}


@lru_cache(maxsize=8)
def _valid_parent_field_cpu(geometry):
    """Exact positive-volume sphere/AABB intersection field in ZYX layout."""
    import numpy as np
    out = []
    for n, origin, size in zip(geometry.parent_resolution, geometry.origin, geometry.parent_size):
        lo = origin + np.arange(n, dtype=np.float64) * size
        hi = lo + size
        closest = np.maximum(lo, 0.0) + np.minimum(hi, 0.0)
        out.append(closest * closest)
    x, y, z = out
    field = (z[:, None, None] + y[None, :, None] + x[None, None, :]) < geometry.sphere_radius ** 2
    field.flags.writeable = False
    return field


def _valid_parent_field(geometry, device):
    key = (geometry.sha256, str(torch.device(device)))
    cached = _VALID_PARENT_GPU.get(key)
    if cached is not None:
        return cached
    # Generate the same float64 AABB distance test as parent_valid_field, on
    # the selected device. X-fastest flatten order matches score ZYX storage.
    xyz = []
    for n, origin, size in zip(geometry.parent_resolution, geometry.origin, geometry.parent_size):
        lo = origin + torch.arange(n, dtype=torch.float64, device=device) * size
        hi = lo + size
        closest = torch.clamp_min(lo, 0.0) + torch.clamp_max(hi, 0.0)
        xyz.append(closest.square())
    x, y, z = xyz
    field = (z[:, None, None] + y[None, :, None] + x[None, None, :]) < geometry.sphere_radius ** 2
    if len(_VALID_PARENT_GPU) >= 8:
        _VALID_PARENT_GPU.pop(next(iter(_VALID_PARENT_GPU)))
    _VALID_PARENT_GPU[key] = field
    return field


def gpu_select_occupancy_candidates(score_zyx, geometry, *, strategy="threshold", threshold=.5,
                                    top_k=None, expansion=0, budget=None, overflow="error"):
    """GPU equivalent of ``select_occupancy_candidates``; IDs stay on device."""
    if not isinstance(score_zyx, torch.Tensor) or score_zyx.device.type != "cuda":
        raise ValueError("score_zyx must be a CUDA tensor")
    if tuple(score_zyx.shape) not in (geometry.parent_resolution[::-1], (1, 1, *geometry.parent_resolution[::-1])):
        raise ValueError("Expected Stage1 score in ZYX layout")
    if score_zyx.ndim == 5:
        if score_zyx.shape[:2] != (1, 1):
            raise ValueError("Only one scene and one score channel are supported")
        score = score_zyx[0, 0]
    else:
        score = score_zyx
    if not torch.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Expected finite occupancy endpoint scores in [0,1]")
    if type(expansion) is not int or expansion < 0:
        raise ValueError("expansion must be a nonnegative integer")
    if overflow not in ("error", "topk_score"):
        raise ValueError("Budget overflow must be explicit")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError("Invalid token budget")

    flat = score.contiguous().view(-1)
    valid = _valid_parent_field(geometry, score.device).reshape(-1)
    if strategy == "threshold":
        if not 0 <= threshold <= 1:
            raise ValueError("Invalid score threshold")
        ids = torch.nonzero((flat > threshold) & valid, as_tuple=False).flatten()
    elif strategy == "topk":
        if type(top_k) is not int or not 0 < top_k <= int(valid.sum().item()):
            raise ValueError("Invalid valid-ROI topK")
        candidates = torch.nonzero(valid, as_tuple=False).flatten()
        # candidates are ascending; stable score sort matches NumPy lexsort's
        # score-descending, then ID-ascending tie break.
        order = torch.argsort(flat[candidates], descending=True, stable=True)[:top_k]
        ids = candidates[order]
    else:
        raise ValueError("Unknown candidate strategy")
    before_expansion = int(ids.numel())
    if expansion:
        mask = torch.zeros(geometry.num_parents, dtype=torch.float32, device=score.device)
        mask[ids] = 1.0
        mask = mask.view(1, 1, *geometry.parent_resolution[::-1])
        for _ in range(expansion):
            mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
        expanded = torch.nonzero(mask.view(-1).bool() & valid, as_tuple=False).flatten()
        ids = expanded
    uncapped = int(ids.numel())
    if budget is not None and len(ids) > budget:
        if overflow == "error":
            raise RuntimeError(f"Candidate budget exceeded:{len(ids)}>{budget}; no truncation performed")
        ids = ids[torch.argsort(flat[ids], descending=True, stable=True)[:budget]]
    ids = torch.sort(ids).values
    scores = flat[ids].float()
    provenance = {"mode": "predicted", "score_definition": "uncalibrated Stage1 endpoint proxy",
                  "before_expansion": before_expansion, "before_budget": uncapped,
                  "selected": int(ids.numel()), "budget_dropped": uncapped-int(ids.numel()),
                  "budget_policy": overflow, "expansion": expansion}
    return ids, scores, provenance


def gpu_observed_children(input_xyz, candidate_ids, geometry):
    """Construct [M,K] input-hit mask with float64 voxel indexing on GPU."""
    if input_xyz.device.type != "cuda" or candidate_ids.device.type != "cuda":
        raise ValueError("input_xyz and candidate_ids must be CUDA tensors")
    if input_xyz.ndim != 2 or input_xyz.shape[1] != 3 or candidate_ids.ndim != 1:
        raise ValueError("Expected XYZ [N,3] and candidate IDs [M]")
    if candidate_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("candidate_ids must be integer IDs")
    if not torch.isfinite(input_xyz).all():
        raise ValueError("Non-finite LiDAR input")
    p = input_xyz.double()
    origin = p.new_tensor(geometry.origin)
    maximum = p.new_tensor(geometry.maximum)
    valid = torch.isfinite(p).all(-1)
    valid &= ((p > origin) & (p < maximum)).all(-1)
    valid &= p.square().sum(-1) < geometry.sphere_radius ** 2
    p = p[valid]
    observed = torch.zeros((len(candidate_ids), geometry.k), dtype=torch.bool, device=input_xyz.device)
    if p.numel() == 0 or candidate_ids.numel() == 0:
        return observed
    size = p.new_tensor(geometry.fine_size)
    fine = torch.floor((p-origin) / size).long()
    split = torch.tensor(geometry.child_split, dtype=torch.long, device=p.device)
    parent = torch.div(fine, split, rounding_mode="floor")
    local = torch.remainder(fine, split)
    child = local[:, 0] + geometry.child_split[0] * (local[:, 1] + geometry.child_split[1] * local[:, 2])
    nx, ny, _ = geometry.parent_resolution
    parent_id = parent[:, 0] + nx * (parent[:, 1] + ny * parent[:, 2])
    candidate_ids = candidate_ids.long()
    pos = torch.searchsorted(candidate_ids, parent_id)
    safe = pos.clamp(max=max(int(candidate_ids.numel())-1, 0))
    match = (pos < candidate_ids.numel()) & (candidate_ids[safe] == parent_id)
    observed[pos[match], child[match]] = True
    return observed


def gpu_occupied_fine_coords(parent_ids, occupancy, valid_child_mask, geometry):
    """Decode sparse [M,K] occupancy directly to Fine XYZ integer coords."""
    if any(x.device.type != "cuda" for x in (parent_ids, occupancy, valid_child_mask)):
        raise ValueError("All inputs must be CUDA tensors")
    if parent_ids.ndim != 1 or occupancy.shape != (len(parent_ids), geometry.k) or valid_child_mask.shape != occupancy.shape:
        raise ValueError("Expected parent IDs [M], occupancy [M,K], and valid mask [M,K]")
    pairs = torch.nonzero(occupancy.bool() & valid_child_mask.bool(), as_tuple=False)
    if pairs.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=parent_ids.device)
    rows, child = pairs[:, 0], pairs[:, 1]
    ids = parent_ids.long()[rows]
    nx, ny, _ = geometry.parent_resolution
    parent_xyz = torch.stack((ids % nx, (ids // nx) % ny, ids // (nx * ny)), dim=-1)
    sx, sy, _ = geometry.child_split
    offset = torch.stack((child % sx, (child // sx) % sy, child // (sx * sy)), dim=-1)
    return parent_xyz * parent_ids.new_tensor(geometry.child_split, dtype=torch.long) + offset


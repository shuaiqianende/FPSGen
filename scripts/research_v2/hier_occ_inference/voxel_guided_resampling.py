"""PatchScene-style uniform resampling inside occupied prediction voxels.

The resampling grid is independent of HS-OccFlow's 768x768x64 fine grid.
Samples are uniform in the intersection of each occupied resampling cell and
the configured box/spherical ROI, using vectorized rejection sampling.
"""
from __future__ import annotations

import math
import time

import numpy as np


def _roi_arrays(bounds, roi_config):
    cfg = dict(bounds or {})
    cfg.update(roi_config or {})
    minimum = cfg.get("minimum", cfg.get("min", cfg.get("origin")))
    maximum = cfg.get("maximum", cfg.get("max"))
    if minimum is None or maximum is None:
        raise ValueError("bounds must provide minimum/origin and maximum XYZ")
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or np.any(minimum >= maximum):
        raise ValueError("bounds must be finite increasing XYZ triples")
    radius = cfg.get("sphere_radius", cfg.get("radius", None))
    if radius is not None and (not np.isfinite(radius) or float(radius) <= 0):
        raise ValueError("sphere_radius must be finite and positive")
    return minimum, maximum, None if radius is None else float(radius)


def voxel_guided_uniform_resample(
    points,
    *,
    bounds,
    voxel_size=0.15625,
    target_points=1_080_000,
    count_policy="exact_target",
    seed=20261019,
    roi_config=None,
):
    """Resample points uniformly within occupied resampling voxels.

    ``bounds`` is a mapping containing ``minimum``/``maximum`` XYZ (aliases
    ``origin``/``min`` are accepted). ``sphere_radius`` optionally adds the
    strict ``||xyz|| < radius`` ROI used by HS-OccFlow.

    ``exact_target`` distributes exactly the requested count as evenly as
    possible over viable occupied cells; if there are more cells than points,
    it selects a reproducible uniform subset. ``patchscene_ceil`` follows the
    paper's ceil rule and emits ``ceil(target/N_cells)*N_cells`` points.
    """
    started = time.perf_counter()
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if not np.issubdtype(p.dtype, np.number):
        raise TypeError("points must be numeric XYZ")
    if not np.isfinite(p).all():
        raise ValueError("points contain NaN or Inf")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    if type(target_points) is not int or target_points < 1:
        raise ValueError("target_points must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if count_policy not in ("exact_target", "patchscene_ceil"):
        raise ValueError("count_policy must be exact_target or patchscene_ceil")
    minimum, maximum, radius = _roi_arrays(bounds, roi_config)
    if len(p) == 0:
        raise ValueError("cannot resample an empty prediction")

    # Use the specified origin and floor convention, while clipping the last
    # partial cell against the exact rectangular bounds.
    in_box = ((p >= minimum) & (p < maximum)).all(axis=1)
    p_box = np.asarray(p[in_box], dtype=np.float64)
    excluded_input = int(len(p) - len(p_box))
    if len(p_box) == 0:
        raise ValueError("prediction has no points inside the configured bounds")
    cell_coord = np.floor((p_box - minimum) / float(voxel_size)).astype(np.int64)
    cell_ids, point_cell = np.unique(cell_coord, axis=0, return_inverse=True)
    raw_occupied_cells = len(cell_ids)

    cell_lo = minimum + cell_ids.astype(np.float64) * float(voxel_size)
    cell_hi = np.minimum(cell_lo + float(voxel_size), maximum)
    if radius is None:
        viable = np.ones(raw_occupied_cells, dtype=bool)
    else:
        # A cell contributes only if its clipped AABB has positive-volume
        # intersection with the open sphere. Closest-point distance is the
        # exact box/sphere intersection test.
        closest = np.maximum(cell_lo, 0.0) + np.minimum(cell_hi, 0.0)
        viable = np.einsum("ij,ij->i", closest, closest) < radius * radius
    eligible_old_ids = np.flatnonzero(viable)
    dropped_no_roi = int(raw_occupied_cells - len(eligible_old_ids))
    if len(eligible_old_ids) == 0:
        raise ValueError("no occupied resampling voxels intersect the configured ROI")

    # Keep a deterministic old-cell -> compact eligible-cell mapping.
    old_to_eligible = np.full(raw_occupied_cells, -1, dtype=np.int64)
    old_to_eligible[eligible_old_ids] = np.arange(len(eligible_old_ids), dtype=np.int64)
    eligible_point = old_to_eligible[point_cell]
    # A cell classified viable can contain only out-of-ROI input points. It
    # still belongs to O and can be sampled from its ROI intersection.
    eligible_ids = cell_ids[eligible_old_ids]
    lo = cell_lo[eligible_old_ids]
    hi = cell_hi[eligible_old_ids]
    n_cells = len(eligible_ids)
    rng = np.random.default_rng(seed)

    if count_policy == "exact_target":
        if target_points >= n_cells:
            base, rem = divmod(target_points, n_cells)
            counts = np.full(n_cells, base, dtype=np.int64)
            if rem:
                counts[rng.choice(n_cells, size=rem, replace=False)] += 1
            selected = np.arange(n_cells, dtype=np.int64)
        else:
            selected = np.sort(rng.choice(n_cells, size=target_points, replace=False))
            counts = np.ones(target_points, dtype=np.int64)
        actual_points = target_points
    else:
        k = int(math.ceil(target_points / n_cells))
        selected = np.arange(n_cells, dtype=np.int64)
        counts = np.full(n_cells, k, dtype=np.int64)
        actual_points = int(k * n_cells)

    # Expand voxel assignments once; rejection is vectorized over all pending
    # samples. Conditional rejection sampling is uniform over cell ∩ ROI.
    source_cell_for_point = np.repeat(selected, counts)
    out = np.empty((actual_points, 3), dtype=np.float64)
    pending = np.arange(actual_points, dtype=np.int64)
    proposals = 0
    rounds = 0
    while len(pending):
        rounds += 1
        if rounds > 10_000:
            raise RuntimeError(
                "ROI rejection sampling did not converge; remaining="
                f"{len(pending)}/{actual_points}, proposals={proposals}"
            )
        assigned = source_cell_for_point[pending]
        candidate = lo[assigned] + rng.random((len(pending), 3)) * (hi[assigned] - lo[assigned])
        proposals += len(pending)
        good = np.isfinite(candidate).all(axis=1)
        if radius is not None:
            good &= np.einsum("ij,ij->i", candidate, candidate) < radius * radius
        accepted = pending[good]
        out[accepted] = candidate[good]
        pending = pending[~good]

    out = out.astype(np.float32)
    # Validate after float32 conversion so the exported artifact itself obeys
    # the range contract. Rare rounding-to-open-boundary values are nudged by
    # resampling those output slots rather than clipping them.
    invalid = ~((out >= minimum.astype(np.float32)) & (out < maximum.astype(np.float32))).all(axis=1)
    if radius is not None:
        invalid |= np.einsum("ij,ij->i", out.astype(np.float64), out.astype(np.float64)) >= radius * radius
    output_cells = np.floor((out.astype(np.float64) - minimum) / float(voxel_size)).astype(np.int64)
    expected_cells = eligible_ids[source_cell_for_point]
    invalid |= (output_cells != expected_cells).any(axis=1)
    retries = 0
    while invalid.any():
        retries += 1
        if retries > 100:
            raise RuntimeError("float32 conversion repeatedly violated the ROI")
        slots = np.flatnonzero(invalid)
        assigned = source_cell_for_point[slots]
        candidate = lo[assigned] + rng.random((len(slots), 3)) * (hi[assigned] - lo[assigned])
        good = np.isfinite(candidate).all(axis=1)
        if radius is not None:
            good &= np.einsum("ij,ij->i", candidate, candidate) < radius * radius
        candidate32 = candidate.astype(np.float32)
        candidate_cells = np.floor((candidate32.astype(np.float64) - minimum) / float(voxel_size)).astype(np.int64)
        good &= (candidate_cells == eligible_ids[assigned]).all(axis=1)
        out[slots[good]] = candidate32[good]
        invalid[slots[good]] = False

    selected_voxels = len(selected)
    dropped_by_budget = n_cells - selected_voxels
    points_per_selected = counts
    metadata = {
        "input_point_count": int(len(p)),
        "input_points_inside_bounds": int(len(p_box)),
        "input_points_excluded_outside_bounds": excluded_input,
        "occupied_resample_voxels": int(n_cells),
        "eligible_occupied_voxels": int(n_cells),
        "raw_occupied_resample_voxels": int(raw_occupied_cells),
        "target_points": int(target_points),
        "actual_output_points": int(actual_points),
        "count_policy": count_policy,
        "voxel_size": float(voxel_size),
        "seed": int(seed),
        "mean_points_per_voxel": float(points_per_selected.mean()),
        "min_points_per_voxel": int(points_per_selected.min()),
        "max_points_per_voxel": int(points_per_selected.max()),
        "dropped_occupied_voxels": int(dropped_no_roi + dropped_by_budget),
        "dropped_no_roi_voxels": dropped_no_roi,
        "dropped_by_point_budget_voxels": int(dropped_by_budget),
        "sampled_voxels": int(selected_voxels),
        "voxel_coverage_ratio": float(selected_voxels / max(raw_occupied_cells, 1)),
        "proposals": int(proposals),
        "roi_rejection_rate": float((proposals - actual_points) / max(proposals, 1)),
        "resampling_seconds": float(time.perf_counter() - started),
    }
    return out, metadata

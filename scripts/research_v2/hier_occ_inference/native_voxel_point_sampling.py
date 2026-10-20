"""Point outputs sampled directly from a model's native occupied voxels."""
from __future__ import annotations

import time

import numpy as np


def _hit_and_run_box_ball(lo, hi, radius, rng, *, burn_in=512):
    """Rare-cell fallback for near-tangent voxel/sphere intersections."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    closest = np.maximum(lo, 0.0) + np.minimum(hi, 0.0)
    center = (lo + hi) * 0.5
    alpha = 0.5
    for _ in range(80):
        point = closest + alpha * (center - closest)
        if ((point > lo).all() and (point < hi).all()
                and np.dot(point, point) < radius * radius):
            break
        alpha *= 0.5
    else:
        raise RuntimeError("could not initialize sampler inside voxel/sphere intersection")

    # Hit-and-run has the uniform distribution on a convex body as its
    # stationary distribution. It is invoked only for pathological slivers
    # after direct rejection has failed many times.
    for _ in range(burn_in + 1):
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction /= norm
        low_t, high_t = -np.inf, np.inf
        for axis in range(3):
            if abs(direction[axis]) < 1e-14:
                if not lo[axis] < point[axis] < hi[axis]:
                    low_t, high_t = 1.0, 0.0
                    break
                continue
            t0 = (lo[axis] - point[axis]) / direction[axis]
            t1 = (hi[axis] - point[axis]) / direction[axis]
            low_t = max(low_t, min(t0, t1))
            high_t = min(high_t, max(t0, t1))
        dot = float(np.dot(point, direction))
        disc = dot * dot + radius * radius - float(np.dot(point, point))
        if disc <= 0:
            raise RuntimeError("hit-and-run sampler reached sphere boundary")
        root = np.sqrt(disc)
        low_t = max(low_t, -dot - root)
        high_t = min(high_t, -dot + root)
        if not low_t < high_t:
            raise RuntimeError("empty chord inside valid voxel/sphere intersection")
        point = point + rng.uniform(low_t, high_t) * direction
    return point


def sample_native_occupied_voxels(
    occupied_coords,
    geometry,
    *,
    point_mode="voxel_center",
    target_points=1_080_000,
    seed=20261019,
    roi=None,
):
    """Export centers or uniform points inside native occupied voxels.

    ``occupied_coords`` must be integer ``[N,3]`` XYZ coordinates from the
    model's native occupancy grid.  The function does not re-voxelize points.
    For ``native_fine_jitter``, sampling is uniform in each source voxel's
    intersection with the configured rectangular bounds and open 3D sphere.
    """
    started = time.perf_counter()
    coords = np.asarray(occupied_coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("occupied_coords must have shape [N,3] in XYZ order")
    if not np.issubdtype(coords.dtype, np.integer):
        raise TypeError("occupied_coords must be integer native-grid coordinates")
    coords = coords.astype(np.int64, copy=False)
    if len(coords) == 0:
        raise ValueError("cannot sample an empty occupied-voxel set")
    if not np.isfinite(coords).all():
        raise ValueError("occupied_coords contain non-finite values")
    max_grid = np.asarray(geometry.fine_resolution, dtype=np.int64)
    if ((coords < 0) | (coords >= max_grid)).any():
        raise ValueError("occupied_coords are outside the native fine grid")
    if point_mode not in ("voxel_center", "native_fine_jitter"):
        raise ValueError("point_mode must be voxel_center or native_fine_jitter")

    origin = np.asarray(geometry.origin, dtype=np.float64)
    maximum = np.asarray(geometry.maximum, dtype=np.float64)
    voxel_size = np.asarray(geometry.fine_size, dtype=np.float64)
    if voxel_size.shape != (3,) or not np.isfinite(voxel_size).all() or (voxel_size <= 0).any():
        raise ValueError("geometry.fine_size must be a finite positive XYZ triple")
    roi = dict(roi or {})
    sphere_radius = float(roi.get("sphere_radius", geometry.sphere_radius))
    if not np.isfinite(sphere_radius) or sphere_radius <= 0:
        raise ValueError("sphere_radius must be finite and positive")

    centers = np.asarray(geometry.centers(coords, fine=True), dtype=np.float32)
    if point_mode == "voxel_center":
        meta = {
            "point_mode": point_mode,
            "native_voxel_size_xyz": voxel_size.tolist(),
            "input_occupied_voxels": int(len(coords)),
            "original_occupied_voxels": int(len(coords)),
            "sampled_occupied_voxels": int(len(coords)),
            "dropped_occupied_voxels": 0,
            "voxel_coverage_ratio": 1.0,
            "actual_output_points": int(len(centers)),
            "target_points": None,
            "seed": None,
            "sampling_seconds": float(time.perf_counter() - started),
        }
        return centers, meta

    if type(target_points) is not int or target_points < 1:
        raise ValueError("target_points must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    lo = origin + coords.astype(np.float64) * voxel_size
    hi = np.minimum(lo + voxel_size, maximum)
    # Keep only voxels with positive-volume intersection with the ROI. Validity
    # uses closest-point-to-origin geometry, matching SparseHierarchyGeometry.
    clipped_lo = np.maximum(lo, origin)
    clipped_hi = np.minimum(hi, maximum)
    closest = np.maximum(clipped_lo, 0.0) + np.minimum(clipped_hi, 0.0)
    valid = (clipped_hi > clipped_lo).all(axis=1)
    valid &= np.einsum("ij,ij->i", closest, closest) < sphere_radius * sphere_radius
    geometric_valid = valid.copy()
    # Output is float32. Exclude only cells whose ROI intersection contains no
    # representable float32 XYZ at all; otherwise a mathematically valid
    # sub-ULP sliver could never satisfy the exported point contract.
    closest32 = closest.astype(np.float32).astype(np.float64)
    below = closest32 < clipped_lo
    above = closest32 > clipped_hi
    lower32 = clipped_lo.astype(np.float32)
    upper32 = clipped_hi.astype(np.float32)
    closest32 = np.where(below, np.nextafter(lower32, np.float32(np.inf)).astype(np.float64), closest32)
    closest32 = np.where(above, np.nextafter(upper32, np.float32(-np.inf)).astype(np.float64), closest32)
    closest32 = np.where(closest32 <= origin, np.nextafter(origin.astype(np.float32), np.float32(np.inf)).astype(np.float64), closest32)
    closest32 = np.where(closest32 >= maximum, np.nextafter(maximum.astype(np.float32), np.float32(-np.inf)).astype(np.float64), closest32)
    for _ in range(8):
        voxel = np.floor((closest32 - origin) / voxel_size).astype(np.int64)
        too_low = voxel < coords
        too_high = voxel > coords
        if not (too_low | too_high).any():
            break
        rounded = closest32.astype(np.float32)
        rounded = np.where(too_low, np.nextafter(rounded, np.float32(np.inf)), rounded)
        rounded = np.where(too_high, np.nextafter(rounded, np.float32(-np.inf)), rounded)
        closest32 = rounded.astype(np.float64)
    representable = ((closest32 >= clipped_lo) & (closest32 <= clipped_hi)).all(axis=1)
    representable &= np.einsum("ij,ij->i", closest32, closest32) < sphere_radius * sphere_radius
    representable &= ((closest32 > origin) & (closest32 < maximum)).all(axis=1)
    closest32_voxel = np.floor((closest32 - origin) / voxel_size).astype(np.int64)
    representable &= (closest32_voxel == coords).all(axis=1)
    valid &= representable
    eligible_coords = coords[valid]
    lo, hi = clipped_lo[valid], clipped_hi[valid]
    n_eligible = len(eligible_coords)
    dropped_invalid = int(len(coords) - n_eligible)
    if n_eligible == 0:
        raise ValueError("no occupied native voxels intersect the configured ROI")
    eligible_representable = closest32[valid]

    rng = np.random.default_rng(seed)
    if target_points >= n_eligible:
        base, remainder = divmod(target_points, n_eligible)
        counts = np.full(n_eligible, base, dtype=np.int64)
        if remainder:
            counts[rng.choice(n_eligible, size=remainder, replace=False)] += 1
        selected = np.arange(n_eligible, dtype=np.int64)
    else:
        selected = np.sort(rng.choice(n_eligible, size=target_points, replace=False))
        counts = np.ones(target_points, dtype=np.int64)
    assigned = np.repeat(selected, counts)
    if len(assigned) != target_points:
        raise AssertionError("native voxel count allocation did not reach exact target")

    out64 = np.empty((target_points, 3), dtype=np.float64)
    pending = np.arange(target_points, dtype=np.int64)
    proposals = 0
    rounds = 0
    fallback_samples = 0
    while len(pending):
        rounds += 1
        if rounds > 256:
            for output_index in pending:
                cell = assigned[output_index]
                candidate = None
                for _ in range(32):
                    trial = _hit_and_run_box_ball(lo[cell], hi[cell], sphere_radius, rng)
                    trial32 = trial.astype(np.float32).astype(np.float64)
                    voxel = np.floor((trial32 - origin) / voxel_size).astype(np.int64)
                    if (np.linalg.norm(trial32) < sphere_radius
                            and (trial32 > origin).all() and (trial32 < maximum).all()
                            and np.array_equal(voxel, eligible_coords[cell])):
                        candidate = trial
                        break
                # A representable point was proved to exist above. Extremely
                # thin intersections can nevertheless round every finite
                # Markov proposal onto a boundary; use their nearest interior
                # float32 witness as the deterministic last resort.
                out64[output_index] = eligible_representable[cell] if candidate is None else candidate
                fallback_samples += 1
            pending = np.empty(0, dtype=np.int64)
            break
        cells = assigned[pending]
        proposal = lo[cells] + rng.random((len(pending), 3)) * (hi[cells] - lo[cells])
        proposals += len(pending)
        proposal32 = proposal.astype(np.float32).astype(np.float64)
        good = np.isfinite(proposal).all(axis=1)
        good &= (proposal > origin).all(axis=1) & (proposal < maximum).all(axis=1)
        good &= np.einsum("ij,ij->i", proposal, proposal) < sphere_radius * sphere_radius
        good &= (proposal32 > origin).all(axis=1) & (proposal32 < maximum).all(axis=1)
        good &= np.einsum("ij,ij->i", proposal32, proposal32) < sphere_radius * sphere_radius
        proposal_voxels = np.floor((proposal32 - origin) / voxel_size).astype(np.int64)
        good &= (proposal_voxels == eligible_coords[cells]).all(axis=1)
        accepted = pending[good]
        out64[accepted] = proposal[good]
        pending = pending[~good]

    out = out64.astype(np.float32)
    # Validate exported float32 points against both the ROI and their source
    # fine voxel. Values are never clipped, which would create boundary mass.
    lo_grid = np.floor((out.astype(np.float64) - origin) / voxel_size).astype(np.int64)
    source_coords = eligible_coords[assigned]
    valid_output = ((out.astype(np.float64) > origin) & (out.astype(np.float64) < maximum)).all(axis=1)
    valid_output &= np.einsum("ij,ij->i", out.astype(np.float64), out.astype(np.float64)) < sphere_radius * sphere_radius
    valid_output &= (lo_grid == source_coords).all(axis=1)
    if not np.isfinite(out).all() or not valid_output.all():
        raise RuntimeError("float32 native voxel samples violated cell or ROI bounds")

    sampled = int(len(selected))
    meta = {
        "point_mode": point_mode,
        "native_voxel_size_xyz": voxel_size.tolist(),
        "input_occupied_voxels": int(len(coords)),
        "original_occupied_voxels": int(len(coords)),
        "eligible_occupied_voxels": int(n_eligible),
        "dropped_unrepresentable_roi_voxels": int((geometric_valid & ~representable).sum()),
        "sampled_occupied_voxels": sampled,
        "dropped_occupied_voxels": int(dropped_invalid + n_eligible - sampled),
        "dropped_invalid_roi_voxels": dropped_invalid,
        "dropped_by_point_budget_voxels": int(n_eligible - sampled),
        "voxel_coverage_ratio": float(sampled / len(coords)),
        "actual_output_points": int(len(out)),
        "target_points": int(target_points),
        "count_policy": "exact_target",
        "seed": int(seed),
        "mean_points_per_sampled_voxel": float(counts.mean()),
        "min_points_per_sampled_voxel": int(counts.min()),
        "max_points_per_sampled_voxel": int(counts.max()),
        "proposals": int(proposals),
        "roi_rejection_rate": float((proposals - target_points) / max(proposals, 1)),
        "roi_hit_and_run_fallback_samples": int(fallback_samples),
        "sampling_seconds": float(time.perf_counter() - started),
    }
    return out, meta

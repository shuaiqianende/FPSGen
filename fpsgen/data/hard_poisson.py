"""Deterministic, index-preserving hard-Poisson point selection.

The selector never moves, interpolates, or synthesizes coordinates.  Its
return value is an index array into the caller's original point array, which
keeps per-point attributes such as SemanticKITTI labels aligned by design.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PoissonSelection:
    """Indices and diagnostics from one hard-Poisson pass."""

    indices: np.ndarray
    radius: float
    scanned_points: int
    truncated_at_max_accept: bool


def _validate_xyz(xyz: np.ndarray, target_points: int) -> np.ndarray:
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}")
    if not np.issubdtype(xyz.dtype, np.floating):
        raise TypeError("xyz must be floating point")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or Inf")
    if not 0 < target_points <= len(xyz):
        raise ValueError(f"target_points must be in [1, {len(xyz)}], got {target_points}")
    return np.ascontiguousarray(xyz, dtype=np.float32)


def hard_poisson_select(
    xyz: np.ndarray,
    *,
    target_points: int,
    initial_radius: float,
    seed: int,
    max_accept: int | None = None,
) -> PoissonSelection:
    """Select original rows with a spatial-hash hard minimum-distance rule.

    Candidate traversal is a deterministic permutation generated from
    ``seed``.  A point is accepted only when it is at least ``initial_radius``
    from all accepted points in its voxel and its 26 adjacent voxels.  A
    finite ``max_accept`` permits radius search to stop once it has proved a
    radius retains enough points; the returned prefix remains a valid
    hard-Poisson subset.
    """
    xyz = _validate_xyz(xyz, target_points)
    radius = float(initial_radius)
    if radius < 0:
        raise ValueError("initial_radius must be non-negative")
    if max_accept is not None and max_accept < target_points:
        raise ValueError("max_accept must be >= target_points")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xyz))
    limit = len(xyz) if max_accept is None else min(int(max_accept), len(xyz))
    if radius == 0.0:
        selected = order[:limit].astype(np.int64, copy=False)
        return PoissonSelection(selected, radius, len(selected), len(selected) == limit < len(xyz))

    radius_sq = radius * radius
    cells = np.floor(xyz / radius).astype(np.int64)
    # A tuple key avoids assumptions about coordinate range and hash packing.
    accepted_by_cell: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    accepted: list[int] = []
    offsets: Iterable[tuple[int, int, int]] = product((-1, 0, 1), repeat=3)
    offsets = tuple(offsets)
    scanned = 0
    for index in order:
        scanned += 1
        cell = cells[index]
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        point = xyz[index]
        allowed = True
        for dx, dy, dz in offsets:
            neighbours = accepted_by_cell.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if not neighbours:
                continue
            delta = xyz[np.asarray(neighbours, dtype=np.int64)] - point
            if np.any(np.einsum("ij,ij->i", delta, delta) < radius_sq):
                allowed = False
                break
        if not allowed:
            continue
        accepted.append(int(index))
        accepted_by_cell[key].append(int(index))
        if len(accepted) >= limit:
            return PoissonSelection(np.asarray(accepted, dtype=np.int64), radius, scanned, limit < len(xyz))
    return PoissonSelection(np.asarray(accepted, dtype=np.int64), radius, scanned, False)


def select_largest_feasible_radius(
    xyz: np.ndarray,
    *,
    target_points: int,
    seed: int,
    initial_high: float = 0.05,
    tolerance: int = 2000,
    refinement_passes: int = 8,
) -> tuple[np.ndarray, dict]:
    """Find a large feasible radius and return exactly ``target_points`` rows.

    The upper bracket is expanded automatically.  Each valid candidate is
    capped at ``target + tolerance`` accepts, which is sufficient to prove it
    can meet cardinality and prevents unnecessary CPU work.  Removing the
    small deterministic excess from a hard-Poisson set cannot reduce its
    minimum spacing.
    """
    xyz = _validate_xyz(xyz, target_points)
    if initial_high <= 0 or tolerance < 0 or refinement_passes < 0:
        raise ValueError("invalid radius-search parameters")
    cap = min(len(xyz), target_points + tolerance)

    def run(radius: float) -> PoissonSelection:
        return hard_poisson_select(xyz, target_points=target_points, initial_radius=radius,
                                   seed=seed, max_accept=cap)

    low_radius = 0.0
    # Radius zero is always feasible because target <= N.
    best = run(low_radius)
    high_radius = float(initial_high)
    high = run(high_radius)
    bracket_expansions = 0
    while len(high.indices) >= target_points:
        low_radius, best = high_radius, high
        high_radius *= 2.0
        bracket_expansions += 1
        if bracket_expansions > 32:
            raise RuntimeError("could not find an infeasible hard-Poisson radius")
        high = run(high_radius)

    for _ in range(refinement_passes):
        middle = 0.5 * (low_radius + high_radius)
        candidate = run(middle)
        if len(candidate.indices) >= target_points:
            low_radius, best = middle, candidate
        else:
            high_radius = middle

    retained = len(best.indices)
    if retained < target_points:
        raise RuntimeError("radius search ended without a feasible point set")
    if retained > target_points:
        rng = np.random.default_rng(seed + 1)
        keep = np.sort(rng.choice(retained, size=target_points, replace=False))
        final_indices = best.indices[keep]
    else:
        final_indices = best.indices
    diagnostics = {
        "final_radius": float(low_radius),
        "radius_upper_bracket": float(high_radius),
        "poisson_retained": int(retained),
        "final_points": int(len(final_indices)),
        "target_tolerance": int(tolerance),
        "bracket_expansions": int(bracket_expansions),
        "refinement_passes": int(refinement_passes),
        "scanned_points": int(best.scanned_points),
        "truncated_at_max_accept": bool(best.truncated_at_max_accept),
    }
    return final_indices.astype(np.int64, copy=False), diagnostics

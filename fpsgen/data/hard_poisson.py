"""Deterministic, index-preserving hard-Poisson point selection.

The selector never moves, interpolates, or synthesizes coordinates.  Its
return value is an index array into the caller's original point array, which
keeps per-point attributes such as SemanticKITTI labels aligned by design.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np


@dataclass(frozen=True)
class PoissonSelection:
    """Indices and diagnostics from one hard-Poisson pass."""

    indices: np.ndarray
    radius: float
    scanned_points: int
    truncated_at_max_accept: bool


_NATIVE_LIBRARY = None


def _native_selector():
    """Build/load the versioned local C++ spatial-hash inner loop once."""
    global _NATIVE_LIBRARY
    if _NATIVE_LIBRARY is not None:
        return _NATIVE_LIBRARY
    source = Path(__file__).with_name("hard_poisson_native.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library_path = Path(tempfile.gettempdir()) / f"fpsgen_hard_poisson_{digest}.so"
    if not library_path.exists():
        temporary = library_path.with_suffix(".tmp.so")
        subprocess.run([
            "g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(temporary),
        ], check=True, capture_output=True, text=True)
        temporary.replace(library_path)
    library = ctypes.CDLL(str(library_path))
    function = library.hard_poisson_select_native
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_double, ctypes.c_uint64,
        ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    function.restype = ctypes.c_int
    _NATIVE_LIBRARY = function
    return function


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

    limit = len(xyz) if max_accept is None else min(int(max_accept), len(xyz))
    if radius == 0.0:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(xyz))
        selected = order[:limit].astype(np.int64, copy=False)
        return PoissonSelection(selected, radius, len(selected), len(selected) == limit < len(xyz))
    output = np.empty(limit, dtype=np.int64)
    selected = ctypes.c_int64()
    scanned = ctypes.c_int64()
    result = _native_selector()(xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(xyz), radius,
                                int(seed), limit,
                                output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                ctypes.byref(selected), ctypes.byref(scanned))
    if result != 0:
        raise RuntimeError(f"native hard-Poisson selector failed with code {result}")
    output = output[:selected.value]
    return PoissonSelection(output, radius, int(scanned.value), len(output) == limit < len(xyz))


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
        high_radius *= 1.5
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

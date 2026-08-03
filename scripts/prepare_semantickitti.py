#!/usr/bin/env python3
"""Create FPSGen fixed-cardinality ``input`` and ``gt`` NumPy arrays.

For each frame, this script filters the local LiDAR scan, crops the aggregated
world map with the frame pose, applies the original 10 m viewpoint support
filter, voxel-downsamples both clouds, and writes arrays with shape ``[N, 4]``:
``[x, y, z, semantic_label]``.  The first three columns are metric XYZ in the
current sensor frame and the fourth column is a raw SemanticKITTI label (zero
when labels are unavailable).

The default directory names ``input_`` and ``gt_`` are consumed directly by
FPSGen training.  Pass ``--input-dir input --gt-dir gt`` for the generation
evaluation layout; the evaluation code accepts either spelling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_calibration(path: Path) -> dict[str, np.ndarray]:
    """Read KITTI calibration matrices from ``calib.txt``."""
    matrices = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            numbers = np.fromstring(values, sep=" ")
            if numbers.size == 12:
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, :4] = numbers.reshape(3, 4)
                matrices[key] = matrix
    return matrices


def load_poses(sequence_dir: Path) -> list[np.ndarray]:
    """Load poses in the LiDAR coordinate frame, matching LiDiff's convention."""
    calibration = parse_calibration(sequence_dir / "calib.txt")
    transform = calibration.get("Tr", np.eye(4, dtype=np.float64))
    inverse_transform = np.linalg.inv(transform)
    poses = []
    with (sequence_dir / "poses.txt").open(encoding="utf-8") as handle:
        for line in handle:
            values = np.fromstring(line, sep=" ")
            if values.size != 12:
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(inverse_transform @ pose @ transform)
    return poses


def read_scan(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """Read a SemanticKITTI scan and its optional point-aligned labels."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Expected x/y/z/intensity records in {path}")
    xyz = raw.reshape(-1, 4)[:, :3]
    label_path = path.parents[1] / "labels" / f"{path.stem}.label"
    if label_path.exists():
        labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        if len(labels) != len(xyz):
            raise ValueError(f"Label count mismatch: {path} has {len(xyz)} points, {label_path} has {len(labels)}")
        return xyz, labels.astype(np.float32), True
    return xyz, np.zeros(len(xyz), dtype=np.float32), False


def load_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read global ``map_clean.npy`` and return XYZ plus semantic labels."""
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Expected [N,3+] map array in {path}, got {array.shape}")
    labels = array[:, 3] if array.shape[1] >= 4 else np.zeros(len(array), dtype=np.float32)
    finite = np.isfinite(array[:, :3]).all(axis=1) & np.isfinite(labels)
    return array[finite, :3], labels[finite]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one deterministic point per XYZ voxel while preserving labels."""
    if len(points) == 0:
        return points
    keys = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def downsample_to_target(
    points: np.ndarray,
    target: int,
    initial_voxel: float,
    min_voxel: float,
    max_voxel: float,
    tolerance: int,
    max_iters: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> tuple[np.ndarray, float, bool]:
    """Find a voxel size near the requested count, then sample exactly ``target``."""
    if target <= 0:
        raise ValueError("target point count must be positive")
    if min_voxel <= 0 or max_voxel < min_voxel:
        raise ValueError("invalid voxel-size search range")
    initial_voxel = float(np.clip(initial_voxel, min_voxel, max_voxel))
    best_points, best_voxel, best_error = points, min_voxel, float("inf")
    lo, hi = min_voxel, max_voxel
    current = initial_voxel
    for _ in range(max_iters):
        candidate = voxel_downsample(points, current)
        error = abs(len(candidate) - target)
        if len(candidate) >= target and error < best_error:
            best_points, best_voxel, best_error = candidate, current, error
        if target <= len(candidate) < target + max(0, tolerance):
            best_points, best_voxel = candidate, current
            break
        if len(candidate) >= target:
            lo = current
        else:
            hi = current
        next_value = 0.5 * (lo + hi)
        if abs(next_value - current) < 1e-6:
            break
        current = next_value

    if len(best_points) < target:
        if not allow_replacement:
            raise ValueError(
                f"Only {len(best_points)} points remain, fewer than target {target}; "
                "reduce the target or omit --strict-cardinality"
            )
        indices = rng.choice(len(best_points), size=target, replace=True)
    else:
        indices = rng.choice(len(best_points), size=target, replace=False)
    return best_points[indices], float(best_voxel), len(best_points) < target


def viewpoint_mask(scan_xyz: np.ndarray, map_local_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep map points in the coarse viewpoint cells occupied by the scan."""
    scan_keys = np.floor(scan_xyz / voxel_size).astype(np.int64)
    map_keys = np.floor(map_local_xyz / voxel_size).astype(np.int64)
    dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    scan_struct = np.unique(scan_keys, axis=0).view(dtype).reshape(-1)
    map_struct = map_keys.view(dtype).reshape(-1)
    return np.isin(map_struct, scan_struct)


def process_frame(
    scan_path: Path,
    pose: np.ndarray,
    map_xyz: np.ndarray,
    map_labels: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build one fixed-cardinality local input/full pair."""
    scan_xyz, scan_labels, labels_available = read_scan(scan_path)
    if len(scan_xyz) == 0:
        raise ValueError(f"Empty scan: {scan_path}")

    # Match LiDiff: remove moving classes for labelled training sequences, but
    # retain all points for test sequences or datasets without labels.
    keep = np.ones(len(scan_xyz), dtype=bool)
    if args.split != "test" and labels_available:
        keep = (scan_labels > 1) & (scan_labels < 252)
    partial_xyz = scan_xyz[keep]
    partial_labels = scan_labels[keep]
    distance = np.linalg.norm(partial_xyz, axis=1)
    keep_range = ((distance > args.min_distance) & (distance < args.max_range) &
                  (partial_xyz[:, 2] > args.min_z))
    partial_xyz = partial_xyz[keep_range]
    partial_labels = partial_labels[keep_range]
    if len(partial_xyz) == 0:
        raise ValueError(f"No input points remain after filtering: {scan_path}")

    translation = pose[:3, 3]
    map_distance = np.linalg.norm(map_xyz - translation[None, :], axis=1)
    nearby = map_xyz[map_distance < args.max_range]
    nearby_labels = map_labels[map_distance < args.max_range]
    if len(nearby) == 0:
        raise ValueError(f"No map points within {args.max_range} m of {scan_path}")
    homogeneous = np.concatenate(
        [nearby, np.ones((len(nearby), 1), dtype=np.float32)], axis=1
    )
    local_xyz = (homogeneous @ np.linalg.inv(pose).T)[:, :3]
    height = local_xyz[:, 2] > args.min_z
    if args.max_z is not None:
        height &= local_xyz[:, 2] < args.max_z
    local_xyz = local_xyz[height]
    nearby_labels = nearby_labels[height]
    visible = viewpoint_mask(scan_xyz, local_xyz, args.viewpoint_voxel_size)
    full_xyz = local_xyz[visible]
    full_labels = nearby_labels[visible]
    if len(full_xyz) == 0:
        raise ValueError(f"Viewpoint filtering removed all map points for {scan_path}")

    partial = np.column_stack((partial_xyz, partial_labels)).astype(np.float32)
    full = np.column_stack((full_xyz, full_labels)).astype(np.float32)
    partial, partial_voxel, partial_repeated = downsample_to_target(
        partial, args.input_points, args.input_voxel_init, args.input_voxel_min,
        args.input_voxel_max, args.input_tolerance, args.max_iters, rng,
        not args.strict_cardinality,
    )
    full, full_voxel, full_repeated = downsample_to_target(
        full, args.full_points, args.full_voxel_init, args.full_voxel_min,
        args.full_voxel_max, args.full_tolerance, args.max_iters, rng,
        not args.strict_cardinality,
    )
    stats = {
        "frame": int(scan_path.stem),
        "raw_scan_points": int(len(scan_xyz)),
        "input_points": int(len(partial)),
        "gt_points": int(len(full)),
        "input_voxel_size": partial_voxel,
        "gt_voxel_size": full_voxel,
        "input_repeated": partial_repeated,
        "gt_repeated": full_repeated,
        "labels_available": labels_available,
    }
    return partial, full, stats


def resolve_sequences(root: Path, specification: str) -> list[str]:
    if specification.lower() == "all":
        return sorted(path.name for path in root.iterdir()
                      if path.is_dir() and path.name.isdigit())
    return [item.strip() for item in specification.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SemanticKITTI scans and map crops for FPSGen."
    )
    parser.add_argument("--data-root", required=True, type=Path,
                        help="dataset root containing sequence folders")
    parser.add_argument("--sequences", default="00",
                        help="comma-separated IDs or 'all' (default: 00)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--input-dir", default="input_",
                        help="partial-array directory name (default: input_)")
    parser.add_argument("--gt-dir", default="gt_",
                        help="full-array directory name (default: gt_)")
    parser.add_argument("--map-name", default="map_clean.npy")
    parser.add_argument("--max-range", type=float, default=50.0)
    parser.add_argument("--min-distance", type=float, default=3.5)
    parser.add_argument("--min-z", type=float, default=-4.0)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--viewpoint-voxel-size", type=float, default=10.0)
    parser.add_argument("--input-points", type=int, default=18000)
    parser.add_argument("--full-points", type=int, default=180000)
    parser.add_argument("--input-voxel-init", type=float, default=0.2)
    parser.add_argument("--input-voxel-min", type=float, default=0.1)
    parser.add_argument("--input-voxel-max", type=float, default=0.5)
    parser.add_argument("--input-tolerance", type=int, default=100)
    parser.add_argument("--full-voxel-init", type=float, default=0.2)
    parser.add_argument("--full-voxel-min", type=float, default=0.1)
    parser.add_argument("--full-voxel-max", type=float, default=0.5)
    parser.add_argument("--full-tolerance", type=int, default=1000)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--strict-cardinality", action="store_true",
                        help="fail instead of sampling with replacement when a target is too sparse")
    args = parser.parse_args()
    if args.max_range <= args.min_distance:
        raise ValueError("max-range must be greater than min-distance")

    root = args.data_root.expanduser().resolve()
    summaries = []
    for sequence in resolve_sequences(root, args.sequences):
        sequence_dir = root / sequence
        scan_dir = sequence_dir / "velodyne"
        map_path = sequence_dir / args.map_name
        if not scan_dir.is_dir() or not map_path.is_file():
            print(f"Skipping {sequence}: need {scan_dir} and {map_path}")
            continue
        poses = load_poses(sequence_dir)
        map_xyz, map_labels = load_map(map_path)
        scans = sorted(scan_dir.glob("*.bin"), key=lambda path: int(path.stem))
        scans = scans[args.start:args.stop:args.stride]
        input_dir, gt_dir = sequence_dir / args.input_dir, sequence_dir / args.gt_dir
        input_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        frame_stats = []
        for scan_path in scans:
            frame_id = int(scan_path.stem)
            if frame_id >= len(poses):
                print(f"Skipping {scan_path}: no matching pose")
                continue
            rng = np.random.default_rng(args.seed + frame_id)
            partial, full, stats = process_frame(
                scan_path, poses[frame_id], map_xyz, map_labels, args, rng
            )
            np.save(input_dir / f"{frame_id:06d}.npy", partial)
            np.save(gt_dir / f"{frame_id:06d}.npy", full)
            frame_stats.append(stats)
            print(f"{sequence}/{frame_id:06d}: input={len(partial)} gt={len(full)}")
        if not frame_stats:
            raise RuntimeError(f"No frames prepared for sequence {sequence}")
        summary = {
            "sequence": sequence,
            "split": args.split,
            "map_path": str(map_path),
            "input_dir": str(input_dir),
            "gt_dir": str(gt_dir),
            "frame_count": len(frame_stats),
            "input_points": args.input_points,
            "full_points": args.full_points,
            "seed": args.seed,
            "frames": frame_stats,
        }
        with (sequence_dir / "fpsgen_preprocess.meta.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        summaries.append(summary)
    if not summaries:
        raise RuntimeError("No sequence was processed")


if __name__ == "__main__":
    main()

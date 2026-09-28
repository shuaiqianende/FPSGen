#!/usr/bin/env python3
"""Generate index-preserving hard-Poisson SemanticKITTI GT arrays.

This tool intentionally never reads ``gt_`` as a candidate source and never
rewrites existing FPSGen data.  It reconstructs the high-density local map
candidate with the same geometry steps as ``prepare_semantickitti.py`` and
writes only ``<sequence>/gt_possion/<frame>.npy``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from fpsgen.data.hard_poisson import select_largest_feasible_radius
from prepare_semantickitti import load_map, load_poses, read_scan, viewpoint_mask


DEFAULT_SEQUENCES = "00,01,02,03,04,05,06,07,08,09,10"


def _atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exact_duplicate_count(xyz: np.ndarray) -> int:
    return int(len(xyz) - len(np.unique(xyz, axis=0)))


def _validate_array(array: np.ndarray, target_points: int) -> dict:
    if array.shape != (target_points, 4):
        raise ValueError(f"invalid shape {array.shape}; expected {(target_points, 4)}")
    if array.dtype != np.float32:
        raise ValueError(f"invalid dtype {array.dtype}; expected float32")
    if not np.isfinite(array).all():
        raise ValueError("output contains NaN or Inf")
    duplicates = _exact_duplicate_count(array[:, :3])
    if duplicates:
        raise ValueError(f"output has {duplicates} exact duplicate XYZ rows")
    return {"exact_duplicate_count": duplicates}


def _knn_quality(xyz: np.ndarray, radius: float, device: str) -> dict:
    """Run exact self-KNN QA with the existing PyKeOps sparse-neighbour path."""
    if device == "none":
        return {"knn_checked": False}
    import torch
    from fpsgen.ops.endpoint_refinement.knn import knn

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    torch_device = torch.device("cuda" if device == "cuda" else "cpu")
    if torch_device.type != "cuda":
        raise ValueError("KNN QA is intentionally GPU-only; choose --device cuda or none")
    points = torch.from_numpy(np.ascontiguousarray(xyz)).to(torch_device)
    d2, _ = knn(points, points, k=2, backend="keops")
    nn = d2[:, 1].sqrt().cpu().numpy()
    del d2, points
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    quality = {
        "knn_checked": True,
        "nn_min": float(nn.min()),
        "nn_mean": float(nn.mean()),
        "nn_p01": float(np.quantile(nn, .01)),
        "nn_p05": float(np.quantile(nn, .05)),
        "nn_p50": float(np.quantile(nn, .50)),
        "nn_p95": float(np.quantile(nn, .95)),
        "near_duplicate_lt_001": float(np.mean(nn < .01)),
        "near_duplicate_lt_002": float(np.mean(nn < .02)),
        "near_duplicate_lt_005": float(np.mean(nn < .05)),
    }
    # Removing excess Poisson points cannot lower the radius; this tolerance
    # only accounts for float32 cell-boundary arithmetic.
    if radius > 0 and quality["nn_min"] + max(1e-5, radius * 1e-4) < radius:
        raise ValueError(f"KNN minimum {quality['nn_min']:.8f} violates radius {radius:.8f}")
    return quality


def _full_candidate(scan_path: Path, pose: np.ndarray, map_xyz: np.ndarray,
                    map_labels: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Reuse FPSGen's pre-voxel GT geometry construction exactly."""
    scan_xyz, _, _ = read_scan(scan_path)
    translation = pose[:3, 3]
    map_distance = np.linalg.norm(map_xyz - translation[None, :], axis=1)
    nearby_mask = map_distance < args.max_range
    nearby = map_xyz[nearby_mask]
    labels = map_labels[nearby_mask]
    if len(nearby) == 0:
        raise ValueError("no map points inside max-range crop")
    homogeneous = np.concatenate((nearby, np.ones((len(nearby), 1), dtype=np.float32)), axis=1)
    local_xyz = (homogeneous @ np.linalg.inv(pose).T)[:, :3]
    height = local_xyz[:, 2] > args.min_z
    if args.max_z is not None:
        height &= local_xyz[:, 2] < args.max_z
    local_xyz, labels = local_xyz[height], labels[height]
    visible = viewpoint_mask(scan_xyz, local_xyz, args.viewpoint_voxel_size)
    candidate = np.column_stack((local_xyz[visible], labels[visible])).astype(np.float32, copy=False)
    finite = np.isfinite(candidate).all(axis=1)
    return np.ascontiguousarray(candidate[finite], dtype=np.float32)


def _parse_sequences(value: str) -> list[str]:
    sequences = [item.strip() for item in value.split(",") if item.strip()]
    if not sequences or any(not item.isdigit() for item in sequences):
        raise ValueError("--sequences must be a non-empty comma-separated numeric list")
    return sequences


def _frame_paths(sequence_dir: Path, args: argparse.Namespace) -> list[Path]:
    paths = sorted((sequence_dir / "velodyne").glob("*.bin"), key=lambda path: int(path.stem))
    if args.frames:
        requested = {f"{int(item):06d}" for item in args.frames.split(",") if item.strip()}
        paths = [path for path in paths if path.stem in requested]
    return paths[args.start:args.stop:args.stride]


def _existing_is_valid(path: Path, target_points: int) -> bool:
    try:
        array = np.load(path, allow_pickle=False)
        _validate_array(array, target_points)
        return True
    except (OSError, ValueError):
        return False


def _aggregate(records: list[dict], failed: list[dict]) -> dict:
    summary = {"successful_frames": len(records), "failed_frames": len(failed), "failures": failed}
    for input_name, output_name in (("raw_candidate_points", "candidate_points"),
                                    ("final_radius", "final_radius"),
                                    ("runtime_sec", "runtime_sec"),
                                    ("nn_mean", "nn_mean"), ("nn_min", "nn_min")):
        values = [record[input_name] for record in records if input_name in record]
        if values:
            summary[output_name] = {
                "min": float(np.min(values)), "p05": float(np.quantile(values, .05)),
                "median": float(np.median(values)), "mean": float(np.mean(values)),
                "p95": float(np.quantile(values, .95)), "max": float(np.max(values)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data-12/M2024-HWZ/KITTI_Odometry"))
    parser.add_argument("--sequences", default=DEFAULT_SEQUENCES)
    parser.add_argument("--target-points", type=int, default=180000)
    parser.add_argument("--output-dir", default="gt_possion")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames", help="comma-separated frame IDs; applied before start/stop/stride")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", choices=("cuda", "none"), default="cuda")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/research_v2/gt_poisson_generation"))
    parser.add_argument("--max-range", type=float, default=50.0)
    parser.add_argument("--min-z", type=float, default=-4.0)
    parser.add_argument("--max-z", type=float)
    parser.add_argument("--viewpoint-voxel-size", type=float, default=10.0)
    parser.add_argument("--initial-radius-high", type=float, default=.05)
    parser.add_argument("--target-tolerance", type=int, default=2000)
    parser.add_argument("--refinement-passes", type=int, default=8)
    args = parser.parse_args()
    if args.target_points <= 0 or args.stride <= 0:
        raise ValueError("target-points and stride must be positive")
    root = args.data_root.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    all_records, all_failures = [], []
    for sequence in _parse_sequences(args.sequences):
        sequence_dir = root / sequence
        map_path = sequence_dir / "map_clean.npy"
        if not map_path.is_file():
            raise FileNotFoundError(map_path)
        poses = load_poses(sequence_dir)
        map_xyz, map_labels = load_map(map_path)
        output_dir = sequence_dir / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        records, failures = [], []
        for offset, scan_path in enumerate(_frame_paths(sequence_dir, args), 1):
            frame = scan_path.stem
            output_path = output_dir / f"{frame}.npy"
            if args.skip_existing and _existing_is_valid(output_path, args.target_points):
                records.append({"sequence": sequence, "frame": frame, "status": "skipped_valid"})
                continue
            started = time.perf_counter()
            try:
                frame_id = int(frame)
                if frame_id >= len(poses):
                    raise ValueError("missing matching pose")
                candidate = _full_candidate(scan_path, poses[frame_id], map_xyz, map_labels, args)
                if len(candidate) < args.target_points:
                    raise ValueError(f"candidate_lt_{args.target_points}: {len(candidate)}")
                frame_seed = args.seed + int(sequence) * 100000 + frame_id
                selected, diagnostics = select_largest_feasible_radius(
                    candidate[:, :3], target_points=args.target_points, seed=frame_seed,
                    initial_high=args.initial_radius_high, tolerance=args.target_tolerance,
                    refinement_passes=args.refinement_passes,
                )
                output = np.ascontiguousarray(candidate[selected], dtype=np.float32)
                _validate_array(output, args.target_points)
                # This explicitly guards the contract that labels and XYZ came
                # from identical source rows rather than a later relabelling pass.
                if not np.array_equal(output, candidate[selected]):
                    raise AssertionError("index-preserving label alignment failed")
                quality = _knn_quality(output[:, :3], diagnostics["final_radius"], args.device)
                _atomic_save(output_path, output)
                record = {"sequence": sequence, "frame": frame, "status": "ok",
                          "raw_candidate_points": int(len(candidate)), "target_points": args.target_points,
                          "seed": int(frame_seed), "runtime_sec": time.perf_counter() - started,
                          **diagnostics, **quality, "exact_duplicate_count": 0}
                records.append(record)
            except Exception as error:  # frame failures are recorded and do not stop a run
                failure = {"sequence": sequence, "frame": frame, "status": "failed",
                           "reason": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
                continue
            if offset % 100 == 0:
                done = [x for x in records if x.get("status") == "ok"]
                print(json.dumps({"sequence": sequence, "completed": offset,
                                  "ok": len(done), "failed": len(failures),
                                  **_aggregate(done, failures)}, sort_keys=True), flush=True)
        sequence_summary = {"sequence": sequence, "output_dir": str(output_dir),
                            "expected_frames": len(_frame_paths(sequence_dir, args)),
                            "target_points": args.target_points, "seed": args.seed,
                            "records": records, **_aggregate([x for x in records if x.get("status") == "ok"], failures)}
        (sequence_dir / "gt_possion.meta.json").write_text(json.dumps(sequence_summary, indent=2) + "\n")
        all_records.extend(records)
        all_failures.extend(failures)
    summary = {"data_root": str(root), "sequences": _parse_sequences(args.sequences),
               "target_points": args.target_points, "records": all_records,
               **_aggregate([x for x in all_records if x.get("status") == "ok"], all_failures)}
    (args.log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.log_dir / "failed_frames.txt").write_text("\n".join(
        f"{x['sequence']}/{x['frame']} {x['reason']}" for x in all_failures
    ) + ("\n" if all_failures else ""))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

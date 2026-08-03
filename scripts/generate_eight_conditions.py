#!/usr/bin/env python3
"""Generate and save one or more samples for all FPSGen condition tuples.

The three-bit condition string follows the public FPSGen convention
``[LiDAR, vehicle-layout, road-layout]``. For example, ``101`` keeps LiDAR
and road conditions while masking the vehicle cue. Each generated cloud is
written as a standalone PLY file by default; a JSON manifest records the
condition, seed, inference steps, format, and point count for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
import torch

ALL_CONDITIONS = tuple(f"{index:03b}" for index in range(8))


def load_input_points(path: Path) -> np.ndarray:
    """Load metric XYZ input points from ``.npy``, KITTI ``.bin``, or ``.pcd``."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"KITTI binary input must contain groups of 4 values: {path}")
        points = raw.reshape(-1, 4)
    elif suffix == ".pcd":
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    else:
        raise ValueError(f"Unsupported input type {suffix!r}; use .npy, .bin, or .pcd")

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        raise ValueError(f"Input must have shape [N, >=3] with N>0, got {points.shape}")
    points = points[:, :3].astype(np.float32, copy=False)
    if not np.isfinite(points).all():
        raise ValueError("Input point cloud contains NaN or infinite coordinates")
    return points


def seed_everything(seed: int) -> None:
    """Seed NumPy, Python, and Torch RNGs before one generated sample."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse the standalone eight-condition generation options."""
    parser = argparse.ArgumentParser(
        description="Generate FPSGen point-cloud samples for all LiDAR/vehicle/road conditions."
    )
    parser.add_argument("--input", required=True, help="SemanticKITTI input_/frame.npy or compatible point file")
    parser.add_argument("--point-ckpt", required=True, help="Stage-3 Point Flow checkpoint")
    parser.add_argument("--bev-ckpt", required=True, help="Stage-1 BEV Flow checkpoint")
    parser.add_argument("--refine", default="", help="optional refinement checkpoint; default disables refinement")
    parser.add_argument("--output", required=True, help="directory receiving point clouds and manifest.json")
    parser.add_argument(
        "--output-format", choices=("ply", "pcd"), default="ply",
        help="point-cloud format for generated samples (default: ply)",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=ALL_CONDITIONS, default=list(ALL_CONDITIONS),
        help="condition tuples in LiDAR/vehicle/road bit order (default: all eight)",
    )
    parser.add_argument("--samples-per-condition", type=int, default=1, help="number of seeded samples per tuple")
    parser.add_argument("--bev-steps", type=int, default=10, help="BEV forward-Euler steps")
    parser.add_argument("--point-steps", type=int, default=4, help="PointFlow forward-Euler steps")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="CFG scale passed to the inference wrapper")
    parser.add_argument("--seed", type=int, default=20260801, help="base random seed")
    parser.add_argument("--dataset", choices=("SemanticKITTI", "KITTI360"), default="SemanticKITTI")
    args = parser.parse_args()
    if args.samples_per_condition < 1:
        parser.error("--samples-per-condition must be positive")
    if args.bev_steps < 1 or args.point_steps < 1:
        parser.error("--bev-steps and --point-steps must be positive")
    return args


def generate_samples(
    model: DiffCompletion,
    points: np.ndarray,
    input_path: Path,
    output_dir: Path,
    conditions: Iterable[str],
    samples_per_condition: int,
    seed: int,
    dataset: str,
    bev_steps: int,
    point_steps: int,
    output_format: str,
) -> list[dict]:
    """Run Unified Inference for each condition and save generated clouds."""
    records = []
    for condition_index, condition in enumerate(conditions):
        for sample_index in range(samples_per_condition):
            sample_seed = seed + condition_index * samples_per_condition + sample_index
            seed_everything(sample_seed)
            generated, _ = model.complete_scan(
                (points, None, str(input_path)),
                cond_mode=condition,
                dataset=dataset,
                steps=(bev_steps, point_steps),
            )
            generated = np.asarray(generated, dtype=np.float32)
            if generated.ndim != 2 or generated.shape[1] != 3 or generated.shape[0] == 0:
                raise RuntimeError(
                    f"Condition {condition} returned invalid cloud shape {generated.shape}"
                )
            if not np.isfinite(generated).all():
                raise RuntimeError(f"Condition {condition} returned non-finite coordinates")

            output_path = output_dir / f"condition_{condition}_sample_{sample_index:02d}.{output_format}"
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(generated)
            if not o3d.io.write_point_cloud(str(output_path), cloud, write_ascii=False):
                raise IOError(f"Open3D failed to write {output_path}")

            records.append(
                {
                    "condition": condition,
                    "condition_order": ["lidar", "vehicle", "road"],
                    "seed": sample_seed,
                    "sample_index": sample_index,
                    "bev_steps": bev_steps,
                    "point_steps": point_steps,
                    "format": output_format,
                    "num_points": int(generated.shape[0]),
                    "path": output_path.name,
                }
            )
            print(
                f"[OK] condition={condition} sample={sample_index} "
                f"points={generated.shape[0]} path={output_path}"
            )
    return records


def main() -> None:
    """Load one model pair, generate requested conditions, and write a manifest."""
    args = parse_args()
    # Import the CUDA-heavy inference stack only after argument validation so
    # ``--help`` and malformed-input errors remain usable without a GPU.
    from fpsgen.inference import DiffCompletion

    input_path = Path(args.input).expanduser().resolve()
    student_path = Path(args.point_ckpt).expanduser().resolve()
    bev_path = Path(args.bev_ckpt).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not student_path.is_file():
        raise FileNotFoundError(student_path)
    if not bev_path.is_file():
        raise FileNotFoundError(bev_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    points = load_input_points(input_path)
    model = DiffCompletion(
        str(student_path), str(bev_path), args.refine, args.point_steps, args.guidance_scale
    )
    model.eval()
    records = generate_samples(
        model=model,
        points=points,
        input_path=input_path,
        output_dir=output_dir,
        conditions=args.conditions,
        samples_per_condition=args.samples_per_condition,
        seed=args.seed,
        dataset=args.dataset,
        bev_steps=args.bev_steps,
        point_steps=args.point_steps,
        output_format=args.output_format,
    )
    manifest = {
        "input": str(input_path),
        "student_checkpoint": str(student_path),
        "bev_checkpoint": str(bev_path),
        "dataset": args.dataset,
        "guidance_scale": args.guidance_scale,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] wrote {len(records)} {args.output_format.upper()} files and {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Stage-3 PointFlow without loading or executing Stage-1 BEVFlow.

The Student architecture always consumes a three-channel BEV main input.  This
diagnostic supplies an oracle BEV rasterized from a selected GT cloud, making
the result a Student-only PointFlow measurement.  With ``--condition 100`` the
LiDAR scan is the only flexible condition; both semantic-layout channels are
literal zeros.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch

from fpsgen.models import minkunet_refine as minknetin
from fpsgen.models.gen_img import BEVDataProcessor


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        points = np.load(path)
    elif path.suffix == ".ply":
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
    else:
        raise ValueError(f"Unsupported input suffix: {path.suffix}")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
        raise ValueError(f"Invalid point cloud: {path}, shape={points.shape}")
    return points[:, :3]


def write_xyz(path: Path, xyz: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise IOError(f"Could not write {path}")


def write_xyz_label(path: Path, points: np.ndarray) -> None:
    """Write a compact binary PLY with float32 XYZ and semantic label."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4 or not np.isfinite(points[:, :3]).all():
        raise ValueError(f"Expected finite [N, >=4] point rows, got {points.shape}")
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"),
                                          ("z", "<f4"), ("label", "<f4")])
    vertex["x"], vertex["y"], vertex["z"], vertex["label"] = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertex)}\n"
              "property float x\nproperty float y\nproperty float z\nproperty float label\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)


def save_reference_cloud(path: Path, source: Path) -> None:
    """Preserve semantic labels for the GT and LiDAR references next to a run."""
    if source.suffix == ".ply":
        # The Poisson GT is already the required float32 XYZ+label PLY. Copy
        # bytes instead of parsing it through Open3D, which discards labels.
        shutil.copyfile(source, path)
    elif source.suffix == ".npy":
        write_xyz_label(path, np.load(source))
    else:
        raise ValueError(f"Unsupported reference suffix: {source.suffix}")


class OracleBEVStudent:
    def __init__(self, checkpoint: Path, guidance_scale: float) -> None:
        loaded = torch.load(checkpoint, map_location="cpu")
        self.hparams = loaded["hyper_parameters"]
        self.device = torch.device("cuda")
        out_dim = self.hparams["model"]["out_dim"]
        self.partial_enc = minknetin.MinkGlobalEncIN(
            in_channels=3, out_channels=out_dim
        ).to(self.device).eval()
        self.model = minknetin.MinkUNetDiffIN(
            in_channels=3, out_channels=out_dim
        ).to(self.device).eval()
        state = loaded["state_dict"]
        self.partial_enc.load_state_dict(
            {key.removeprefix("partial_enc."): value for key, value in state.items()
             if key.startswith("partial_enc.")}, strict=True
        )
        self.model.load_state_dict(
            {key.removeprefix("model."): value for key, value in state.items()
             if key.startswith("model.")}, strict=True
        )
        self.guidance_scale = float(guidance_scale)
        self.processor = BEVDataProcessor(max_density=50.0, min_z=-4.0, max_z=5.4,
                                          grid_size=256, pc_range=50.0)

    def points_to_tensor(self, points: torch.Tensor) -> ME.TensorField:
        coordinates = ME.utils.batched_coordinates(list(points), dtype=torch.float32,
                                                   device=self.device)
        quantized = coordinates[:, :4].clone()
        quantized[:, 1:] = torch.round(
            coordinates[:, 1:4] / self.hparams["data"]["resolution"]
        )
        return ME.TensorField(
            features=coordinates[:, 1:], coordinates=quantized,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

    def preprocess_lidar(self, scan: np.ndarray) -> torch.Tensor:
        radius = np.linalg.norm(scan, axis=1)
        scan = scan[(radius > 3.5) & (radius < self.hparams["data"]["max_range"])]
        count = int(self.hparams["data"]["num_points"] / 10)
        if len(scan) == 0:
            raise ValueError("No valid LiDAR points after range filtering")
        if len(scan) < count:
            scan = np.concatenate([scan, scan[np.random.choice(len(scan), count - len(scan), replace=True)]])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(scan)
        scan = np.asarray(cloud.farthest_point_down_sample(count).points, dtype=np.float32)
        return torch.from_numpy(scan)[None].to(self.device)

    def sample_source(self, bev: torch.Tensor, target_points: int) -> torch.Tensor:
        density = bev[:, 0].reshape(1, -1).clamp(-1, 1)
        density = torch.expm1((density + 1.0) * 0.5 * math.log1p(50.0))
        occupancy = bev[:, 2].reshape(1, -1) > 0
        grid = 256
        x_index, y_index = torch.meshgrid(torch.arange(grid, device=self.device),
                                          torch.arange(grid, device=self.device), indexing="ij")
        valid_disk = (x_index.float().add(0.5).div(grid).mul(100).sub(50).square()
                      + y_index.float().add(0.5).div(grid).mul(100).sub(50).square()).sqrt() <= 50
        density[~occupancy] = 0
        density[:, ~valid_disk.reshape(-1)] = 0
        if bool((density.sum(dim=1) <= 0).any()):
            raise RuntimeError("Oracle BEV has no occupied source cell")
        sampled = torch.multinomial(density, target_points, replacement=True)
        x = sampled.div(grid, rounding_mode="floor").float()
        y = sampled.remainder(grid).float()
        xyz = torch.stack([(x + .5) / grid * 100 - 50,
                           (y + .5) / grid * 100 - 50,
                           torch.zeros_like(x)], dim=-1)
        return xyz + torch.randn_like(xyz)

    @torch.no_grad()
    def generate(self, scan: np.ndarray, gt: np.ndarray, point_steps: int,
                 trajectory_steps: set[int] | None = None) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        gt_tensor = torch.from_numpy(gt)[None].to(self.device)
        oracle_bev = self.processor.points_to_bev_target(gt_tensor)
        source = self.sample_source(oracle_bev, target_points=gt_tensor.shape[1])
        lidar = self.preprocess_lidar(scan)
        x_cond = self.points_to_tensor(lidar)
        x_uncond = self.points_to_tensor(torch.zeros_like(lidar))
        layout = torch.zeros((1, 2, 256, 256), device=self.device)
        current = self.points_to_tensor(source)
        trajectory_steps = trajectory_steps or set()
        trajectory = {0: source[0].cpu().numpy()} if 0 in trajectory_steps else {}
        for step_index, time_value in enumerate(
                torch.linspace(0, 1, point_steps + 1, device=self.device)[:-1], start=1):
            time = time_value.reshape(1)
            # Build the sparse map exactly once per Euler state.  Rebuilding it
            # for conditional/unconditional CFG calls produces incompatible
            # CoordinateMapKeys inside ME skip concatenations.
            current_sparse = current.sparse()
            if self.guidance_scale == 1:
                part = self.partial_enc(x_cond)
                velocity = self.model(oracle_bev, layout, current, current_sparse, part, time)
            else:
                part_uncond = self.partial_enc(x_uncond)
                part_cond = self.partial_enc(x_cond)
                v_uncond = self.model(oracle_bev, layout, current, current_sparse, part_uncond, time)
                v_cond = self.model(oracle_bev, layout, current, current_sparse, part_cond, time)
                velocity = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            current = self.points_to_tensor(
                (current.F + velocity[:, :3] / point_steps).reshape(1, -1, 3)
            )
            if step_index in trajectory_steps:
                trajectory[step_index] = current.F.cpu().numpy()
        return source[0].cpu().numpy(), current.F.cpu().numpy(), trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="input_/frame.npy")
    parser.add_argument("--gt", type=Path, required=True, help="gt_possion/frame.ply or .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--trajectory-steps", default="",
        help="Comma-separated Euler states to save as PLY, e.g. 0,10,20,30,40,50.",
    )
    parser.add_argument(
        "--trajectory-every", type=int, default=0,
        help="Also save every N Euler states; use 1 to save a full video trajectory.",
    )
    args = parser.parse_args()
    if args.point_steps < 1:
        parser.error("--point-steps must be positive")
    try:
        trajectory_steps = ({int(value) for value in args.trajectory_steps.split(",") if value}
                            if args.trajectory_steps else set())
    except ValueError as error:
        parser.error(f"Invalid --trajectory-steps: {error}")
    if any(step < 0 or step > args.point_steps for step in trajectory_steps):
        parser.error("--trajectory-steps values must be within [0, point-steps]")
    if args.trajectory_every < 0:
        parser.error("--trajectory-every must be non-negative")
    if args.trajectory_every:
        trajectory_steps.update(range(0, args.point_steps + 1, args.trajectory_every))
        trajectory_steps.add(args.point_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = OracleBEVStudent(args.student_ckpt, args.guidance_scale)
    source, prediction, trajectory = runner.generate(
        load_points(args.input), load_points(args.gt), args.point_steps, trajectory_steps
    )
    if prediction.shape != source.shape or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid Student output: source={source.shape}, prediction={prediction.shape}")
    args.output.mkdir(parents=True, exist_ok=True)
    save_reference_cloud(args.output / "gt_possion.ply", args.gt)
    save_reference_cloud(args.output / "lidar_scan.ply", args.input)
    write_xyz(args.output / "source_oracle_bev.ply", source)
    write_xyz(args.output / "student_cond100.ply", prediction)
    for step, points in trajectory.items():
        write_xyz(args.output / f"step_{step:03d}_t{step / args.point_steps:.2f}.ply", points)
    (args.output / "manifest.json").write_text(json.dumps({
        "student_checkpoint": str(args.student_ckpt.resolve()),
        "input": str(args.input.resolve()), "oracle_gt": str(args.gt.resolve()),
        "saved_references": {
            "lidar_scan": "lidar_scan.ply", "gt_possion": "gt_possion.ply",
            "format": "binary_little_endian PLY, float32 x/y/z/label",
        },
        "condition": "100", "layout": "literal_zero", "point_steps": args.point_steps,
        "guidance_scale": args.guidance_scale, "seed": args.seed,
        "source_points": int(len(source)), "prediction_points": int(len(prediction)),
        "trajectory_steps": sorted(trajectory),
    }, indent=2) + "\n")
    print(f"[OK] wrote {len(prediction)} points to {args.output / 'student_cond100.ply'}")


if __name__ == "__main__":
    main()

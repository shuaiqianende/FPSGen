#!/usr/bin/env python3
"""Verify TensorField source-row recovery before adding feature supervision.

This command has two deliberately separate checks.  ``--synthetic-only``
proves the MinkowskiEngine ``slice`` ordering contract in the installed
runtime, including collisions and multiple batches.  Supplying checkpoints
and a SemanticKITTI frame additionally records actual teacher/student decoder
feature shapes and each input cloud's collision ratio.  It never trains.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _environment():
    return {
        "commit": _git_sha(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def _field(points: torch.Tensor, resolution: float, device):
    import MinkowskiEngine as ME
    rows = ME.utils.batched_coordinates(list(points), dtype=torch.float32, device=device)
    coords = rows.clone()
    coords[:, 1:] = torch.round(coords[:, 1:] / resolution)
    return ME.TensorField(
        features=rows[:, 1:], coordinates=coords,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device,
    )


def _collision_stats(points: torch.Tensor, resolution: float):
    quantized = torch.round(points / resolution).to(torch.int64)
    unique = torch.unique(quantized, dim=0).shape[0]
    total = points.shape[0]
    return {"num_points": int(total), "unique_voxels": int(unique),
            "collision_ratio": float(1.0 - unique / total)}


def run_synthetic(resolution: float, device):
    """Check that slicing returns one row per original field row, in order."""
    import MinkowskiEngine as ME
    # Repeated and near-boundary coordinates exercise collision handling; the
    # two batches deliberately reuse spatial voxels to catch cross-batch leaks.
    points = torch.tensor([
        [[0.001, 0.001, 0.0], [0.002, 0.002, 0.0], [0.051, 0.0, 0.0], [0.101, 0.0, 0.0]],
        [[0.001, 0.001, 0.0], [0.049, 0.0, 0.0], [0.051, 0.0, 0.0], [0.101, 0.0, 0.0]],
    ], dtype=torch.float32, device=device)
    rows = ME.utils.batched_coordinates(list(points), dtype=torch.float32, device=device)
    coords = rows.clone()
    coords[:, 1:] = torch.round(coords[:, 1:] / resolution)
    source_id = torch.arange(rows.shape[0], dtype=torch.float32, device=device)[:, None]
    field = ME.TensorField(
        features=source_id, coordinates=coords,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device,
    )
    recovered = field.sparse().slice(field).F[:, 0]

    # For each original row, the expected value is the average source ID in
    # its voxel. Exact agreement proves both row ordering and batch isolation.
    expected = torch.empty_like(recovered)
    for i in range(rows.shape[0]):
        same_voxel = (coords == coords[i]).all(dim=1)
        expected[i] = source_id[same_voxel, 0].mean()
    return {
        "synthetic_shape": list(recovered.shape),
        "synthetic_source_identity_preserved": bool(torch.allclose(recovered, expected)),
        "synthetic_max_abs_error": float((recovered - expected).abs().max().item()),
        "synthetic_collision_ratio": float(1 - field.sparse().F.shape[0] / rows.shape[0]),
    }


def _bev_source(gt: torch.Tensor, target_points: int):
    """Standalone copy of the fixed FPSGen BEV source sampler."""
    grid = 256
    pixels = (((gt[:, :2] + 50.0) / 100.0) * grid).long().clamp(0, grid - 1)
    flat = pixels[:, 0] * grid + pixels[:, 1]
    weight = torch.zeros(grid * grid, device=gt.device).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=torch.float32)
    ).add_(1e-8)
    sampled = torch.multinomial(weight, target_points, replacement=True)
    x = ((sampled // grid).float() + .5) / grid * 100.0 - 50.0
    y = ((sampled % grid).float() + .5) / grid * 100.0 - 50.0
    anchors = torch.stack((x, y, torch.zeros_like(x)), dim=-1)
    return anchors + torch.randn_like(anchors) * torch.tensor(
        [1.0, 1.0, 1.0], device=gt.device
    )


def run_real(args, device):
    import MinkowskiEngine as ME  # noqa: F401 - clear runtime dependency
    import fpsgen.models.minkunet as teacher_net
    import fpsgen.models.minkunet_refine as student_net
    from fpsgen.models.gen_img import BEVDataProcessor

    teacher_checkpoint = torch.load(args.teacher_ckpt, map_location="cpu")
    student_checkpoint = torch.load(args.student_ckpt, map_location="cpu")
    teacher_encoder = teacher_net.MinkGlobalEnc(in_channels=3, out_channels=96).to(device).eval()
    teacher_model = teacher_net.MinkUNet_NoTime(in_channels=3, out_channels=96).to(device).eval()
    student_encoder = student_net.MinkGlobalEncIN(in_channels=3, out_channels=96).to(device).eval()
    student_model = student_net.MinkUNetDiffIN(in_channels=3, out_channels=96).to(device).eval()

    def select(state, prefix):
        return {key.removeprefix(prefix): value for key, value in state["state_dict"].items()
                if key.startswith(prefix)}
    teacher_encoder.load_state_dict(select(teacher_checkpoint, "partial_enc."), strict=True)
    teacher_model.load_state_dict(select(teacher_checkpoint, "model."), strict=True)
    student_encoder.load_state_dict(select(student_checkpoint, "partial_enc."), strict=True)
    student_model.load_state_dict(select(student_checkpoint, "model."), strict=True)

    gt_np = np.load(args.frame)
    if gt_np.ndim != 2 or gt_np.shape[1] < 4:
        raise ValueError("frame must be a GT .npy array containing XYZ and semantic label")
    gt = torch.from_numpy(gt_np[:, :3]).float().to(device)
    labels = torch.from_numpy(gt_np[:, 3:]).float().to(device)
    if gt.shape[0] != args.num_points:
        raise ValueError(f"expected {args.num_points} points, got {gt.shape[0]}")
    part_path = str(args.frame).replace("/gt_/", "/input_/")
    part = torch.from_numpy(np.load(part_path)[:, :3]).float().to(device)
    source = _bev_source(gt, args.num_points)
    x_source, x_gt, x_part = (_field(z[None], args.resolution, device) for z in (source, gt, part))

    with torch.no_grad():
        part_teacher = teacher_encoder(x_gt)
        teacher_residual, teacher_features = teacher_model(
            x_source, x_source.sparse(), part_teacher, return_features=True
        )
        endpoint = source - teacher_residual
        t = torch.tensor([0.5], device=device)
        point_state = .5 * (source + endpoint)
        x_state = _field(point_state[None], args.resolution, device)
        processor = BEVDataProcessor(max_density=50., min_z=-4., max_z=5.4,
                                     grid_size=256, pc_range=50.)
        img = processor.points_to_bev_target(gt[None])
        layout = processor.get_layout_bev(gt[None], labels[None]) * 2.0 - 1.0
        part_student = student_encoder(x_part)
        _, student_features = student_model(
            img, layout, x_state, x_state.sparse(), part_student, t,
            return_features=True,
        )
    return {
        "frame": str(args.frame),
        "teacher_shape": list(teacher_features["final_point_feature"].shape),
        "student_shape": list(student_features["final_point_feature"].shape),
        "num_source_points": args.num_points,
        "teacher": _collision_stats(source, args.resolution),
        "student": _collision_stats(point_state, args.resolution),
        "elapsed_ms": None,  # wall-clock belongs to the separate runtime benchmark.
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/research_v2/feature_correspondence/report.json"))
    parser.add_argument("--resolution", type=float, default=.05)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--teacher-ckpt", type=Path)
    parser.add_argument("--student-ckpt", type=Path)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--num-points", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"environment": _environment(), "seed": args.seed,
              "resolution": args.resolution, "real": None,
              # Stable Gate-1 schema. These are populated by --frame rather
              # than inferred from a synthetic test.
              "teacher_shape": None, "student_shape": None,
              "num_source_points": None, "source_identity_preserved": None,
              "teacher_unique_voxels": None, "student_unique_voxels": None,
              "teacher_collision_ratio": None, "student_collision_ratio": None}
    report.update(run_synthetic(args.resolution, device))
    # The synthetic result is the direct source-ID proof. It is meaningful
    # even before a checkpoint/frame is supplied, so expose it in the stable
    # Gate-1 field rather than leaving a successful A0 run ambiguous.
    report["source_identity_preserved"] = report["synthetic_source_identity_preserved"]
    if not args.synthetic_only:
        missing = [name for name in ("teacher_ckpt", "student_ckpt", "frame") if getattr(args, name) is None]
        if missing:
            raise ValueError("Real diagnostic requires " + ", ".join("--" + x.replace("_", "-") for x in missing))
        report["real"] = run_real(args, device)
        real = report["real"]
        report.update({
            "teacher_shape": real["teacher_shape"],
            "student_shape": real["student_shape"],
            "num_source_points": real["num_source_points"],
            "source_identity_preserved": report["synthetic_source_identity_preserved"],
            "teacher_unique_voxels": real["teacher"]["unique_voxels"],
            "student_unique_voxels": real["student"]["unique_voxels"],
            "teacher_collision_ratio": real["teacher"]["collision_ratio"],
            "student_collision_ratio": real["student"]["collision_ratio"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Gate A over a frame manifest while loading both checkpoints once."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from diagnose_feature_correspondence import (_bev_source, _collision_stats, _environment,
                                             _field, run_synthetic)


def _manifest(path: Path):
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _summary(values, *, percentiles=(.05, .5, .95)):
    array = np.asarray(values, dtype=np.float64)
    result = {"mean": float(array.mean()), "median": float(np.median(array)),
              "min": float(array.min()), "max": float(array.max())}
    for percentile in percentiles:
        result[f"p{int(percentile * 100):02d}"] = float(np.quantile(array, percentile))
    return result


class FeatureGateRunner:
    """Frozen Teacher/Student pair reused for all frames in a Gate-A run."""

    def __init__(self, teacher_ckpt: Path, student_ckpt: Path, device):
        import fpsgen.models.minkunet as teacher_net
        import fpsgen.models.minkunet_refine as student_net
        self.device = device
        teacher_state = torch.load(teacher_ckpt, map_location="cpu")["state_dict"]
        student_state = torch.load(student_ckpt, map_location="cpu")["state_dict"]
        self.teacher_encoder = teacher_net.MinkGlobalEnc(in_channels=3, out_channels=96).to(device).eval()
        self.teacher_model = teacher_net.MinkUNet_NoTime(in_channels=3, out_channels=96).to(device).eval()
        self.student_encoder = student_net.MinkGlobalEncIN(in_channels=3, out_channels=96).to(device).eval()
        self.student_model = student_net.MinkUNetDiffIN(in_channels=3, out_channels=96).to(device).eval()

        def select(state, prefix):
            return {key.removeprefix(prefix): value for key, value in state.items()
                    if key.startswith(prefix)}
        self.teacher_encoder.load_state_dict(select(teacher_state, "partial_enc."), strict=True)
        self.teacher_model.load_state_dict(select(teacher_state, "model."), strict=True)
        self.student_encoder.load_state_dict(select(student_state, "partial_enc."), strict=True)
        self.student_model.load_state_dict(select(student_state, "model."), strict=True)

        from fpsgen.models.gen_img import BEVDataProcessor
        self.processor = BEVDataProcessor(max_density=50., min_z=-4., max_z=5.4,
                                           grid_size=256, pc_range=50.)

    @torch.no_grad()
    def evaluate(self, frame: Path, resolution: float, num_points: int):
        raw = np.load(frame)
        if raw.ndim != 2 or raw.shape[1] < 4 or raw.shape[0] != num_points:
            raise ValueError(f"Expected {num_points} XYZ+label rows in {frame}, got {raw.shape}")
        part_path = Path(str(frame).replace("/gt_/", "/input_/"))
        part_raw = np.load(part_path)
        gt = torch.from_numpy(raw[:, :3]).float().to(self.device)
        labels = torch.from_numpy(raw[:, 3:]).float().to(self.device)
        part = torch.from_numpy(part_raw[:, :3]).float().to(self.device)
        source = _bev_source(gt, num_points)
        x_source, x_gt, x_part = (_field(x[None], resolution, self.device)
                                  for x in (source, gt, part))
        start = time.perf_counter()
        teacher_residual, teacher_features = self.teacher_model(
            x_source, x_source.sparse(), self.teacher_encoder(x_gt), return_features=True
        )
        endpoint = source - teacher_residual
        point_state = .5 * (source + endpoint)
        x_state = _field(point_state[None], resolution, self.device)
        img = self.processor.points_to_bev_target(gt[None])
        layout = self.processor.get_layout_bev(gt[None], labels[None]) * 2. - 1.
        _, student_features = self.student_model(
            img, layout, x_state, x_state.sparse(), self.student_encoder(x_part),
            torch.tensor([.5], device=self.device), return_features=True,
        )
        teacher_stats, student_stats = (_collision_stats(x, resolution)
                                        for x in (source, point_state))
        both_singleton = (teacher_stats.pop("_singleton_mask") &
                          student_stats.pop("_singleton_mask"))
        teacher_feature = teacher_features["final_point_feature"]
        student_feature = student_features["final_point_feature"]
        return {
            "frame": frame.stem, "num_source_points": num_points,
            "teacher_shape": list(teacher_feature.shape),
            "student_shape": list(student_feature.shape),
            "teacher": teacher_stats, "student": student_stats,
            "both_singleton_ratio": float(both_singleton.float().mean().item()),
            "teacher_output_rows_valid": teacher_feature.shape[0] == num_points,
            "student_output_rows_valid": student_feature.shape[0] == num_points,
            "teacher_feature_finite": bool(torch.isfinite(teacher_feature).all().item()),
            "student_feature_finite": bool(torch.isfinite(student_feature).all().item()),
            "runtime_ms": (time.perf_counter() - start) * 1e3,
            "peak_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                                if torch.cuda.is_available() else None),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", default="08")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/research_v2/feature_correspondence/a2_20"))
    parser.add_argument("--num-points", type=int, default=180000)
    parser.add_argument("--resolution", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gate A2 requires CUDA; refusing 180k-point CPU execution")
    stems = _manifest(args.manifest)
    if not stems:
        raise ValueError("Manifest is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    synthetic = run_synthetic(args.resolution, device)
    if not synthetic["synthetic_source_identity_preserved"]:
        raise RuntimeError("Gate A0 regression failed; refusing A2")
    runner = FeatureGateRunner(args.teacher_ckpt, args.student_ckpt, device)
    records = []
    with (args.output / "per_frame.jsonl").open("w") as stream:
        for index, stem in enumerate(stems):
            torch.manual_seed(args.seed + index)
            torch.cuda.reset_peak_memory_stats()
            frame = args.dataset_root / args.sequence / "gt_" / f"{stem}.npy"
            record = runner.evaluate(frame, args.resolution, args.num_points)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            records.append(record)
            print(f"completed {stem}")
    summary = {
        "environment": _environment(), "seed": args.seed, "manifest": str(args.manifest),
        "num_frames": len(records), "synthetic": synthetic,
        "source_row_order_preserved": True,
        "teacher_collision_ratio": _summary([x["teacher"]["collision_ratio"] for x in records]),
        "student_collision_ratio": _summary([x["student"]["collision_ratio"] for x in records]),
        "both_singleton_ratio": _summary([x["both_singleton_ratio"] for x in records]),
        "teacher_output_row_valid_ratio": float(np.mean([x["teacher_output_rows_valid"] for x in records])),
        "student_output_row_valid_ratio": float(np.mean([x["student_output_rows_valid"] for x in records])),
        "teacher_finite_ratio": float(np.mean([x["teacher_feature_finite"] for x in records])),
        "student_finite_ratio": float(np.mean([x["student_feature_finite"] for x in records])),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

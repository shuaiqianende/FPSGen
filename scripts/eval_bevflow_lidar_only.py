#!/usr/bin/env python3
"""Evaluate Stage-1 BEVFlow with only the LiDAR (condition ``100``) enabled.

This script deliberately does not instantiate the Teacher or PointFlow.  It
implements the production CFG Euler BEV sampler and compares generated
``[density, height, occupancy]`` directly against ``BEVDataProcessor`` GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fpsgen.datasets.dataloader.semantic_kitti import TemporalKITTISet
from fpsgen.models.gen_img import FlowIMG
from fpsgen.utils.bev_eval_metrics import (
    compute_bev_metrics,
    compute_range_metrics,
    decode_height,
    make_valid_disk_mask,
    process_generated_bev,
)
from fpsgen.utils.collations import SparseSegmentCollationGen


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(rows: List[Dict], *, skip=()) -> Dict[str, Dict[str, float]]:
    output = {}
    for key in sorted({key for row in rows for key in row}):
        if key in skip:
            continue
        values = [row[key] for row in rows if isinstance(row.get(key), (float, int, np.floating, np.integer))]
        if values:
            output[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return output


@torch.no_grad()
def sample_bev_cfg(flow: FlowIMG, raw_pc_cond: torch.Tensor, layout_cond: torch.Tensor,
                   *, steps: int, guidance_scale: float, seed: int) -> tuple[torch.Tensor, float]:
    """Production-equivalent CFG forward-Euler sampling without PointFlow."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if raw_pc_cond.ndim != 4 or layout_cond.ndim != 4:
        raise ValueError("LiDAR condition and layout must be BCHW tensors")
    if torch.count_nonzero(layout_cond).item() != 0:
        raise ValueError("LiDAR-only evaluation requires an all-zero layout mask")
    device = raw_pc_cond.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    batch_size = raw_pc_cond.shape[0]
    xt = torch.randn((batch_size, 3, 256, 256), device=device, generator=generator)
    raw_pc_uncond = torch.zeros_like(raw_pc_cond)
    layout_uncond = torch.zeros_like(layout_cond)
    dt = 1.0 / steps
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for i in range(steps):
        t = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        if guidance_scale > 1.0:
            velocities = flow.model(
                torch.cat((xt, xt)), torch.cat((t, t)),
                torch.cat((raw_pc_uncond, raw_pc_cond)),
                torch.cat((layout_uncond, layout_cond)),
            )
            velocity_uncond, velocity_cond = velocities.chunk(2, dim=0)
            velocity = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
        else:
            velocity = flow.model(xt, t, raw_pc_cond, layout_cond)
        xt = xt + dt * velocity
    torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    if not torch.isfinite(xt).all():
        raise FloatingPointError("Non-finite BEV generated during Euler sampling")
    return xt, runtime_ms


def _save_visual(path: Path, input_bev: torch.Tensor, gt_bev: torch.Tensor,
                 pred_bev: torch.Tensor) -> None:
    """Save the requested 4x3 LiDAR/GT/prediction/absolute-error panel."""
    pred = process_generated_bev(pred_bev.unsqueeze(0))[0]
    input_bev, gt_bev = input_bev.cpu(), gt_bev.cpu()
    pred = pred.cpu()
    rows = [input_bev, gt_bev, pred]
    titles = ("Density (normalized log)", "Height (m)", "Occupancy")
    fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)
    for row, bev in enumerate(rows):
        display = (bev[0], decode_height(bev[1]), bev[2])
        for col, image in enumerate(display):
            cmap = "turbo" if col == 0 else ("terrain" if col == 1 else "gray")
            limits = (-1, 1) if col != 1 else (-4, 5.4)
            axes[row, col].imshow(image.numpy(), cmap=cmap, vmin=limits[0], vmax=limits[1])
            axes[row, col].set_title(titles[col] if row == 0 else "")
            axes[row, col].set_ylabel(("LiDAR input", "Full GT", "BEVFlow")[row])
            axes[row, col].axis("off")
    errors = ((pred[0] - gt_bev[0]).abs(),
              (decode_height(pred[1]) - decode_height(gt_bev[1])).abs(),
              (pred[2] - gt_bev[2]).abs())
    for col, image in enumerate(errors):
        axes[3, col].imshow(image.numpy(), cmap="magma")
        axes[3, col].set_ylabel("Absolute error")
        axes[3, col].axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _resolve_gt_dir(checkpoint: Mapping, requested: str, allow_mismatch: bool) -> str:
    checkpoint_data = checkpoint.get("hyper_parameters", {}).get("data", {})
    checkpoint_gt_dir = checkpoint_data.get("gt_dir", "gt_")
    if requested == "auto":
        return checkpoint_gt_dir
    if requested != checkpoint_gt_dir and not allow_mismatch:
        raise ValueError(
            f"--gt-dir={requested!r} conflicts with checkpoint gt_dir={checkpoint_gt_dir!r}; "
            "use --allow-gt-mismatch only for an explicitly non-formal diagnostic."
        )
    return requested


def _load_manifest(path: Path) -> List[str]:
    frames = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not frames:
        raise ValueError(f"No frame stems in manifest: {path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research_v2/eval_bevflow_lidar_only.yaml")
    parser.add_argument("--bev-ckpt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--gt-dir", default="auto")
    parser.add_argument("--allow-gt-mismatch", action="store_true")
    parser.add_argument("--samples-per-frame", type=int, default=None)
    parser.add_argument("--save-visuals", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None,
                        help="Override eval.base_seed for a fixed campaign seed.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    evaluate = config["eval"]
    data_config = config["data"]
    if evaluate["condition"] != "100":
        raise ValueError("This evaluator is intentionally restricted to LiDAR-only condition=100")
    if not torch.cuda.is_available():
        raise RuntimeError("BEVFlow evaluation requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(args.bev_ckpt, map_location="cpu")
    if "state_dict" not in checkpoint or "hyper_parameters" not in checkpoint:
        raise KeyError("BEV checkpoint must contain state_dict and hyper_parameters")
    gt_dir = _resolve_gt_dir(checkpoint, args.gt_dir, args.allow_gt_mismatch)
    raw_data_root = args.dataset_root or checkpoint["hyper_parameters"].get("data", {}).get("data_dir", "")
    if not raw_data_root:
        raise ValueError("Set --dataset-root or use a checkpoint with data.data_dir")
    data_root = Path(raw_data_root)
    manifest = Path(args.manifest or evaluate["manifest"])
    frame_stems = _load_manifest(manifest)
    samples_per_frame = args.samples_per_frame or int(evaluate["samples_per_frame"])
    visual_limit = args.save_visuals if args.save_visuals is not None else int(evaluate["save_visuals"])
    if samples_per_frame < 1:
        raise ValueError("samples_per_frame must be positive")

    flow = FlowIMG(checkpoint["hyper_parameters"])
    flow.load_state_dict(checkpoint["state_dict"], strict=True)
    flow.to(device).eval()
    dataset = TemporalKITTISet(
        data_dir=str(data_root), seqs=[str(data_config["sequence"])], split="test",
        resolution=.05, num_points=int(data_config["num_points"]),
        max_range=float(data_config["max_range"]), gt_dir=gt_dir,
    )
    index_for_stem = {Path(item).stem: i for i, item in enumerate(dataset.points_datapath)}
    missing = [stem for stem in frame_stems if stem not in index_for_stem]
    if missing:
        raise FileNotFoundError(f"Manifest frames absent from {gt_dir}: {missing[:10]}")
    loader = DataLoader(Subset(dataset, [index_for_stem[stem] for stem in frame_stems]),
                        batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=SparseSegmentCollationGen())
    valid_mask = make_valid_disk_mask(256, float(data_config["max_range"]), device=device)
    output = args.output
    visuals = output / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    sample_rows: List[Dict] = []
    input_rows: List[Dict] = []
    range_rows: List[Dict] = []
    saved_visuals = 0
    radial_bins = evaluate["radial_bins"]
    base_seed = int(evaluate["base_seed"] if args.base_seed is None else args.base_seed)

    for frame_index, batch in enumerate(loader):
        stem = frame_stems[frame_index]
        full = batch["pcd_full"].to(device, non_blocking=True).float()
        part = batch["pcd_part"].to(device, non_blocking=True).float()
        with torch.no_grad():
            gt_bev = flow.processor.points_to_bev_target(full)
            input_bev = flow.processor.points_to_bev_target(part)
            # No semantic layout is built or read in this script.
            layout_cond = torch.zeros((1, 2, 256, 256), device=device, dtype=torch.float32)
            raw_pc_cond = flow.model.get_raw_pc_bev(part)
        if torch.count_nonzero(raw_pc_cond).item() == 0:
            raise RuntimeError(f"LiDAR PointPillar condition is unexpectedly all-zero for {stem}")
        baseline = compute_bev_metrics(input_bev, gt_bev, input_bev, valid_mask)[0]
        baseline.update({"frame": stem, "method": "lidar_input"})
        input_rows.append(baseline)
        for result in compute_range_metrics(input_bev, gt_bev, input_bev,
                                            pc_range=float(data_config["max_range"]), radial_bins=radial_bins):
            result.update({"frame": stem, "seed": "input", "method": "lidar_input"})
            range_rows.append(result)

        for sample_index in range(samples_per_frame):
            seed = base_seed + frame_index * 1000 + sample_index
            pred_raw, runtime_ms = sample_bev_cfg(
                flow, raw_pc_cond, layout_cond, steps=int(evaluate["bev_steps"]),
                guidance_scale=float(evaluate["guidance_scale"]), seed=seed,
            )
            metrics = compute_bev_metrics(pred_raw, gt_bev, input_bev, valid_mask)[0]
            metrics.update({"frame": stem, "seed": seed, "sample_index": sample_index,
                            "method": "bevflow_lidar_only", "bev_generation_ms": runtime_ms})
            sample_rows.append(metrics)
            for result in compute_range_metrics(pred_raw, gt_bev, input_bev,
                                                pc_range=float(data_config["max_range"]), radial_bins=radial_bins):
                result.update({"frame": stem, "seed": seed, "sample_index": sample_index,
                               "method": "bevflow_lidar_only"})
                range_rows.append(result)
            if sample_index == 0 and saved_visuals < visual_limit:
                _save_visual(visuals / f"{stem}_seed0.png", input_bev[0], gt_bev[0], pred_raw[0])
                saved_visuals += 1

    per_frame_rows: List[Dict] = []
    by_frame: Dict[str, List[Dict]] = defaultdict(list)
    for row in sample_rows:
        by_frame[row["frame"]].append(row)
    baseline_by_frame = {row["frame"]: row for row in input_rows}
    metric_keys = sorted({key for row in sample_rows for key, value in row.items()
                          if isinstance(value, (int, float, np.integer, np.floating))
                          and key not in {"seed", "sample_index"}})
    for stem in frame_stems:
        row = {"frame": stem}
        flow_rows = by_frame[stem]
        for key in metric_keys:
            values = [float(item[key]) for item in flow_rows if key in item]
            row[f"bevflow_{key}_mean"] = float(np.mean(values))
            row[f"bevflow_{key}_std"] = float(np.std(values))
        for key, value in baseline_by_frame[stem].items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[f"input_{key}"] = float(value)
        per_frame_rows.append(row)

    headline_lower = ("density_mass_tv", "height_mae_gtocc_m")
    headline_higher = ("occupancy_iou", "completion_f1")
    win_rates = {}
    for key in headline_lower:
        win_rates[key] = float(np.mean([row[f"bevflow_{key}_mean"] < row[f"input_{key}"]
                                        for row in per_frame_rows]))
    for key in headline_higher:
        win_rates[key] = float(np.mean([row[f"bevflow_{key}_mean"] > row[f"input_{key}"]
                                        for row in per_frame_rows]))

    summary = {
        "checkpoint": str(args.bev_ckpt.resolve()),
        "checkpoint_sha256": _sha256(args.bev_ckpt),
        "git_commit": _git_sha(),
        "gt_dir": gt_dir,
        "condition": "100",
        "layout_source": "literal_all_zero_no_semantic_layout",
        "manifest": str(manifest),
        "num_frames": len(frame_stems),
        "samples_per_frame": samples_per_frame,
        "base_seed": base_seed,
        "bev_steps": int(evaluate["bev_steps"]),
        "guidance_scale": float(evaluate["guidance_scale"]),
        "occupancy_threshold": "pred_M > 0",
        "valid_radius_m": float(data_config["max_range"]),
        "input_bev": _mean_std(input_rows, skip=("frame", "method")),
        "bevflow_lidar_only": _mean_std(
            [{key.removeprefix("bevflow_").removesuffix("_mean"): value
              for key, value in row.items() if key.startswith("bevflow_") and key.endswith("_mean")}
             for row in per_frame_rows]
        ),
        "mean_stochastic_std_across_frames": _mean_std(
            [{key.removeprefix("bevflow_").removesuffix("_std"): value
              for key, value in row.items() if key.startswith("bevflow_") and key.endswith("_std")}
             for row in per_frame_rows]
        ),
        "improved_frame_ratio": win_rates,
        "runtime": _mean_std(sample_rows, skip=("frame", "seed", "sample_index", "method")),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "samples.csv", sample_rows)
    _write_csv(output / "per_frame.csv", per_frame_rows)
    _write_csv(output / "range_metrics.csv", range_rows)
    _write_csv(output / "completion_metrics.csv", [
        {key: value for key, value in row.items()
         if key in {"frame", "seed", "sample_index", "method"} or key.startswith("completion_")}
        for row in sample_rows + input_rows
    ])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "frames": len(frame_stems),
                      "samples": len(sample_rows), "gt_dir": gt_dir}, indent=2))


if __name__ == "__main__":
    main()

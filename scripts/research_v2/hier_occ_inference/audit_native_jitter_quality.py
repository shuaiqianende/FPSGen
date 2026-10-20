#!/usr/bin/env python3
"""Exact raw-map quality comparison for native CUDA voxel jitter seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import file_sha
from eval_seq08_combinations import physical_grid_ids, raw_chamfer, support_counts
from eval_seq08_skip10_joint_2x4_gpu3 import jsd_metrics
from native_voxel_point_sampling import sample_native_occupied_voxels, sample_native_occupied_voxels_cuda
import skip10_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu", type=int, choices=(2, 3), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--gt-reference", type=Path, required=True,
                   help="Audited seq08 skip100 output root containing targets/08_FRAME.npz")
    p.add_argument("--prediction", type=Path, required=True,
                   help="Saved baseline prediction npz with occupied_coords_xyz")
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--seed-base", type=int, default=20261119)
    p.add_argument("--target-points", type=int, default=1_080_000)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available() or __import__("os").environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("Expose only the selected physical GPU via CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    geometry = SparseHierarchyGeometry(parent_resolution=(384, 384, 32), child_split=(2, 2, 2))
    input_path = a.data_root / "08" / "input_" / f"{a.frame:06d}.npy"
    record = {"frame": a.frame, "input": str(input_path), "input_sha256": file_sha(input_path)}
    skip10_metrics.initialize(a.data_root, geometry.to_dict(), a.gt_reference)
    gt, target_meta = skip10_metrics.load_target(record)
    with np.load(a.prediction, allow_pickle=False) as saved:
        fine = saved["occupied_coords_xyz"].astype(np.int32, copy=False)
    centers, _ = sample_native_occupied_voxels(fine, geometry, point_mode="voxel_center")
    def evaluate(predicted, target):
        result = raw_chamfer(predicted, target, workers=4)
        for resolution, key in ((.5, "iou05"), (.2, "iou02"), (.1, "iou01")):
            result[key] = support_counts(
                physical_grid_ids(predicted, geometry, resolution=resolution),
                physical_grid_ids(target, geometry, resolution=resolution, target_points=True),
            )["iou"]
        result.update(jsd_metrics(target, predicted))
        return result
    rows = []
    for seed in range(a.seed_base, a.seed_base + 3):
        # CPU reference is intentionally not used here: the target is to test
        # seed sensitivity of the new GPU path, converted to CPU for the shared
        # unified metric entry point after sampling.
        start = time.perf_counter()
        prediction_gpu, sample_meta = sample_native_occupied_voxels_cuda(
            fine, geometry, target_points=a.target_points, seed=seed, device=device)
        torch.cuda.synchronize(device)
        sample_seconds = time.perf_counter() - start
        prediction = prediction_gpu.cpu().numpy()
        metrics = evaluate(prediction, gt)
        rows.append({"seed": seed, "sample_seconds": sample_seconds,
                     "sample_metadata": sample_meta, "metrics": metrics})
        print(json.dumps({"seed": seed, "sample_seconds": sample_seconds,
                          **{k: metrics[k] for k in ("cd_m", "iou05", "iou02", "iou01", "jsd_bev", "jsd_3d")}}), flush=True)
    center_metrics = evaluate(centers, gt)
    payload = {
        "physical_gpu": a.physical_gpu, "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "frame": a.frame, "input_sha256": record["input_sha256"],
        "prediction_path": str(a.prediction.resolve()), "occupied_voxels": int(len(fine)),
        "target_points": a.target_points, "native_voxel_size_xyz": geometry.fine_size,
        "gt_raw_map_metadata": target_meta, "center_metrics": center_metrics,
        "jitter_seed_runs": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

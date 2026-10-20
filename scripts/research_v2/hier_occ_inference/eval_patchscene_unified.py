"""Evaluate cached GT-free PatchScene occupancy with the shared FPSGen metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from native_voxel_point_sampling import sample_native_occupied_voxels
from unified_pointcloud_metrics import evaluate_prediction


def _sha_array(arr):
    return hashlib.sha256(memoryview(np.ascontiguousarray(arr)).cast("B")).hexdigest()


class PatchSceneGeometry:
    def __init__(self):
        self.origin = (-50., -50., -3.75)
        self.maximum = (50., 50., 3.75)
        self.fine_size = (.15625, .15625, .15625)
        self.fine_resolution = (640, 640, 48)
        self.sphere_radius = 50.

    def centers(self, coords, *, fine=True):
        c = np.asarray(coords, dtype=np.float64)
        return np.asarray(self.origin) + (c + .5) * np.asarray(self.fine_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage2-config", type=Path, required=True,
                   help="Supplies the frozen data root and common HS evaluation geometry")
    p.add_argument("--physical-gpu", type=int, default=3)
    p.add_argument("--target-points", type=int, default=1_080_000)
    p.add_argument("--seed-base", type=int, default=20261019)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must map only --physical-gpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA metric backend requires CUDA")
    cfg, hs_geometry = load_stage2_config(a.stage2_config)
    data_root = Path(cfg["data_root"])
    source_spec = PatchSceneGeometry()
    pred_protocol = json.loads((a.prediction_dir / "protocol.json").read_text())
    if pred_protocol["status"] != "COMPLETE":
        raise RuntimeError("PatchScene inference must be complete before evaluation")
    protocol = {
        "status": "RUNNING", "source_protocol_sha256": file_sha(a.prediction_dir / "protocol.json"),
        "source_commit": pred_protocol["patchscene_commit"],
        "checkpoint_sha256": pred_protocol["checkpoint_sha256"],
        "frames": pred_protocol["frames"], "gt_protocol": "same RawMapSource fast_viewpoint and HS frozen geometry ROI",
        "evaluation_geometry_sha256": hs_geometry.sha256,
        "evaluation_roi": "XYZ bounds and open 3D sphere from frozen HS geometry",
        "point_modes": ["voxel_center", "native_fine_jitter"],
        "target_points": a.target_points, "seed_base": a.seed_base,
        "metric_backend": "exact GPU KD-tree CD; shared corrected physical-grid IoU; shared sparse FPSGen JSD",
        "native_grid": [640, 640, 48], "native_voxel_size_xyz": list(source_spec.fine_size),
        "no_gt_in_inference": True,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    pp = a.output / "protocol.json"
    journal = a.output / "per_frame.jsonl"
    if a.resume:
        saved = json.loads(pp.read_text()); saved["status"] = "RUNNING"
        if saved != protocol:
            raise RuntimeError("PatchScene metric resume protocol mismatch")
        rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
    else:
        if any(a.output.iterdir()):
            raise FileExistsError("fresh metric output must be empty")
        atomic_json(pp, protocol); rows = []
    done = {int(r["frame"]) for r in rows}

    from skip10_metrics import initialize as initialize_gt, load_target
    reference = ROOT.parent / "hs-occflow-384-stage1occ" / "experiments/hier_occ_flow/seq08_stride100_step024000_20261017"
    initialize_gt(str(data_root), hs_geometry.to_dict(), str(reference))
    metric = GPUKDTreeMetrics(("cd",), geometry=hs_geometry)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    frames = pred_protocol["frames"]
    for frame in frames:
        if int(frame) in done:
            continue
        input_path = data_root / "08" / "input_" / f"{int(frame):06d}.npy"
        source_input_sha = file_sha(input_path)
        pred_file = a.prediction_dir / "predictions" / f"08_{int(frame):06d}.npz"
        with np.load(pred_file, allow_pickle=False) as data:
            coords = data["occupied_coords_xyz"].astype(np.int64, copy=False)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"invalid native occupancy coordinates in {pred_file}")
        target_record = {"sequence": "08", "frame": int(frame),
                         "input": str(input_path), "input_sha256": source_input_sha}
        target, target_meta = load_target(target_record)
        target = np.ascontiguousarray(target, dtype=np.float32)
        target_sha = target_meta["target_sha256"]
        all_centers = source_spec.centers(coords, fine=True).astype(np.float32)
        common_lo = np.asarray(hs_geometry.origin, dtype=np.float64)
        common_hi = np.asarray(hs_geometry.maximum, dtype=np.float64)
        center_xyz64 = all_centers.astype(np.float64)
        common_roi = ((center_xyz64 >= common_lo) & (center_xyz64 < common_hi)).all(axis=1)
        common_roi &= np.einsum("ij,ij->i", center_xyz64, center_xyz64) < hs_geometry.sphere_radius ** 2
        center = np.ascontiguousarray(all_centers[common_roi])
        if not len(center):
            raise RuntimeError(f"PatchScene has no native centers in the shared evaluation ROI: frame {frame}")
        center, center_meta = sample_native_occupied_voxels(
            coords[common_roi], source_spec, point_mode="voxel_center")
        center_metrics = evaluate_prediction(center, target, hs_geometry, metric, device=device)
        jitter, jitter_meta = sample_native_occupied_voxels(
            coords, source_spec, point_mode="native_fine_jitter",
            target_points=a.target_points, seed=int(a.seed_base + int(frame)),
            roi={"sphere_radius": hs_geometry.sphere_radius})
        torch.cuda.synchronize(); t0 = time.perf_counter()
        jitter_metrics = evaluate_prediction(jitter, target, hs_geometry, metric, device=device)
        torch.cuda.synchronize(); metric_seconds = time.perf_counter() - t0
        row = {"sequence": "08", "frame": int(frame),
               "input_sha256": source_input_sha, "gt_sha256": target_sha,
               "gt_points": int(len(target)), "native_occupied_voxels": len(coords),
               "native_centers_in_common_roi": int(common_roi.sum()),
               "native_centers_outside_common_roi": int((~common_roi).sum()),
               **{f"center_{k}": v for k, v in center_metrics.items()},
               **{f"jitter_{k}": v for k, v in jitter_metrics.items()},
               "native_voxel_size_xyz": list(source_spec.fine_size),
               "jitter_seed": int(a.seed_base + int(frame)),
               "jitter_sampling_seconds": jitter_meta["sampling_seconds"],
               "jitter_metric_seconds": metric_seconds,
               "jitter_point_count": jitter_meta["actual_output_points"],
               "jitter_dropped_voxels": jitter_meta["dropped_occupied_voxels"],
               "jitter_dropped_by_roi_voxels": jitter_meta["dropped_invalid_roi_voxels"],
               "jitter_dropped_unrepresentable_voxels": jitter_meta["dropped_unrepresentable_roi_voxels"],
               "jitter_dropped_by_point_budget_voxels": jitter_meta["dropped_by_point_budget_voxels"],
               "jitter_sampled_occupied_voxels": jitter_meta["sampled_occupied_voxels"],
               "jitter_coverage_ratio": jitter_meta["voxel_coverage_ratio"],
               "center_point_count": center_meta["actual_output_points"]}
        with journal.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row); done.add(int(frame))
        atomic_json(a.output / "progress.json", {"status": "RUNNING", "completed_frames": len(rows),
            "requested_frames": len(frames), "last_frame": int(frame),
            "elapsed_seconds": time.perf_counter() - started})
        print(f"{len(rows)}/{len(frames)} frame={int(frame):04d} native={len(coords)} "
              f"centerCD={row['center_cd_m']:.5f} jitterCD={row['jitter_cd_m']:.5f} "
              f"points={row['jitter_point_count']}", flush=True)

    if len(done) != len(frames):
        raise RuntimeError(f"incomplete metrics {len(done)}/{len(frames)}")
    summaries = []
    for mode in ("center", "jitter"):
        item = {"mode": mode, "frames": len(rows)}
        for name in ("cd_m", "iou05", "iou02", "iou01", "jsd_bev", "jsd_3d"):
            item[f"{name}_macro"] = float(np.mean([r[f"{mode}_{name}"] for r in rows]))
        for name in ("iou05", "iou02", "iou01"):
            tp = sum(r[f"{mode}_{name}_tp"] for r in rows)
            den = sum(r[f"{mode}_{name}_tp"] + r[f"{mode}_{name}_fp"] + r[f"{mode}_{name}_fn"] for r in rows)
            item[f"{name}_micro"] = tp / max(den, 1)
        item["mean_point_count"] = float(np.mean([r[f"{mode}_predicted_points"] for r in rows]))
        summaries.append(item)
    protocol["status"] = "COMPLETE"; atomic_json(pp, protocol)
    result = {"status": "COMPLETE", "summary": summaries, "rows": len(rows),
              "elapsed_seconds": time.perf_counter() - started,
              "mean_jitter_seconds": float(np.mean([r["jitter_metric_seconds"] + r["jitter_sampling_seconds"] for r in rows]))}
    atomic_json(a.output / "results.json", result)
    atomic_json(a.output / "progress.json", result)
    with (a.output / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    with (a.output / "per_frame.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

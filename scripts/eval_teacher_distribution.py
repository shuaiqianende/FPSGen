#!/usr/bin/env python3
"""Compare Teacher endpoint geometry and spatial distributions on cached tuples.

Each cache directory contains the immutable ``p0``, ``endpoint`` and ``gt``
tensors written by ``scripts/cache_teacher_endpoints.py``.  Supplying multiple
directories makes the script reject a comparison unless every frame has the
same source and target tensor bit-for-bit.  Sinkhorn is optional and uses one
fixed parameter set for every supplied Teacher.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fpsgen.ops.endpoint_refinement.knn import knn
from fpsgen.ops.endpoint_refinement.sparse_global_ot import refine_endpoint


HEIGHT_EDGES = (-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.4)
RANGE_EDGES = (0.0, 20.0, 35.0, 50.0)


def _scalar(value: torch.Tensor) -> float:
    return float(value.item())


def _histogram(points: torch.Tensor, edges: tuple[float, ...], value: torch.Tensor) -> list[dict]:
    """Return fixed-bin counts/fractions for a scalar point attribute."""
    total = max(int(points.shape[0]), 1)
    output = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (value >= low) & (value < high)
        count = int(mask.sum().item())
        output.append({"low": low, "high": high, "count": count, "fraction": count / total})
    return output


def _distribution(points: torch.Tensor, gt: torch.Tensor) -> dict:
    pred_height = _histogram(points, HEIGHT_EDGES, points[:, 2])
    gt_height = _histogram(gt, HEIGHT_EDGES, gt[:, 2])
    height_abs = [abs(p["fraction"] - t["fraction"]) for p, t in zip(pred_height, gt_height)]

    pred_range = torch.linalg.vector_norm(points[:, :2], dim=1)
    gt_range = torch.linalg.vector_norm(gt[:, :2], dim=1)
    range_height = []
    for r_low, r_high in zip(RANGE_EDGES[:-1], RANGE_EDGES[1:]):
        pred_r = (pred_range >= r_low) & (pred_range < r_high)
        gt_r = (gt_range >= r_low) & (gt_range < r_high)
        for z_low, z_high in zip(HEIGHT_EDGES[:-1], HEIGHT_EDGES[1:]):
            pred_count = int((pred_r & (points[:, 2] >= z_low) & (points[:, 2] < z_high)).sum().item())
            gt_count = int((gt_r & (gt[:, 2] >= z_low) & (gt[:, 2] < z_high)).sum().item())
            range_height.append({
                "range_low": r_low, "range_high": r_high, "z_low": z_low, "z_high": z_high,
                "pred_count": pred_count, "gt_count": gt_count,
                "pred_fraction": pred_count / max(len(points), 1),
                "gt_fraction": gt_count / max(len(gt), 1),
                "pred_gt_ratio": pred_count / max(gt_count, 1),
            })

    pred_gt_height = {
        "z_gt_2_ratio": int((points[:, 2] > 2.).sum().item()) / max(int((gt[:, 2] > 2.).sum().item()), 1),
        "z_gt_3_ratio": int((points[:, 2] > 3.).sum().item()) / max(int((gt[:, 2] > 3.).sum().item()), 1),
        "height_hist_l1": float(sum(height_abs)),
        "height_hist_tv": float(.5 * sum(height_abs)),
    }
    return {"height_bins": pred_height, "gt_height_bins": gt_height,
            "range_height_bins": range_height, **pred_gt_height}


@torch.no_grad()
def _metrics(points: torch.Tensor, gt: torch.Tensor, source: torch.Tensor, backend: str, threshold: float) -> dict:
    """Geometry, density and GT-to-pred assignment multiplicity metrics."""
    d2_forward, pred_to_gt = knn(points, gt, 1, backend)
    d2_reverse, gt_to_pred = knn(gt, points, 1, backend)
    forward, reverse = d2_forward[:, 0].sqrt(), d2_reverse[:, 0].sqrt()
    precision = (forward <= threshold).float().mean()
    recall = (reverse <= threshold).float().mean()
    fscore = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    # k=2 includes the diagonal self-match; k=9 gives the eighth non-self NN.
    self_d2, _ = knn(points, points, 9, backend)
    self_nn = self_d2[:, 1].sqrt()
    eighth_nn = self_d2[:, 8].sqrt()
    nearest_pred = gt_to_pred[:, 0].long()
    multiplicity = torch.bincount(nearest_pred, minlength=len(points))
    unique_rows = torch.unique(points, dim=0).shape[0]
    displacement = torch.linalg.vector_norm(points - source, dim=1)

    return {
        "chamfer": _scalar((forward.mean() + reverse.mean()) * .5),
        "endpoint_to_gt_mean": _scalar(forward.mean()),
        "gt_to_endpoint_mean": _scalar(reverse.mean()),
        "fscore": _scalar(fscore),
        "coverage": _scalar(recall),
        "nn_target_coverage": float(torch.unique(pred_to_gt[:, 0]).numel() / len(gt)),
        "exact_duplicate_rows": int(len(points) - unique_rows),
        "nn_lt_001_ratio": _scalar((self_nn < .01).float().mean()),
        "nn_lt_002_ratio": _scalar((self_nn < .02).float().mean()),
        "nn_lt_005_ratio": _scalar((self_nn < .05).float().mean()),
        "self_nn_mean": _scalar(self_nn.mean()),
        "self_nn_p01": _scalar(torch.quantile(self_nn, .01)),
        "self_nn_p05": _scalar(torch.quantile(self_nn, .05)),
        "self_nn_p50": _scalar(torch.quantile(self_nn, .50)),
        "self_nn_p95": _scalar(torch.quantile(self_nn, .95)),
        "eighth_nn_mean": _scalar(eighth_nn.mean()),
        "eighth_nn_std": _scalar(eighth_nn.std(unbiased=False)),
        "eighth_nn_cv": _scalar(eighth_nn.std(unbiased=False) / eighth_nn.mean().clamp_min(1e-12)),
        "assignment_count_eq_1_ratio": _scalar((multiplicity == 1).float().mean()),
        "assignment_count_ge_2_ratio": _scalar((multiplicity >= 2).float().mean()),
        "assignment_count_ge_3_ratio": _scalar((multiplicity >= 3).float().mean()),
        "assignment_count_ge_5_ratio": _scalar((multiplicity >= 5).float().mean()),
        "assignment_count_mean": _scalar(multiplicity.float().mean()),
        "assignment_count_p50": _scalar(torch.quantile(multiplicity.float(), .50)),
        "assignment_count_p90": _scalar(torch.quantile(multiplicity.float(), .90)),
        "assignment_count_p95": _scalar(torch.quantile(multiplicity.float(), .95)),
        "assignment_count_max": int(multiplicity.max().item()),
        "transport_mean": _scalar(displacement.mean()),
        "transport_p95": _scalar(torch.quantile(displacement, .95)),
        **_distribution(points, gt),
    }


def _parse_method(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--method must be NAME=CACHE_DIRECTORY")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--method must be NAME=CACHE_DIRECTORY")
    return name, Path(path)


def _mean_summary(records: list[dict]) -> dict:
    scalar_keys = [key for key, value in records[0]["metrics"].items() if isinstance(value, (float, int))]
    return {key: float(np.mean([record["metrics"][key] for record in records])) for key in scalar_keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", type=_parse_method, required=True,
                        help="Method name and cached endpoint directory: NAME=DIR. Repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knn-backend", choices=("keops", "scipy"), default="keops")
    parser.add_argument("--fscore-threshold", type=float, default=.2)
    parser.add_argument("--sinkhorn", action="store_true", help="Evaluate each endpoint after fixed refinement.")
    parser.add_argument("--sinkhorn-k", type=int, default=16)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=.01)
    parser.add_argument("--sinkhorn-iterations", type=int, default=100)
    parser.add_argument("--sinkhorn-alpha", type=float, default=1.0)
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.knn_backend == "keops":
        raise RuntimeError("KeOps endpoint distribution evaluation requires a CUDA-visible GPU.")

    methods = dict(args.method)
    if len(methods) != len(args.method):
        raise ValueError("Each --method name must be unique.")
    frame_sets = {name: {file.stem for file in path.glob("*.pt")} for name, path in methods.items()}
    if not all(frame_sets.values()):
        missing = [name for name, frames in frame_sets.items() if not frames]
        raise ValueError(f"Empty or missing cache directories: {missing}")
    expected_frames = next(iter(frame_sets.values()))
    if any(frames != expected_frames for frames in frame_sets.values()):
        raise ValueError("All method cache directories must contain exactly the same frame stems.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for frame in sorted(expected_frames):
        cached = {name: torch.load(path / f"{frame}.pt", map_location="cpu") for name, path in methods.items()}
        reference = next(iter(cached.values()))
        for name, item in cached.items():
            if not torch.equal(item["p0"], reference["p0"]):
                raise RuntimeError(f"P0 mismatch in frame {frame}: {name} is not bit-identical.")
            if not torch.equal(item["gt"], reference["gt"]):
                raise RuntimeError(f"GT mismatch in frame {frame}: {name} is not bit-identical.")
        p0, gt = reference["p0"].float().to(device), reference["gt"].float().to(device)
        for name, item in cached.items():
            endpoint = item["endpoint"].float().to(device)
            base = {"frame": frame, "method": name,
                    "metrics": _metrics(endpoint, gt, p0, args.knn_backend, args.fscore_threshold)}
            records.append(base)
            if args.sinkhorn:
                refined, diagnostic = refine_endpoint(
                    endpoint, gt, k=args.sinkhorn_k, alpha=args.sinkhorn_alpha,
                    epsilon=args.sinkhorn_epsilon, iterations=args.sinkhorn_iterations,
                    backend=args.knn_backend,
                )
                records.append({
                    "frame": frame, "method": f"{name}_sinkhorn",
                    "metrics": _metrics(refined, gt, p0, args.knn_backend, args.fscore_threshold),
                    "sinkhorn": diagnostic,
                })
        print(f"evaluated {frame}", flush=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    summary = {
        "frames": len(expected_frames),
        "methods": {method: _mean_summary(method_records) for method, method_records in grouped.items()},
        "sinkhorn": ({"k": args.sinkhorn_k, "epsilon": args.sinkhorn_epsilon,
                      "iterations": args.sinkhorn_iterations, "alpha": args.sinkhorn_alpha}
                     if args.sinkhorn else None),
        "p0_identity": "verified bit-identical for every method and frame",
    }
    (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    scalar_columns = ["frame", "method", *sorted({
        key for record in records for key, value in record["metrics"].items() if isinstance(value, (float, int))
    })]
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_columns)
        writer.writeheader()
        for record in records:
            scalar_metrics = {
                key: value for key, value in record["metrics"].items()
                if isinstance(value, (float, int))
            }
            writer.writerow({"frame": record["frame"], "method": record["method"], **scalar_metrics})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

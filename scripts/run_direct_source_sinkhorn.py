#!/usr/bin/env python3
"""Offline P0-to-GT sparse-Sinkhorn diagnostic using cached FPSGen sources.

The script evaluates geometry and local point distribution. A low Chamfer
distance alone is not enough: many source points may share a barycentre.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fpsgen.ops.endpoint_refinement.knn import knn
from fpsgen.ops.endpoint_refinement.sparse_global_ot import refine_endpoint


def _value(tensor):
    return float(tensor.item())


def geometry_metrics(points, gt, source):
    """Return bidirectional geometry, self-NN, duplicate, and transport stats."""
    d2_forward, nearest_gt = knn(points, gt, 1, "keops")
    d2_reverse, _ = knn(gt, points, 1, "keops")
    forward = d2_forward[:, 0].sqrt()
    reverse = d2_reverse[:, 0].sqrt()
    precision = (forward <= 0.2).float().mean()
    recall = (reverse <= 0.2).float().mean()
    fscore = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    # k=2 excludes the diagonal zero distance and yields true self-NN spacing.
    spacing = knn(points, points, 2, "keops")[0][:, 1].sqrt()
    displacement = torch.linalg.vector_norm(points - source, dim=1)
    unique_rows = torch.unique(points, dim=0).shape[0]
    return {
        "chamfer": _value((forward.mean() + reverse.mean()) * 0.5),
        "endpoint_to_gt": _value(forward.mean()),
        "gt_to_endpoint": _value(reverse.mean()),
        "fscore": _value(fscore),
        "coverage": _value(recall),
        "nn_target_coverage": float(torch.unique(nearest_gt[:, 0]).numel() / len(gt)),
        "nn_spacing": _value(spacing.mean()),
        "nn_spacing_min": _value(spacing.min()),
        "nn_spacing_p01": _value(torch.quantile(spacing, 0.01)),
        "nn_spacing_p05": _value(torch.quantile(spacing, 0.05)),
        "nn_spacing_p50": _value(torch.quantile(spacing, 0.50)),
        "nn_spacing_p95": _value(torch.quantile(spacing, 0.95)),
        # Count surplus rows, e.g. [a, a, a] contributes two duplicate rows.
        "exact_duplicate_rows": int(len(points) - unique_rows),
        "nn_lt_001_ratio": _value((spacing < 0.01).float().mean()),
        "nn_lt_002_ratio": _value((spacing < 0.02).float().mean()),
        "nn_lt_005_ratio": _value((spacing < 0.05).float().mean()),
        "transport_mean": _value(displacement.mean()),
        "transport_p50": _value(torch.quantile(displacement, 0.50)),
        "transport_p90": _value(torch.quantile(displacement, 0.90)),
        "transport_p95": _value(torch.quantile(displacement, 0.95)),
        "transport_p99": _value(torch.quantile(displacement, 0.99)),
        "transport_max": _value(displacement.max()),
    }


def write_ply(path, points):
    array = points.detach().cpu().numpy().astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(array)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(array.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--visuals", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, cache_file in enumerate(sorted(args.cache.glob("*.pt"))):
        cached = torch.load(cache_file, map_location="cpu")
        p0 = cached["p0"].float().cuda()
        teacher = cached["endpoint"].float().cuda()
        gt = cached["gt"].float().cuda()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        direct, diagnostics = refine_endpoint(
            p0, gt, k=args.k, alpha=1.0, epsilon=args.epsilon,
            iterations=args.iterations, backend="keops",
        )
        torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - started) * 1e3
        teacher_sinkhorn, _ = refine_endpoint(
            teacher, gt, k=16, alpha=1.0, epsilon=0.01,
            iterations=100, backend="keops",
        )
        rows.append({
            "frame": str(cached["frame"]),
            "p0": geometry_metrics(p0, gt, p0),
            "gt": geometry_metrics(gt, gt, p0),
            "teacher": geometry_metrics(teacher, gt, p0),
            "direct": geometry_metrics(direct, gt, p0),
            "teacher_sinkhorn": geometry_metrics(teacher_sinkhorn, gt, p0),
            "runtime_ms": runtime_ms,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "diagnostics": diagnostics,
        })
        if args.visuals and index in (0, 3, 6, 9):
            base = args.output / "visuals" / str(cached["frame"])
            write_ply(base / "p0.ply", p0)
            write_ply(base / "direct_sinkhorn.ply", direct)
            write_ply(base / "teacher_endpoint.ply", teacher)
            write_ply(base / "teacher_sinkhorn.ply", teacher_sinkhorn)
            write_ply(base / "gt.ply", gt)

    metric_keys = list(rows[0]["p0"])
    methods = ("p0", "gt", "teacher", "direct", "teacher_sinkhorn")
    summary = {
        "config": {"k": args.k, "epsilon": args.epsilon, "iterations": args.iterations, "alpha": 1.0},
        "frames": len(rows),
        "means": {
            method: {key: float(np.mean([row[method][key] for row in rows])) for key in metric_keys}
            for method in methods
        },
        "runtime_ms": {
            "mean": float(np.mean([row["runtime_ms"] for row in rows])),
            "max": float(np.max([row["runtime_ms"] for row in rows])),
        },
        "peak_memory_mb": float(np.max([row["peak_memory_mb"] for row in rows])),
        "direct_better_than_teacher_cd_ratio": float(
            np.mean([row["direct"]["chamfer"] < row["teacher"]["chamfer"] for row in rows])
        ),
    }
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

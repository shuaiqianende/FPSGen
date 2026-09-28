#!/usr/bin/env python3
"""Run Gate B over a manifest, with a frozen Teacher loaded exactly once."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_teacher_endpoint import _source_from_gt, _tensor_field
from fpsgen.ops.endpoint_refinement.knn import knn
from fpsgen.ops.endpoint_refinement.sparse_global_ot import knn_barycentre, refine_endpoint


def _manifest(path):
    return [x.strip() for x in path.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")]


def _sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "p25": float(np.quantile(values, .25)),
            "p75": float(np.quantile(values, .75))}


class FrozenTeacher:
    def __init__(self, checkpoint, device):
        import fpsgen.models.minkunet as network
        state = torch.load(checkpoint, map_location="cpu")["state_dict"]
        self.encoder = network.MinkGlobalEnc(in_channels=3, out_channels=96).to(device).eval()
        self.model = network.MinkUNet_NoTime(in_channels=3, out_channels=96).to(device).eval()
        self.encoder.load_state_dict({k.removeprefix("partial_enc."): v for k, v in state.items()
                                      if k.startswith("partial_enc.")}, strict=True)
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()
                                    if k.startswith("model.")}, strict=True)
        self.device = device

    @torch.no_grad()
    def endpoint(self, gt, resolution):
        source = _source_from_gt(gt, gt.shape[0])
        x_source, x_gt = (_tensor_field(x, resolution, self.device) for x in (source, gt))
        residual = self.model(x_source, x_source.sparse(), self.encoder(x_gt))
        return source, source - residual.reshape_as(source)


@torch.no_grad()
def _geometry(points, target, raw_endpoint, args):
    d2_forward, nearest = knn(points, target, k=1, backend=args.knn_backend)
    d2_reverse, _ = knn(target, points, k=1, backend=args.knn_backend)
    forward = d2_forward[:, 0].sqrt()
    reverse = d2_reverse[:, 0].sqrt()
    precision = (forward <= args.fscore_threshold).float().mean()
    recall = (reverse <= args.fscore_threshold).float().mean()
    fscore = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    spacing_d2, _ = knn(points, points, k=2, backend=args.knn_backend)
    spacing = spacing_d2[:, 1].sqrt()
    displacement = torch.linalg.vector_norm(points - raw_endpoint, dim=1)
    return {
        "chamfer": float((forward.mean() + reverse.mean()).mul(.5).item()),
        "endpoint_to_gt_mean": float(forward.mean().item()),
        "endpoint_to_gt_p95": float(torch.quantile(forward, .95).item()),
        "gt_to_endpoint_mean": float(reverse.mean().item()),
        "gt_to_endpoint_p95": float(torch.quantile(reverse, .95).item()),
        "fscore": float(fscore.item()), "coverage": float(recall.item()),
        "nn_target_coverage": float(torch.unique(nearest[:, 0]).numel() / target.shape[0]),
        "nn_spacing_mean": float(spacing.mean().item()),
        "nn_spacing_p05": float(torch.quantile(spacing, .05).item()),
        "nn_spacing_p50": float(torch.quantile(spacing, .5).item()),
        "refinement_displacement_mean": float(displacement.mean().item()),
        "refinement_displacement_p95": float(torch.quantile(displacement, .95).item()),
    }


def _record(frame, method, alpha, points, source, raw_endpoint, target, diagnostic, elapsed, args):
    record = _geometry(points, target, raw_endpoint, args)
    record.update({
        "frame": frame, "method": method, "alpha": alpha,
        "transport_length": float(torch.linalg.vector_norm(points - source, dim=1).mean().item()),
        "runtime_ms": elapsed * 1e3,
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20),
        "row_error": diagnostic.get("row_error", None), "col_error": diagnostic.get("col_error", None),
        "row_error_mean": diagnostic.get("row_error_mean", None),
        "row_error_max": diagnostic.get("row_error_max", None),
        "col_error_mean": diagnostic.get("col_error_mean", None),
        "col_error_max": diagnostic.get("col_error_max", None),
        "bad_row_ratio": diagnostic.get("bad_row_ratio", None),
        "bad_col_ratio": diagnostic.get("bad_col_ratio", None),
        "mean_row_entropy": diagnostic.get("mean_row_entropy", None),
        "max_row_probability": diagnostic.get("max_row_probability", None),
        "max_marginal_error": max(diagnostic.get("row_error", 0.), diagnostic.get("col_error", 0.)),
        "entropy": diagnostic.get("entropy", None), "num_edges": diagnostic.get("num_edges", None),
        "fallback": False, "fallback_reason": "",
        "active_k": args.k, "seed": args.seed, "commit": _sha(),
    })
    return record


def _summary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    output = {}
    metrics = ("chamfer", "endpoint_to_gt_mean", "gt_to_endpoint_mean", "fscore", "coverage",
               "nn_target_coverage", "nn_spacing_mean", "runtime_ms", "peak_memory_mb")
    raw_by_frame = {x["frame"]: x for x in grouped["raw_teacher"]}
    lower_is_better = {"chamfer", "endpoint_to_gt_mean", "gt_to_endpoint_mean"}
    for method, rows in grouped.items():
        stats = {metric: _distribution([row[metric] for row in rows]) for metric in metrics}
        if method != "raw_teacher":
            comparisons = {}
            for metric in ("chamfer", "endpoint_to_gt_mean", "gt_to_endpoint_mean", "fscore",
                           "coverage", "nn_target_coverage"):
                improved = [((row[metric] < raw_by_frame[row["frame"]][metric])
                             if metric in lower_is_better else
                             (row[metric] > raw_by_frame[row["frame"]][metric])) for row in rows]
                comparisons[f"{metric}_improved_frame_ratio"] = float(np.mean(improved))
            comparisons["spacing_degraded_frame_ratio"] = float(np.mean([
                row["nn_spacing_mean"] < raw_by_frame[row["frame"]]["nn_spacing_mean"] for row in rows
            ]))
            stats["relative_to_raw_teacher"] = comparisons
        output[method] = stats
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", default="08")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/research_v2/sinkhorn_diagnostic/b10"))
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--alpha", type=float, nargs="+", default=[.25, .5, .75, 1.])
    parser.add_argument("--epsilon", type=float, default=.05)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--resolution", type=float, default=.05)
    parser.add_argument("--fscore-threshold", type=float, default=.2)
    parser.add_argument("--knn-backend", choices=("keops",), default="keops")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Gate B requires CUDA; refusing 180k-point CPU execution")
    if not args.alpha or any(alpha <= 0 or alpha > 1 for alpha in args.alpha):
        raise ValueError("Gate B stores raw_teacher separately; alpha must be in (0, 1]")
    stems = _manifest(args.manifest)
    if not stems:
        raise ValueError("Manifest is empty")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = FrozenTeacher(args.teacher_ckpt, device)
    records = []
    fields = None
    for frame_index, stem in enumerate(stems):
        torch.manual_seed(args.seed + frame_index)
        raw = np.load(args.dataset_root / args.sequence / "gt_" / f"{stem}.npy")
        target = torch.from_numpy(raw[:, :3]).float().to(device)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        source, endpoint = teacher.endpoint(target, args.resolution)
        records.append(_record(stem, "raw_teacher", 0., endpoint, source, endpoint, target,
                               {}, time.perf_counter() - start, args))
        for method, alpha, operation in (
            ("nearest_gt", 1., lambda: target[knn(endpoint, target, 1, args.knn_backend)[1][:, 0]]),
            ("knn_barycentric", 1., lambda: knn_barycentre(endpoint, target, k=args.k,
                                                            backend=args.knn_backend)),
        ):
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            points = operation()
            records.append(_record(stem, method, alpha, points, source, endpoint, target,
                                   {}, time.perf_counter() - start, args))
        for alpha in args.alpha:
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            points, diagnostic = refine_endpoint(endpoint, target, k=args.k, alpha=alpha,
                                                  epsilon=args.epsilon, iterations=args.iterations,
                                                  backend=args.knn_backend)
            records.append(_record(stem, f"sinkhorn_a{int(alpha * 100):03d}", alpha, points,
                                   source, endpoint, target, diagnostic,
                                   time.perf_counter() - start, args))
        print(f"completed {stem}")
    fields = list(records[0])
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {"num_frames": len(stems), "manifest": str(args.manifest), "seed": args.seed,
               "k": args.k, "epsilon": args.epsilon, "iterations": args.iterations,
               "methods": _summary(records)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

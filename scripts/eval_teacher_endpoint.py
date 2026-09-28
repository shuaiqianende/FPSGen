#!/usr/bin/env python3
"""Offline sparse-Sinkhorn diagnostic for frozen FPSGen teacher endpoints.

This is intentionally training-free. It writes one CSV row per frame/method
and compares raw teacher endpoints with nearest-GT and unbalanced KNN means
before reporting the sparse balanced-OT ablation. No dense point-pair matrix
is constructed.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from fpsgen.ops.endpoint_refinement.knn import knn
from fpsgen.ops.endpoint_refinement.sparse_global_ot import knn_barycentre, refine_endpoint


FIELDS = ["frame", "method", "alpha", "chamfer", "forward_nn", "reverse_nn",
          "fscore", "transport_length", "nn_spacing", "runtime_ms", "peak_memory_mb",
          "row_error", "col_error", "max_marginal_error", "entropy", "num_edges",
          "knn_backend", "k", "epsilon", "sinkhorn_iterations", "seed", "commit"]


def _sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


@torch.no_grad()
def _metrics(points, target, *, backend, threshold):
    d2_forward, _ = knn(points, target, k=1, backend=backend)
    d2_reverse, _ = knn(target, points, k=1, backend=backend)
    forward = d2_forward.sqrt().mean().item()
    reverse = d2_reverse.sqrt().mean().item()
    precision = (d2_forward.sqrt() <= threshold).float().mean()
    recall = (d2_reverse.sqrt() <= threshold).float().mean()
    fscore = (2 * precision * recall / (precision + recall).clamp_min(1e-12)).item()
    if points.shape[0] < 2:
        spacing = float("nan")
    else:
        d2_spacing, _ = knn(points, points, k=2, backend=backend)
        spacing = d2_spacing[:, 1].sqrt().mean().item()
    return {"chamfer": (forward + reverse) / 2, "forward_nn": forward,
            "reverse_nn": reverse, "fscore": fscore, "nn_spacing": spacing}


def _source_from_gt(gt, target_points):
    """FPSGen's BEV-supported P0 sampler, kept dependency-free for this tool."""
    grid = 256
    pixel = (((gt[:, :2] + 50.) / 100.) * grid).long().clamp(0, grid - 1)
    flat = pixel[:, 0] * grid + pixel[:, 1]
    density = torch.zeros(grid * grid, device=gt.device).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=torch.float32)
    ) + 1e-8
    sampled = torch.multinomial(density, target_points, replacement=True)
    x = ((sampled // grid).float() + .5) / grid * 100. - 50.
    y = ((sampled % grid).float() + .5) / grid * 100. - 50.
    anchors = torch.stack((x, y, torch.zeros_like(x)), dim=-1)
    return anchors + torch.randn_like(anchors)


def _tensor_field(points, resolution, device):
    import MinkowskiEngine as ME
    rows = ME.utils.batched_coordinates([points], dtype=torch.float32, device=device)
    coordinates = rows.clone()
    coordinates[:, 1:] = torch.round(coordinates[:, 1:] / resolution)
    return ME.TensorField(features=rows[:, 1:], coordinates=coordinates,
                          quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                          minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                          device=device)


def _teacher_endpoint(gt, args, device):
    """Run the frozen teacher once; only this path needs ME and PyKeOps."""
    import fpsgen.models.minkunet as network
    checkpoint = torch.load(args.teacher_ckpt, map_location="cpu")
    state = checkpoint["state_dict"]
    encoder = network.MinkGlobalEnc(in_channels=3, out_channels=96).to(device).eval()
    teacher = network.MinkUNet_NoTime(in_channels=3, out_channels=96).to(device).eval()
    encoder.load_state_dict({k.removeprefix("partial_enc."): v for k, v in state.items()
                             if k.startswith("partial_enc.")}, strict=True)
    teacher.load_state_dict({k.removeprefix("model."): v for k, v in state.items()
                             if k.startswith("model.")}, strict=True)
    source = _source_from_gt(gt, gt.shape[0])
    x_source, x_gt = (_tensor_field(x, args.resolution, device) for x in (source, gt))
    with torch.no_grad():
        residual = teacher(x_source, x_source.sparse(), encoder(x_gt))
    return source, source - residual.reshape_as(source)


def _read_endpoint_pair(frame: Path, args, device):
    gt = torch.from_numpy(np.load(frame)[:, :3]).float().to(device)
    if args.endpoint_dir:
        endpoint_path = args.endpoint_dir / frame.name
        if not endpoint_path.is_file():
            raise FileNotFoundError(f"Missing endpoint file: {endpoint_path}")
        endpoint = torch.from_numpy(np.load(endpoint_path)[:, :3]).float().to(device)
        source = endpoint.clone()  # transport is unknown in pre-exported mode.
    else:
        source, endpoint = _teacher_endpoint(gt, args, device)
    if endpoint.shape[0] != gt.shape[0]:
        raise ValueError(f"Endpoint/GT cardinality mismatch for {frame}: {endpoint.shape[0]} vs {gt.shape[0]}")
    return source, endpoint, gt


def _row(frame, method, alpha, points, source, target, elapsed_ms, diagnostic, args):
    result = _metrics(points, target, backend=args.knn_backend, threshold=args.fscore_threshold)
    result.update({
        "frame": frame, "method": method, "alpha": alpha,
        "transport_length": torch.linalg.vector_norm(points - source, dim=1).mean().item(),
        "runtime_ms": elapsed_ms,
        "peak_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                            if torch.cuda.is_available() else ""),
        "row_error": diagnostic.get("row_error", ""),
        "col_error": diagnostic.get("col_error", ""),
        "max_marginal_error": max(diagnostic.get("row_error", 0.), diagnostic.get("col_error", 0.)),
        "entropy": diagnostic.get("entropy", ""), "num_edges": diagnostic.get("num_edges", ""),
        "knn_backend": args.knn_backend, "k": args.k, "epsilon": args.epsilon,
        "sinkhorn_iterations": args.iterations, "seed": args.seed, "commit": _sha(),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", type=Path, required=True,
                        help="GT .npy files; array columns begin with XYZ")
    parser.add_argument("--teacher-ckpt", type=Path,
                        help="Use frozen teacher to generate P0 and endpoint")
    parser.add_argument("--endpoint-dir", type=Path,
                        help="Alternative: endpoint .npy files named like the GT frames")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/research_v2/sinkhorn_diagnostic/results.csv"))
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--alpha", type=float, nargs="+", default=[.25, .5, .75, 1.])
    parser.add_argument("--epsilon", type=float, default=.05)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--resolution", type=float, default=.05)
    parser.add_argument("--fscore-threshold", type=float, default=.2)
    parser.add_argument("--knn-backend", choices=("keops", "scipy"), default="keops")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if bool(args.teacher_ckpt) == bool(args.endpoint_dir):
        raise ValueError("Specify exactly one of --teacher-ckpt or --endpoint-dir")
    if args.k < 1 or not all(0 <= value <= 1 for value in args.alpha):
        raise ValueError("k must be positive and alpha values must be in [0, 1]")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for offset, frame in enumerate(args.frames):
            torch.manual_seed(args.seed + offset)
            source, endpoint, target = _read_endpoint_pair(frame, args, device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            raw_time = time.perf_counter()
            writer.writerow(_row(frame.name, "teacher", 0., endpoint, source, target,
                                 (time.perf_counter() - raw_time) * 1e3, {}, args))
            start = time.perf_counter()
            _, nearest_index = knn(endpoint, target, k=1, backend=args.knn_backend)
            nearest = target[nearest_index[:, 0]]
            writer.writerow(_row(frame.name, "nearest_gt", 1., nearest, source, target,
                                 (time.perf_counter() - start) * 1e3, {}, args))
            start = time.perf_counter()
            bary = knn_barycentre(endpoint, target, k=args.k, backend=args.knn_backend)
            writer.writerow(_row(frame.name, "knn_barycentric", 1., bary, source, target,
                                 (time.perf_counter() - start) * 1e3, {}, args))
            for alpha in args.alpha:
                start = time.perf_counter()
                refined, diagnostic = refine_endpoint(
                    endpoint, target, k=args.k, alpha=alpha, epsilon=args.epsilon,
                    iterations=args.iterations, backend=args.knn_backend,
                )
                writer.writerow(_row(frame.name, "sparse_sinkhorn", alpha, refined,
                                     source, target, (time.perf_counter() - start) * 1e3,
                                     diagnostic, args))
            output.flush()
            print(f"completed {frame}")


if __name__ == "__main__":
    main()

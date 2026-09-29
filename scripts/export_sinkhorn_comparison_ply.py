#!/usr/bin/env python3
"""Export raw and Sparse-Sinkhorn endpoints from cached Teacher tuples as PLY.

The cache files are immutable evaluation tuples produced by
``cache_teacher_endpoints.py``.  This utility deliberately only reads them;
it writes binary little-endian float32 XYZ PLY files for visual comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fpsgen.ops.endpoint_refinement import refine_endpoint


def parse_method(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--method expects NAME=CACHE_DIRECTORY") from error
    return name, Path(path)


def write_ply(path: Path, xyz: torch.Tensor) -> None:
    """Write a binary PLY containing exactly float32 x/y/z fields."""
    points = xyz.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(points.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Sparse Sinkhorn export requires a CUDA-visible GPU.")
    methods = dict(args.method)
    if len(methods) != len(args.method):
        raise ValueError("Method names must be unique.")
    device = torch.device("cuda")

    for frame in args.frames:
        entries = {name: torch.load(path / f"{frame}.pt", map_location="cpu")
                   for name, path in methods.items()}
        reference = next(iter(entries.values()))
        for name, entry in entries.items():
            if not torch.equal(entry["p0"], reference["p0"]):
                raise RuntimeError(f"P0 mismatch for {frame}: {name}")
            if not torch.equal(entry["gt"], reference["gt"]):
                raise RuntimeError(f"GT mismatch for {frame}: {name}")

        frame_out = args.output / frame
        write_ply(frame_out / "gt.ply", reference["gt"])
        write_ply(frame_out / "p0.ply", reference["p0"])
        gt = reference["gt"].float().to(device)
        for name, entry in entries.items():
            endpoint = entry["endpoint"].float().to(device)
            refined, _ = refine_endpoint(
                endpoint, gt, k=args.k, epsilon=args.epsilon,
                iterations=args.iterations, alpha=args.alpha,
            )
            write_ply(frame_out / f"{name}_raw.ply", endpoint)
            write_ply(frame_out / f"{name}_sinkhorn.ply", refined)
        print(f"exported {frame}", flush=True)


if __name__ == "__main__":
    main()

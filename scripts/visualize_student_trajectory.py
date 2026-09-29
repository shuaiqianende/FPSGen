#!/usr/bin/env python3
"""Render a compact BEV/XZ visualization of saved Student PointFlow states."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def read_xyz(path: Path) -> np.ndarray:
    xyz = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid PLY: {path}")
    return xyz


def subsample(xyz: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(xyz) <= count:
        return xyz
    return xyz[np.random.default_rng(seed).choice(len(xyz), size=count, replace=False)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=8000)
    args = parser.parse_args()
    pattern = re.compile(r"step_(\d+)_t([0-9.]+)\.ply")
    states = []
    for path in args.trajectory_dir.glob("step_*_t*.ply"):
        match = pattern.fullmatch(path.name)
        if match:
            states.append((int(match.group(1)), float(match.group(2)), path))
    states.sort()
    if not states:
        raise ValueError("No step_XXX_tY.YY.ply files found")
    panels = [(f"t={time_value:.2f}", read_xyz(path)) for _, time_value, path in states]
    panels.append(("GT Poisson", read_xyz(args.gt)))
    figure, axes = plt.subplots(2, len(panels), figsize=(3.0 * len(panels), 6.0),
                                 constrained_layout=True)
    for index, (title, cloud) in enumerate(panels):
        points = subsample(cloud, args.max_points, seed=20260929 + index)
        top, side = axes[0, index], axes[1, index]
        scatter_top = top.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=.25,
                                  cmap="turbo", vmin=-4, vmax=5.4, linewidths=0, rasterized=True)
        side.scatter(points[:, 0], points[:, 2], c=points[:, 2], s=.25,
                     cmap="turbo", vmin=-4, vmax=5.4, linewidths=0, rasterized=True)
        top.set_title(title)
        top.set_xlim(-50, 50); top.set_ylim(-50, 50); top.set_aspect("equal")
        side.set_xlim(-50, 50); side.set_ylim(-4, 5.4)
        if index == 0:
            top.set_ylabel("Y (m)")
            side.set_ylabel("Z (m)")
        top.set_xticks([]); top.set_yticks([])
        side.set_xlabel("X (m)"); side.set_yticks([-4, 0, 5])
    figure.colorbar(scatter_top, ax=axes.ravel().tolist(), shrink=.72, label="Z (m)")
    figure.suptitle("Student PointFlow trajectory — Oracle-BEV, condition=100, CFG=1", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()

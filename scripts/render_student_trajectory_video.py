#!/usr/bin/env python3
"""Render saved Student PointFlow Euler states to a self-contained MP4 video."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


STEP_PATTERN = re.compile(r"step_(\d+)_t([0-9.]+)\.ply")


def read_xyz(path: Path) -> np.ndarray:
    xyz = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid PLY: {path}")
    return xyz


def configure_top(axis, title: str) -> None:
    axis.set_title(title)
    axis.set_xlim(-50, 50)
    axis.set_ylim(-50, 50)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def configure_side(axis, title: str) -> None:
    axis.set_title(title)
    axis.set_xlim(-50, 50)
    axis.set_ylim(-4, 5.4)
    axis.set_xlabel("X (m)")
    axis.set_yticks([-4, 0, 5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=7000)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    if args.fps < 1 or args.max_points < 1:
        parser.error("--fps and --max-points must be positive")
    states = []
    for path in args.trajectory_dir.glob("step_*_t*.ply"):
        match = STEP_PATTERN.fullmatch(path.name)
        if match:
            states.append((int(match.group(1)), float(match.group(2)), path))
    states.sort()
    if len(states) < 2:
        raise ValueError("At least two saved step_XXX_tY.YY.ply files are required")
    first = read_xyz(states[0][2])
    rng = np.random.default_rng(20260929)
    indices = (np.arange(len(first)) if len(first) <= args.max_points else
               rng.choice(len(first), size=args.max_points, replace=False))
    gt = read_xyz(args.gt)
    gt_indices = (np.arange(len(gt)) if len(gt) <= args.max_points else
                  rng.choice(len(gt), size=args.max_points, replace=False))
    gt = gt[gt_indices]
    frame_dir = args.output.parent / f"{args.output.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    for frame_number, (step, t_value, path) in enumerate(states):
        current = read_xyz(path)
        if len(current) != len(first):
            raise ValueError(f"Point count changed at {path}: {len(current)} vs {len(first)}")
        current = current[indices]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        scatter = axes[0, 0].scatter(current[:, 0], current[:, 1], c=current[:, 2], s=.35,
                                      cmap="turbo", vmin=-4, vmax=5.4, linewidths=0, rasterized=True)
        axes[1, 0].scatter(current[:, 0], current[:, 2], c=current[:, 2], s=.35,
                           cmap="turbo", vmin=-4, vmax=5.4, linewidths=0, rasterized=True)
        axes[0, 1].scatter(gt[:, 0], gt[:, 1], c=gt[:, 2], s=.35,
                           cmap="turbo", vmin=-4, vmax=5.4, linewidths=0, rasterized=True)
        axes[1, 1].scatter(gt[:, 0], gt[:, 2], c=gt[:, 2], s=.35,
                           cmap="turbo", vmin=-4, vmax=5.4, linewidths=0, rasterized=True)
        configure_top(axes[0, 0], f"Student BEV, t={t_value:.2f} (step {step})")
        configure_side(axes[1, 0], "Student X–Z")
        configure_top(axes[0, 1], "GT Poisson BEV (reference)")
        configure_side(axes[1, 1], "GT Poisson X–Z (reference)")
        axes[0, 0].set_ylabel("Y (m)")
        axes[1, 0].set_ylabel("Z (m)")
        figure.colorbar(scatter, ax=axes.ravel().tolist(), shrink=.83, label="Z (m)")
        figure.suptitle("Student PointFlow — Oracle-BEV, LiDAR-only condition=100, CFG=1", fontsize=14)
        figure.savefig(frame_dir / f"frame_{frame_number:03d}.png", dpi=150)
        plt.close(figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", str(frame_dir / "frame_%03d.png"), "-vcodec", "libx264",
        "-pix_fmt", "yuv420p", str(args.output),
    ], check=True)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    print(f"[OK] wrote {args.output} ({len(states)} Euler states at {args.fps} fps)")


if __name__ == "__main__":
    main()

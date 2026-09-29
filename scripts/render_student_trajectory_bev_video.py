#!/usr/bin/env python3
"""Create a presentation-quality top-down PointFlow trajectory video.

Only the BEV view is rendered. Height is encoded by a continuous blue (low) to
red (high) map; the right panel is a fixed GT-Poisson reference so the movement
of the left Student panel remains interpretable at every Euler state.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np
import open3d as o3d


PATTERN = re.compile(r"step_(\d+)_t([0-9.]+)\.ply")
# LiDiff's public qualitative figure is deliberately minimal: a bright canvas,
# coloured points, and no intrusive 3D axes. Keep that presentation grammar
# while retaining our metrically fixed camera/crop underneath.
BACKGROUND = "#ffffff"
PANEL = "#ffffff"
GRID = "#9aa7b5"
TEXT = "#172033"
# Deliberately avoid a white midpoint: most driving-scene points sit near the
# ground plane, and a diverging white-centred map makes their height unreadable.
HEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "fpsgen_height", ["#1238c8", "#007bff", "#00b9d8", "#ffb000", "#f04a23", "#a90026"], N=256
)


def read_xyz(path: Path) -> np.ndarray:
    xyz = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid PLY: {path}")
    return xyz


def decorate(axis: plt.Axes, title: str) -> None:
    axis.set_facecolor(PANEL)
    axis.set_title(title, color=TEXT, fontsize=17, fontweight="semibold", pad=16)
    axis.set_xlim(-50, 50)
    axis.set_ylim(-50, 50)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for radius in (10, 20, 30, 40, 50):
        axis.add_patch(Circle((0, 0), radius, fill=False, edgecolor=GRID,
                              linewidth=.65, alpha=.22, zorder=0))
    axis.axhline(0, color=GRID, alpha=.16, linewidth=.6, zorder=0)
    axis.axvline(0, color=GRID, alpha=.16, linewidth=.6, zorder=0)
    axis.scatter([0], [0], marker="+", s=80, color=TEXT, linewidths=1.2, zorder=3)
    axis.text(-48, -47, "50 m", color=TEXT, fontsize=9, alpha=.7)


def oblique_crop_bounds(reference: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Use a fixed, robust GT crop for every frame of the trajectory video."""
    lower = np.percentile(reference, .1, axis=0)
    upper = np.percentile(reference, 99.9, axis=0)
    padding = np.array([2.0, 2.0, .35], dtype=np.float32)
    lower -= padding
    upper += padding
    # Keep the usual 50 m BEV domain, while discarding only empty 3D margins.
    lower[:2] = np.maximum(lower[:2], -50.0)
    upper[:2] = np.minimum(upper[:2], 50.0)
    lower[2] = max(float(lower[2]), -4.0)
    upper[2] = min(float(upper[2]), 4.0)
    return ((float(lower[0]), float(upper[0])),
            (float(lower[1]), float(upper[1])),
            (float(lower[2]), float(upper[2])))


def decorate_oblique(axis: plt.Axes, title: str,
                     xlim: tuple[float, float], ylim: tuple[float, float],
                     zlim: tuple[float, float]) -> None:
    """Style a shallow 3D camera for readable point-cloud structure."""
    axis.set_facecolor(PANEL)
    axis.set_title(title, color=TEXT, fontsize=15, fontweight="semibold", pad=13)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_zlim(*zlim)
    # Physical world ratio: the displayed cuboid is 100 m × 100 m × 8 m.
    # Do not stretch Z for visual effect; Student and GT must be geometrically
    # interpretable in the same real-world coordinate system.
    axis.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    # A higher (but still oblique) camera uses the panel efficiently for a
    # physically shallow 100 m × 100 m × ~8 m driving scene.
    axis.view_init(elev=42, azim=-58)
    # Zoom the camera optically, rather than stretching coordinate axes. This
    # keeps X/Y/Z in real metres while using the panel more efficiently.
    try:
        axis.set_proj_type("persp", focal_length=1.75)
    except TypeError:  # pragma: no cover - compatibility with older Matplotlib
        axis.set_proj_type("persp")
    # LiDiff-style clean qualitative panel: camera/crop remain metric, but the
    # wireframe, ticks and pane decoration are intentionally not rendered.
    axis.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--height-min", type=float, default=-2.8,
                        help="Lower display height in metres; lower values saturate blue")
    parser.add_argument("--height-max", type=float, default=3.0,
                        help="Upper display height in metres; higher values saturate red")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    if args.fps < 1 or args.max_points < 1 or args.point_size <= 0:
        parser.error("--fps, --max-points and --point-size must be positive")
    if args.height_min >= args.height_max:
        parser.error("--height-min must be less than --height-max")
    states = []
    for path in args.trajectory_dir.glob("step_*_t*.ply"):
        match = PATTERN.fullmatch(path.name)
        if match:
            states.append((int(match.group(1)), float(match.group(2)), path))
    states.sort()
    if len(states) < 2:
        raise ValueError("At least two saved trajectory states are required")
    initial = read_xyz(states[0][2])
    rng = np.random.default_rng(20260930)
    point_indices = (np.arange(len(initial)) if len(initial) <= args.max_points else
                     rng.choice(len(initial), size=args.max_points, replace=False))
    gt = read_xyz(args.gt)
    gt_indices = (np.arange(len(gt)) if len(gt) <= args.max_points else
                  rng.choice(len(gt), size=args.max_points, replace=False))
    gt = gt[gt_indices]
    xlim, ylim, zlim = oblique_crop_bounds(gt)
    frame_dir = args.output.parent / f"{args.output.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    for frame_index, (step, time_value, path) in enumerate(states):
        current = read_xyz(path)
        if len(current) != len(initial):
            raise ValueError(f"Point count changed at {path}")
        current = current[point_indices]
        figure = plt.figure(figsize=(21, 7.2), facecolor=BACKGROUND)
        grid = figure.add_gridspec(1, 3, wspace=.06)
        axes = [figure.add_subplot(grid[0, 0]),
                figure.add_subplot(grid[0, 1], projection="3d"),
                figure.add_subplot(grid[0, 2], projection="3d")]
        for axis in axes:
            for spine in axis.spines.values():
                spine.set_color("#c6d0da")
                spine.set_linewidth(.8)
        scatter = axes[0].scatter(current[:, 0], current[:, 1], c=current[:, 2], s=args.point_size,
                                  cmap=HEIGHT_CMAP, vmin=args.height_min, vmax=args.height_max, alpha=.94,
                                  linewidths=0, rasterized=True)
        axes[1].scatter(current[:, 0], current[:, 1], current[:, 2], c=current[:, 2],
                        s=args.point_size * .9, cmap=HEIGHT_CMAP,
                        vmin=args.height_min, vmax=args.height_max, alpha=.94,
                        linewidths=0, depthshade=False, rasterized=True)
        axes[2].scatter(gt[:, 0], gt[:, 1], gt[:, 2], c=gt[:, 2],
                        s=args.point_size * .9, cmap=HEIGHT_CMAP,
                        vmin=args.height_min, vmax=args.height_max, alpha=.94,
                        linewidths=0, depthshade=False, rasterized=True)
        decorate(axes[0], f"STUDENT POINTFLOW   •   t = {time_value:.2f}   •   step {step:02d}")
        decorate_oblique(axes[1], "STUDENT POINTFLOW   •   OBLIQUE VIEW", xlim, ylim, zlim)
        decorate_oblique(axes[2], "GT POISSON   •   OBLIQUE REFERENCE", xlim, ylim, zlim)
        # Put the legend outside every panel: 3D axes do not shrink reliably
        # when Matplotlib's automatic colorbar layout is used.
        colorbar_axis = figure.add_axes([.956, .14, .012, .70])
        colorbar = figure.colorbar(scatter, cax=colorbar_axis)
        colorbar.ax.tick_params(colors=TEXT, labelsize=10)
        colorbar.set_label("HEIGHT (m)  •  low (blue) → high (red)", color=TEXT, fontsize=11, labelpad=12)
        figure.subplots_adjust(left=.02, right=.945, top=.89, bottom=.055)
        figure.savefig(frame_dir / f"frame_{frame_index:03d}.png", dpi=120,
                       facecolor=BACKGROUND)
        plt.close(figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The legacy conda environment can shadow the system ffmpeg with a build
    # that lacks libx264. Prefer the server binary when it is available.
    ffmpeg = "/usr/local/bin/ffmpeg" if Path("/usr/local/bin/ffmpeg").is_file() else "ffmpeg"
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", str(frame_dir / "frame_%03d.png"), "-vcodec", "libx264",
        "-pix_fmt", "yuv420p", str(args.output),
    ], check=True)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    print(f"[OK] wrote {args.output} ({len(states)} states at {args.fps} fps)")


if __name__ == "__main__":
    main()

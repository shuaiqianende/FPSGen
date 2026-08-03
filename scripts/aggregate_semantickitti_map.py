#!/usr/bin/env python3
"""Aggregate SemanticKITTI map chunks into a reusable ``map_clean.npy``.

Map chunks are expected to be stored in sequence directories and named
``map_clean_*.pcd`` (``.ply`` and ``.npy`` are also accepted).  When a PCD/PLY
contains semantic labels encoded in the red colour channel, the labels are
decoded and saved as the fourth column of the NumPy output.  The resulting
array is ``[x, y, z, semantic_label]`` in the global/world coordinate frame.

The default NumPy voxel backend is deterministic and works on CPU.  The
optional MinkowskiEngine backend is useful for very large maps when the FPSGen
training environment is already installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _natural_key(path: Path) -> tuple:
    """Sort chunk names numerically while retaining a stable lexical fallback."""
    import re

    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", path.name))


def _read_chunk(path: Path) -> np.ndarray:
    """Load one chunk as float32 ``[x, y, z, label]`` rows."""
    if path.suffix.lower() == ".npy":
        array = np.asarray(np.load(path), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Expected [N,3+] array in {path}, got {array.shape}")
        xyz = array[:, :3]
        labels = array[:, 3] if array.shape[1] >= 4 else np.zeros(len(array), dtype=np.float32)
    elif path.suffix.lower() in {".pcd", ".ply"}:
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(cloud.points, dtype=np.float32)
        if len(cloud.colors) == len(xyz) and len(xyz):
            # LiDiff's map exporter stores the integer semantic ID as R/255.
            labels = np.rint(np.asarray(cloud.colors)[:, 0] * 255.0).astype(np.float32)
        else:
            labels = np.zeros(len(xyz), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported map chunk format: {path}")

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected XYZ points in {path}, got {xyz.shape}")
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(labels)
    return np.column_stack((xyz[finite], labels[finite])).astype(np.float32, copy=False)


def _voxel_first_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep the first point in each voxel using a deterministic NumPy hash."""
    coordinates = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, indices = np.unique(coordinates, axis=0, return_index=True)
    return points[np.sort(indices)]


def _voxel_first_minkowski(points: np.ndarray, voxel_size: float, device: str) -> np.ndarray:
    """Use MinkowskiEngine sparse quantization and return selected row indices."""
    import torch
    import MinkowskiEngine as ME

    coordinates = torch.from_numpy(points[:, :3]).to(device=device) / float(voxel_size)
    _, indices = ME.utils.sparse_quantize(coordinates=coordinates, return_index=True)
    return points[indices.detach().cpu().numpy()]


def voxel_first(points: np.ndarray, voxel_size: float, backend: str, device: str) -> tuple[np.ndarray, str]:
    """Voxel-deduplicate points, falling back to NumPy when requested."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if backend in {"auto", "minkowski"}:
        try:
            return _voxel_first_minkowski(points, voxel_size, device), "minkowski"
        except (ImportError, ModuleNotFoundError, RuntimeError) as error:
            if backend == "minkowski":
                raise RuntimeError(
                    "MinkowskiEngine quantization failed; use --backend numpy or install "
                    "the matching MinkowskiEngine build"
                ) from error
            print(f"MinkowskiEngine unavailable ({error}); using NumPy quantization.")
    return _voxel_first_numpy(points, voxel_size), "numpy"


def _write_visualization(path: Path, points: np.ndarray) -> None:
    """Write an optional Open3D visualization with labels encoded as red."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = np.zeros((len(points), 3), dtype=np.float64)
    colors[:, 0] = np.clip(points[:, 3], 0, 255) / 255.0
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f"Open3D failed to write {path}")


def aggregate_sequence(
    sequence_dir: Path,
    voxel_size: float,
    pattern: str,
    backend: str,
    device: str,
    write_pcd: bool,
) -> dict:
    """Aggregate all map chunks in one sequence and return a JSON summary."""
    chunks = sorted(
        (path for path in sequence_dir.glob(pattern)
         if path.suffix.lower() in {".pcd", ".ply", ".npy"} and path.stem != "map_clean"),
        key=_natural_key,
    )
    if not chunks:
        raise FileNotFoundError(
            f"No map chunks matching {pattern!r} were found in {sequence_dir}"
        )

    rows = [_read_chunk(path) for path in chunks]
    rows = [array for array in rows if len(array)]
    if not rows:
        raise ValueError(f"All map chunks are empty in {sequence_dir}")
    points = np.concatenate(rows, axis=0)
    clean, selected_backend = voxel_first(points, voxel_size, backend, device)
    clean = np.asarray(clean, dtype=np.float32)

    npy_path = sequence_dir / "map_clean.npy"
    np.save(npy_path, clean)
    pcd_path = sequence_dir / "map_clean.pcd"
    if write_pcd:
        _write_visualization(pcd_path, clean)

    summary = {
        "sequence": sequence_dir.name,
        "chunk_count": len(chunks),
        "input_points": int(len(points)),
        "output_points": int(len(clean)),
        "voxel_size_m": float(voxel_size),
        "backend": selected_backend,
        "output_npy": str(npy_path),
        "output_pcd": str(pcd_path) if write_pcd else None,
    }
    with (sequence_dir / "map_clean.meta.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def _resolve_sequences(root: Path, specification: str) -> list[str]:
    if specification.lower() == "all":
        return sorted(path.name for path in root.iterdir()
                      if path.is_dir() and path.name.isdigit())
    return [item.strip() for item in specification.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate map_clean_*.pcd chunks into map_clean.npy per sequence."
    )
    parser.add_argument("--data-root", required=True, type=Path,
                        help="SemanticKITTI/KITTI odometry root containing sequence folders")
    parser.add_argument("--sequences", default="all",
                        help="comma-separated sequence IDs, or 'all' (default: all numeric folders)")
    parser.add_argument("--pattern", default="map_clean_*",
                        help="chunk filename glob (default: map_clean_*)")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="world-map voxel size in metres (default: 0.1)")
    parser.add_argument("--backend", choices=["auto", "numpy", "minkowski"], default="auto")
    parser.add_argument("--device", default="cuda",
                        help="MinkowskiEngine device, e.g. cuda or cpu (default: cuda)")
    parser.add_argument("--no-pcd", action="store_true",
                        help="only write map_clean.npy and metadata, without map_clean.pcd")
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    summaries = []
    for sequence in _resolve_sequences(root, args.sequences):
        sequence_dir = root / sequence
        if not sequence_dir.is_dir():
            print(f"Skipping missing sequence: {sequence_dir}")
            continue
        summaries.append(aggregate_sequence(
            sequence_dir, args.voxel_size, args.pattern, args.backend,
            args.device, not args.no_pcd,
        ))
    if not summaries:
        raise RuntimeError("No sequence was processed")


if __name__ == "__main__":
    main()

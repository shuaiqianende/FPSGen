"""Numeric frame pairing and strict GT audit without semantic/camera inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .voxel_spec import VoxelGridSpec

TRAIN_SEQUENCES = tuple(f"{i:02d}" for i in (*range(8), 9, 10))


def load_xyz(path):
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
        if value.ndim != 2 or value.shape[1] < 3:
            raise ValueError(f"Invalid point array {path}: {value.shape}")
        value = value[:, :3]  # Fourth input_ column is a semantic label. Never read it.
    elif path.suffix == ".ply":
        from plyfile import PlyData
        vertex = PlyData.read(path)["vertex"].data
        value = np.stack([vertex[name] for name in ("x", "y", "z")], -1)
    else:
        raise ValueError(f"Unsupported XYZ format: {path}")
    value = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite XYZ in {path}")
    return torch.from_numpy(value)


def paired_frames(root, sequences=TRAIN_SEQUENCES):
    frames = []
    for sequence in sequences:
        directory = Path(root) / sequence
        gt_files = sorted((directory / "gt_possion").glob("*.ply"), key=lambda p: int(p.stem))
        if not gt_files:
            raise FileNotFoundError(f"Missing gt_possion PLYs: {directory}")
        if list((directory / "gt_possion").glob("*.npy")):
            raise ValueError(f"Mixed GT formats: {directory}")
        partial_by_frame = {}
        for partial in (directory / "input_").glob("*.npy"):
            index = int(partial.stem)
            if index in partial_by_frame:
                raise ValueError(f"Duplicate frame {sequence}/{index}")
            partial_by_frame[index] = partial
        seen = set()
        for gt in gt_files:
            frame = int(gt.stem)
            if frame in seen:
                raise ValueError(f"Duplicate GT frame {sequence}/{frame}")
            seen.add(frame)
            if frame not in partial_by_frame:
                raise FileNotFoundError(f"Missing paired input for {gt}")
            frames.append({"sequence": sequence, "frame": frame,
                           "input": str(partial_by_frame[frame].resolve()), "gt": str(gt.resolve())})
    return frames


def development_split(frames, *, block_size=100, guard_frames=20, development_block=5):
    """One fixed contiguous block per training sequence; remove neighboring guards."""
    train, development, excluded = [], [], []
    lower, upper = development_block * block_size, (development_block + 1) * block_size
    for item in frames:
        if item["sequence"] == "08":
            raise ValueError("seq08 cannot enter development/training manifests")
        frame = item["frame"]
        destination = development if lower <= frame < upper else (
            excluded if lower - guard_frames <= frame < upper + guard_frames else train)
        destination.append(item)
    if not development:
        raise ValueError("No development frames in registered block")
    return {"train": train, "development": development, "excluded_guard": excluded,
            "policy": {"block_size": block_size, "guard_frames": guard_frames,
                       "development_block": development_block}}


def manifest_sha(frames):
    return hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()


class VoxelPointDataset(Dataset):
    def __init__(self, frames, spec=None):
        self.frames = list(frames)
        self.spec = spec or VoxelGridSpec()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        item = self.frames[index]
        try:
            partial, gt = load_xyz(item["input"]), load_xyz(item["gt"])
            counts = self.spec.counts(gt)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Rejected scene {item['sequence']}/{item['frame']:06d}: {exc}") from exc
        return {"input_xyz": partial, "gt_xyz": gt, "gt_counts": counts, "frame": item}


def density_statistics(counts, spec):
    occupied = counts[counts > 0].float()
    return {"points": int(counts.sum()), "nonempty_voxel_count": len(occupied),
            "occupancy_ratio": len(occupied) / int(spec.valid_circle_mask().sum()),
            "maximum_count": int(occupied.max()) if occupied.numel() else 0,
            "occupied_quantiles": dict(zip(("p50", "p90", "p99"),
                torch.quantile(occupied, torch.tensor([.5, .9, .99])).tolist())) if occupied.numel() else {},
            "voxel_count_histogram": torch.bincount(counts.flatten()).tolist()}

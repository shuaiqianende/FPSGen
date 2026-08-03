"""FPSGen generation evaluation for SemanticKITTI and KITTI-360."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime

import click
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

SUPPORTED_DATASETS = ("SemanticKITTI", "KITTI360")


@dataclass(frozen=True)
class FrameRecord:
    """One evaluation frame and its dataset-specific point-cloud paths."""
    scan_path: str
    gt_path: str
    pose: np.ndarray
    frame_id: int


def _parse_kitti_calibration(path: str) -> np.ndarray:
    """Read the LiDAR-to-camera transform from a SemanticKITTI calibration file."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key, values = line.split(":", 1)
            if key == "Tr":
                transform = np.eye(4)
                transform[:3, :4] = np.fromstring(values, sep=" ").reshape(3, 4)
                return transform
    raise ValueError(f"Calibration transform Tr is missing from {path}")


def _load_semantickitti_poses(sequence_dir: str) -> list[np.ndarray]:
    """Load SemanticKITTI poses in the LiDAR coordinate frame."""
    transform = _parse_kitti_calibration(os.path.join(sequence_dir, "calib.txt"))
    inverse_transform = np.linalg.inv(transform)
    poses: list[np.ndarray] = []
    with open(os.path.join(sequence_dir, "poses.txt"), encoding="utf-8") as handle:
        for line in handle:
            values = np.fromstring(line, sep=" ")
            if values.size != 12:
                continue
            pose = np.eye(4)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(inverse_transform @ pose @ transform)
    return poses


def _load_kitti360_poses(sequence_dir: str) -> dict[int, np.ndarray]:
    """Load KITTI-360 frame-indexed poses when a pose file is available."""
    poses_path = os.path.join(sequence_dir, "poses.txt")
    poses: dict[int, np.ndarray] = {}
    if not os.path.exists(poses_path):
        return poses
    with open(poses_path, encoding="utf-8") as handle:
        for line in handle:
            values = np.fromstring(line, sep=" ")
            if values.size != 13:
                continue
            pose = np.eye(4)
            pose[:3, :4] = values[1:].reshape(3, 4)
            poses[int(values[0])] = pose
    return poses


def load_dataset_records(dataset: str, root: str, sequences: str) -> list[FrameRecord]:
    """Load evaluation frames and point-cloud ground truth for supported datasets."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}")

    records: list[FrameRecord] = []
    for sequence in (item.strip() for item in sequences.split(",")):
        if not sequence:
            continue
        sequence_dir = os.path.join(root, sequence)
        if dataset == "SemanticKITTI":
            poses = _load_semantickitti_poses(sequence_dir)
            scan_dir = os.path.join(sequence_dir, "velodyne")
            # Generation evaluation uses the fixed-cardinality target prepared
            # for FPSGen (180k XYZ points per frame).  ``gt_pcd`` is retained
            # as a compatibility fallback for older datasets that do not have
            # the NumPy preparation directory.
            prepared_gt_dirs = (
                os.path.join(sequence_dir, "gt"),
                os.path.join(sequence_dir, "gt_"),
            )
            legacy_gt_dir = os.path.join(sequence_dir, "gt_pcd")
            scans = [name for name in natsorted(os.listdir(scan_dir)) if name.endswith(".bin")]
            for name in scans:
                stem = os.path.splitext(name)[0]
                frame_id = int(stem)
                if frame_id >= len(poses):
                    # A sequence may contain scans beyond the available pose
                    # table; skip those frames instead of silently pairing a
                    # scan with the wrong pose by directory order.
                    continue
                prepared_candidates = [
                    os.path.join(directory, f"{stem}.npy")
                    for directory in prepared_gt_dirs
                ]
                prepared_gt = next(
                    (candidate for candidate in prepared_candidates if os.path.exists(candidate)),
                    None,
                )
                gt_path = prepared_gt if prepared_gt else os.path.join(legacy_gt_dir, f"{stem}.pcd")
                records.append(FrameRecord(
                    os.path.join(scan_dir, name), gt_path, poses[frame_id], frame_id
                ))
        else:
            poses = _load_kitti360_poses(sequence_dir)
            scan_dir, gt_dir = os.path.join(sequence_dir, "input"), os.path.join(sequence_dir, "gt")
            scans = {os.path.splitext(name)[0]: name for name in natsorted(os.listdir(scan_dir))
                     if os.path.splitext(name)[1].lower() in {".pcd", ".ply", ".npy"}}
            gts = {os.path.splitext(name)[0]: name for name in natsorted(os.listdir(gt_dir))
                   if os.path.splitext(name)[1].lower() in {".pcd", ".ply", ".npy"}}
            for stem in natsorted(set(scans) & set(gts)):
                try:
                    frame_id = int(stem)
                except ValueError:
                    continue
                records.append(FrameRecord(
                    os.path.join(scan_dir, scans[stem]), os.path.join(gt_dir, gts[stem]),
                    poses.get(frame_id, np.eye(4)), frame_id
                ))
    if not records:
        raise RuntimeError("No evaluation frames were found for the requested dataset and sequences")
    return records


def select_records(records: list[FrameRecord], mode: str, interval_frames: int, interval_m: float) -> list[FrameRecord]:
    """Subsample a temporal sequence by fixed frame stride or traveled distance."""
    if mode == "frame":
        if interval_frames <= 0:
            raise ValueError("interval_frames must be positive")
        return records[::interval_frames]
    if mode != "distance":
        raise ValueError("select_mode must be 'frame' or 'distance'")
    if interval_m <= 0:
        raise ValueError("interval_m must be positive")
    selected, next_distance, distance = [records[0]], interval_m, 0.0
    for previous, current in zip(records, records[1:]):
        distance += float(np.linalg.norm(current.pose[:3, 3] - previous.pose[:3, 3]))
        if distance >= next_distance:
            selected.append(current)
            next_distance += interval_m
    return selected


def read_points(path: str) -> np.ndarray:
    """Read XYZ coordinates from the point-cloud formats supported by FPSGen."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".bin":
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)[:, :3]
    if suffix in {".pcd", ".ply"}:
        import open3d as o3d

        return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float32)
    raise ValueError(f"Unsupported point-cloud file: {path}")


def load_fpsgen_input(record: FrameRecord, dataset: str, max_range: float) -> np.ndarray:
    """Load and range-filter the partial observation supplied to FPSGen."""
    if dataset == "SemanticKITTI":
        sequence_dir = os.path.dirname(os.path.dirname(record.scan_path))
        prepared_candidates = (
            os.path.join(sequence_dir, "input", f"{record.frame_id:06d}.npy"),
            os.path.join(sequence_dir, "input_", f"{record.frame_id:06d}.npy"),
        )
        prepared = next((candidate for candidate in prepared_candidates if os.path.exists(candidate)), None)
        points = read_points(prepared) if prepared else read_points(record.scan_path)
        distance = np.linalg.norm(points, axis=1)
        points = points[(distance > 3.5) & (distance < max_range) & (points[:, 2] > -4.0)]
        return points
    return read_points(record.scan_path)


def load_raw_fpsgen_input(record: FrameRecord, dataset: str, max_range: float) -> np.ndarray:
    """Load the raw scan used by the original trajectory evaluator.

    Generation evaluation intentionally uses the prepared ``input/*.npy``
    condition.  The pose-cropped trajectory protocol, however, feeds the raw
    SemanticKITTI scan through the same range filter before FPSGen's internal
    farthest-point reduction.  Keeping this path explicit prevents the two
    evaluation protocols from silently sharing different inputs.
    """
    points = read_points(record.scan_path)
    if dataset == "SemanticKITTI":
        distance = np.linalg.norm(points, axis=1)
        points = points[(distance > 3.5) & (distance < max_range) & (points[:, 2] > -4.0)]
    return points


def load_fpsgen_completion_class():
    """Import inference lazily so dataset inspection does not require CUDA extensions."""
    return getattr(importlib.import_module("fpsgen.inference"), "DiffCompletion")


def complete_fpsgen(
    model,
    record: FrameRecord,
    dataset: str,
    max_range: float,
    steps: list[int],
    cond_mode: str,
    use_prepared_input: bool = True,
):
    """Run one FPSGen completion and return input, ground truth, and prediction."""
    import open3d as o3d

    input_loader = load_fpsgen_input if use_prepared_input else load_raw_fpsgen_input
    input_points = input_loader(record, dataset, max_range)
    ground_truth_points = read_points(record.gt_path)
    ground_truth = o3d.geometry.PointCloud()
    ground_truth.points = o3d.utility.Vector3dVector(ground_truth_points)
    coarse, fine = model.complete_scan([input_points, ground_truth, record.scan_path], cond_mode, dataset, steps=steps)
    return input_points, ground_truth_points, np.asarray(coarse, dtype=np.float32), fine


def _resample(point_clouds: list[np.ndarray], count: int, seed: int) -> torch.Tensor:
    """Create equal-size CUDA batches for set-level distribution metrics."""
    rng = np.random.default_rng(seed)
    samples = []
    for points in point_clouds:
        indices = rng.choice(len(points), size=count, replace=len(points) < count)
        samples.append(points[indices])
    return torch.from_numpy(np.stack(samples)).cuda().float()


def evaluate_distribution(predictions: list[np.ndarray], references: list[np.ndarray], num_points: int, batch_size: int, max_range: float) -> dict:
    """Compute EMD/CD, distribution metrics, and JSD for generated point clouds."""
    # The structural-loss extension is required only for distribution metrics,
    # not for dataset inspection or CLI argument parsing.
    from fpsgen.utils.metrics_gen.evaluation_metrics import EMD_CD, compute_all_metrics, jsd_between_point_cloud_sets

    predicted = _resample(predictions, num_points, 42)
    reference = _resample(references, num_points, 43)
    emd_cd = EMD_CD(predicted, reference, batch_size=batch_size, accelerated_cd=True, dcd_alpha=1)
    all_metrics = compute_all_metrics(predicted, reference, batch_size=batch_size, accelerated_cd=True, dcd_alpha=1)
    jsd = jsd_between_point_cloud_sets(
        np.clip(predicted.cpu().numpy() / (2.0 * max_range), -0.5, 0.5),
        np.clip(reference.cpu().numpy() / (2.0 * max_range), -0.5, 0.5),
    )
    return {"emd_cd": _json_value(emd_cd), "all_metrics": _json_value(all_metrics), "jsd": float(jsd)}


def _json_value(value):
    """Convert tensors and NumPy scalars recursively into JSON-compatible values."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@click.command()
@click.option("--path", required=True, type=click.Path(exists=True, file_okay=False), help="dataset root")
@click.option("--dataset", type=click.Choice(SUPPORTED_DATASETS), default="SemanticKITTI")
@click.option("--sequences", default="08", help="comma-separated sequence identifiers")
@click.option("--point-ckpt", required=True, type=click.Path(exists=True),
              help="Stage-3 Point Flow checkpoint")
@click.option("--bev-ckpt", required=True, type=click.Path(exists=True),
              help="Stage-1 BEV Flow checkpoint")
@click.option("--refine", default="", type=click.Path(), help="optional refinement checkpoint")
@click.option("--img-steps", default=50, show_default=True)
@click.option("--point-steps", default=16, show_default=True)
@click.option("--cond-weight", default=2.0, show_default=True,
              help="classifier-free guidance scale; 2 matches the reference inference")
@click.option("--cond-mode", default="100", show_default=True)
@click.option("--max-range", default=50.0, show_default=True)
@click.option("--select-mode", type=click.Choice(["frame", "distance"]), default="frame")
@click.option("--interval-frames", default=20, show_default=True)
@click.option("--interval-m", default=20.0, show_default=True)
@click.option("--max-frames", default=0, show_default=True)
@click.option("--num-points", default=8192, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--save-pcd/--no-save-pcd", default=False)
def main(path, dataset, sequences, point_ckpt, bev_ckpt, refine, img_steps, point_steps, cond_weight, cond_mode, max_range,
         select_mode, interval_frames, interval_m, max_frames, num_points, batch_size, save_pcd):
    import open3d as o3d

    records = select_records(load_dataset_records(dataset, path, sequences), select_mode, interval_frames, interval_m)
    if max_frames:
        records = records[:max_frames]
    model = load_fpsgen_completion_class()(point_ckpt, bev_ckpt, refine, point_steps, cond_weight)
    predictions, references = [], []
    output_dir = os.environ.get("FPSGEN_OUTPUT_DIR", "outputs")
    if save_pcd:
        os.makedirs(output_dir, exist_ok=True)
    for index, record in enumerate(tqdm(records, desc="FPSGen evaluation")):
        _, ground_truth, coarse, _ = complete_fpsgen(model, record, dataset, max_range, [img_steps, point_steps], cond_mode)
        predictions.append(coarse)
        references.append(ground_truth)
        if save_pcd:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(coarse)
            o3d.io.write_point_cloud(os.path.join(output_dir, f"{dataset}_{record.frame_id:06d}_fpsgen.pcd"), cloud)
    result = {
        "dataset": dataset, "sequences": sequences, "frame_count": len(records),
        "metrics": evaluate_distribution(predictions, references, num_points, batch_size, max_range),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fpsgen_generation_{dataset}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    click.echo(json.dumps(result, indent=2))
    click.echo(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

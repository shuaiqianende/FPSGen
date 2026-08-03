"""FPSGen trajectory evaluation for SemanticKITTI and KITTI-360."""

from __future__ import annotations

import json
import os
from datetime import datetime

import click
import numpy as np
from tqdm import tqdm


SUPPORTED_DATASETS = ("SemanticKITTI", "KITTI360")


_SEMANTICKITTI_MAP_CACHE = {}


def load_map_cropped_ground_truth(record, input_points: np.ndarray, max_range: float) -> np.ndarray:
    """Crop the SemanticKITTI static map into the current LiDAR frame.

    The trajectory evaluator intentionally follows the original FPSGen
    protocol: it does *not* use the fixed 180k ``gt/*.npy`` generation target.
    Instead, it selects all map points within ``max_range`` of the frame pose,
    transforms them back to the sensor frame, applies the vertical limits, and
    keeps only map points inside the coarse 10 m viewpoint grid of the current
    observation.  This produces the dense pose-cropped reference used by the
    multi-range completion metrics.
    """
    import open3d as o3d

    sequence_dir = os.path.dirname(os.path.dirname(record.scan_path))
    map_path = os.path.join(sequence_dir, "map_clean.npy")
    if not os.path.exists(map_path):
        # KITTI-360 and legacy prepared-only datasets have no static map.  The
        # caller can still evaluate them with their record-level target.
        return None

    seq_map = _SEMANTICKITTI_MAP_CACHE.get(map_path)
    if seq_map is None:
        seq_map = np.load(map_path, mmap_mode="r")
        _SEMANTICKITTI_MAP_CACHE[map_path] = seq_map
    map_xyz = seq_map[:, :3]
    translation = np.asarray(record.pose[:3, 3], dtype=np.float32)
    delta = map_xyz - translation
    near_mask = np.sum(delta * delta, axis=1) < float(max_range) ** 2
    local_world = np.asarray(map_xyz[near_mask], dtype=np.float32)
    if local_world.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    homogeneous = np.concatenate(
        [local_world, np.ones((len(local_world), 1), dtype=np.float32)], axis=1
    )
    local = (homogeneous @ np.linalg.inv(record.pose).T)[:, :3]
    local = local[(local[:, 2] > -4.0) & (local[:, 2] < 4.4)]

    # Match the original evaluator's viewpoint filtering.  It suppresses map
    # geometry outside the coarse spatial support of the observed scan.
    current_cloud = o3d.geometry.PointCloud()
    current_cloud.points = o3d.utility.Vector3dVector(input_points[:, :3])
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        current_cloud, voxel_size=10.0
    )
    included = viewpoint_grid.check_if_included(
        o3d.utility.Vector3dVector(local.astype(np.float64, copy=False))
    )
    return local[np.asarray(included, dtype=bool)]


def build_metrics(max_range: float):
    """Create trajectory metrics over the requested radial evaluation range."""
    # Keep metric imports lazy so ``--help`` and argument validation work before
    # the optional Chamfer CUDA extension has been installed.
    from fpsgen.utils.metrics import (
        ChamferDistance_MultiRange,
        CompletionIoU_MultiRange_Morphology,
        EMD_MultiRange,
        PrecisionRecall_MultiRange,
    )

    distance_bins = [(0, max_range)]
    metrics = {
        "iou": CompletionIoU_MultiRange_Morphology(
            voxel_sizes=[0.5, 0.2, 0.1], dist_bins=distance_bins, kernel_radii=[0, 1, 2]
        ),
        "chamfer": ChamferDistance_MultiRange(dist_bins=distance_bins),
        "precision_recall": PrecisionRecall_MultiRange(thresholds=[0.1, 0.2, 0.5], dist_bins=distance_bins),
        "emd": EMD_MultiRange(voxel_size=0.5, dist_bins=[]),
    }
    return metrics


def summarise_metrics(metrics, jsd_3d, jsd_bev):
    """Aggregate per-frame trajectory metrics into one serializable report."""
    from fpsgen.utils.eval_generation import _json_value

    chamfer, chamfer_mean = metrics["chamfer"].compute()
    emd, emd_mean = metrics["emd"].compute()
    precision_recall = metrics["precision_recall"]
    return _json_value({
        "jsd_3d": {key: float(np.mean([item[key] for item in jsd_3d])) for key in jsd_3d[0]},
        "jsd_bev": {key: float(np.mean([item[key] for item in jsd_bev])) for key in jsd_bev[0]},
        "completion_iou": metrics["iou"].compute(),
        "chamfer": chamfer,
        "chamfer_mean": chamfer_mean,
        "precision_recall": precision_recall.compute_at_all_thresholds(),
        "precision_recall_auc": precision_recall.compute_auc(),
        "emd": emd,
        "emd_mean": emd_mean,
    })


@click.command()
@click.option("--path", required=True, type=click.Path(exists=True, file_okay=False), help="dataset root")
@click.option("--dataset", type=click.Choice(SUPPORTED_DATASETS), default="SemanticKITTI")
@click.option("--sequences", default="08", help="comma-separated sequence identifiers")
@click.option("--point-ckpt", required=True, type=click.Path(exists=True),
              help="Stage-3 Point Flow checkpoint")
@click.option("--bev-ckpt", required=True, type=click.Path(exists=True),
              help="Stage-1 BEV Flow checkpoint")
@click.option("--refine", default="", type=click.Path(), help="optional refinement checkpoint")
@click.option("--img-steps", default=10, show_default=True)
@click.option("--point-steps", default=16, show_default=True)
@click.option("--cond-weight", default=2.0, show_default=True,
              help="classifier-free guidance scale; 2 matches the reference inference")
@click.option("--cond-mode", default="100", show_default=True)
@click.option("--max-range", default=50.0, show_default=True)
@click.option("--select-mode", type=click.Choice(["frame", "distance"]), default="frame")
@click.option("--interval-frames", default=1, show_default=True)
@click.option("--interval-m", default=20.0, show_default=True)
@click.option("--max-frames", default=0, show_default=True)
@click.option("--save-pcd/--no-save-pcd", default=False)
@click.option("--save-ply/--no-save-ply", default=False,
              help="also save each prediction as a binary PLY point cloud")
@click.option("--save-gt-ply/--no-save-gt-ply", default=False,
              help="save the pose-cropped trajectory reference as a binary PLY")
def main(path, dataset, sequences, point_ckpt, bev_ckpt, refine, img_steps, point_steps, cond_weight, cond_mode, max_range,
         select_mode, interval_frames, interval_m, max_frames, save_pcd, save_ply, save_gt_ply):
    import open3d as o3d
    from fpsgen.utils.eval_generation import (
        complete_fpsgen,
        load_dataset_records,
        load_fpsgen_completion_class,
        select_records,
    )

    from fpsgen.utils.histogram_metrics import compute_hist_metrics_multirange

    records = select_records(load_dataset_records(dataset, path, sequences), select_mode, interval_frames, interval_m)
    if max_frames:
        records = records[:max_frames]
    model = load_fpsgen_completion_class()(point_ckpt, bev_ckpt, refine, point_steps, cond_weight)
    metrics = build_metrics(max_range)
    jsd_3d, jsd_bev = [], []
    output_dir = os.environ.get("FPSGEN_OUTPUT_DIR", "outputs")
    if save_pcd or save_ply or save_gt_ply:
        os.makedirs(output_dir, exist_ok=True)

    for record in tqdm(records, desc="FPSGen trajectory evaluation"):
        input_points, ground_truth, prediction, _ = complete_fpsgen(
            model,
            record,
            dataset,
            max_range,
            [img_steps, point_steps],
            cond_mode,
            use_prepared_input=False,
        )
        # Trajectory completion metrics use the dense pose-cropped map target
        # for SemanticKITTI.  ``complete_fpsgen`` returns the prepared 180k
        # generation target; that target remains the reference for
        # ``eval_generation`` but is deliberately replaced here.
        if dataset == "SemanticKITTI":
            map_ground_truth = load_map_cropped_ground_truth(record, input_points, max_range)
            if map_ground_truth is not None:
                ground_truth = map_ground_truth
        # ``complete_fpsgen`` returns numeric XYZ arrays, whereas the legacy
        # multi-range metric classes consume Open3D ``PointCloud`` objects.
        # Keep the arrays for histogram/JSD metrics and provide the compatible
        # view to IoU, Chamfer, precision/recall, and EMD implementations.
        ground_truth_cloud = o3d.geometry.PointCloud()
        ground_truth_cloud.points = o3d.utility.Vector3dVector(ground_truth)
        prediction_cloud = o3d.geometry.PointCloud()
        prediction_cloud.points = o3d.utility.Vector3dVector(prediction)
        jsd_3d.append(compute_hist_metrics_multirange(ground_truth, prediction, bev=False, dist_bins=[(0, max_range)]))
        jsd_bev.append(compute_hist_metrics_multirange(ground_truth, prediction, bev=True, dist_bins=[(0, max_range)]))
        metrics["iou"].update(ground_truth_cloud, prediction_cloud)
        metrics["chamfer"].update(ground_truth_cloud, prediction_cloud)
        metrics["precision_recall"].update(ground_truth_cloud, prediction_cloud)
        metrics["emd"].update(ground_truth_cloud, prediction_cloud)
        if save_pcd:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(prediction)
            o3d.io.write_point_cloud(os.path.join(output_dir, f"{dataset}_{record.frame_id:06d}_trajectory.pcd"), cloud)
        if save_ply:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(prediction)
            o3d.io.write_point_cloud(
                os.path.join(output_dir, f"{dataset}_{record.frame_id:06d}_trajectory.ply"),
                cloud,
                write_ascii=False,
            )
        if save_gt_ply:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(ground_truth)
            suffix = "map_crop_gt" if dataset == "SemanticKITTI" else "gt"
            o3d.io.write_point_cloud(
                os.path.join(output_dir, f"{dataset}_{record.frame_id:06d}_{suffix}.ply"),
                cloud,
                write_ascii=False,
            )

    if not jsd_3d:
        raise RuntimeError("No frames were evaluated")
    result = {
        "dataset": dataset,
        "sequences": sequences,
        "frame_count": len(records),
        "gt_source": "map_clean_pose_crop" if dataset == "SemanticKITTI" else "prepared_record_gt",
        "input_source": "raw_velodyne" if dataset == "SemanticKITTI" else "record_input",
        "metrics": summarise_metrics(metrics, jsd_3d, jsd_bev),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fpsgen_trajectory_{dataset}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    click.echo(json.dumps(result, indent=2))
    click.echo(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

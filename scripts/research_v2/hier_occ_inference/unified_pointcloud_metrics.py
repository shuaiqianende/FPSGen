"""One evaluation implementation shared by HS-OccFlow and PatchScene exports."""
from __future__ import annotations

import numpy as np
import torch

from eval_seq08_combinations import physical_grid_ids, support_counts


def evaluate_prediction(predicted_xyz, gt_xyz, geometry, metric_backend, *, device):
    """Evaluate a prediction against the frozen raw-map GT protocol.

    CD is delegated to the caller-provided exact GPU metric backend. IoU uses
    the project's fixed physical-grid support function; JSD uses the audited
    sparse equivalent of the existing FPSGen histogram metric.
    """
    predicted = np.ascontiguousarray(predicted_xyz, dtype=np.float32)
    target = np.ascontiguousarray(gt_xyz, dtype=np.float32)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or not len(predicted):
        raise ValueError("prediction must be a nonempty [N,3] array")
    if target.ndim != 2 or target.shape[1] != 3 or not len(target):
        raise ValueError("GT must be a nonempty [N,3] array")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("prediction and GT must be finite")

    p = torch.from_numpy(predicted).to(device=device)
    g = torch.from_numpy(target).to(device=device)
    cd = metric_backend.evaluate(p, g)
    counts = {}
    for resolution, key in ((0.5, "iou05"), (0.2, "iou02"), (0.1, "iou01")):
        pred_ids = physical_grid_ids(predicted, geometry, resolution=resolution)
        gt_ids = physical_grid_ids(target, geometry, resolution=resolution, target_points=True)
        counts[key] = support_counts(pred_ids, gt_ids)
    from eval_seq08_skip10_joint_2x4_gpu3 import jsd_metrics
    result = {"cd_m": float(cd["cd_m"]), **jsd_metrics(target, predicted),
              "predicted_points": int(len(predicted)), "gt_points": int(len(target))}
    for key, row in counts.items():
        result[key] = row["iou"]
        for part in ("tp", "fp", "fn"):
            result[f"{key}_{part}"] = row[part]
    return result

"""Semantics-aware metrics for FPSGen's three-channel BEV representation.

The BEV is not an RGB image: channels are log-normalized density, maximum
height, and binary occupancy.  These helpers intentionally mirror production
``process_pred_bev`` semantics while keeping evaluation independent from the
PointFlow stage.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def make_valid_disk_mask(grid_size: int = 256, pc_range: float = 50.0,
                         *, device=None) -> torch.Tensor:
    """Return the downstream-valid BEV cells whose centres lie within range."""
    centres = ((torch.arange(grid_size, device=device, dtype=torch.float32) + .5)
               / grid_size * (2.0 * pc_range) - pc_range)
    x, y = torch.meshgrid(centres, centres, indexing="ij")
    return x.square() + y.square() <= pc_range ** 2


def process_generated_bev(pred_bev: torch.Tensor) -> torch.Tensor:
    """Use the exact production occupancy postprocessing contract.

    An occupancy logit strictly greater than zero is occupied.  Density and
    height in all other cells are set to the normalized empty value ``-1``.
    """
    if pred_bev.ndim != 4 or pred_bev.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W] BEV, got {tuple(pred_bev.shape)}")
    output = pred_bev.clone()
    occupied = (output[:, 2:3] > 0.0).to(output.dtype)
    output[:, :2] = output[:, :2] * occupied + (-1.0) * (1.0 - occupied)
    output[:, 2:3] = occupied * 2.0 - 1.0
    return output


def decode_density(density_norm: torch.Tensor, max_density: float = 50.0) -> torch.Tensor:
    """Decode normalized log-density into clipped physical points-per-cell."""
    density_norm = density_norm.clamp(-1.0, 1.0)
    return torch.expm1((density_norm + 1.0) * .5 * math.log1p(max_density))


def decode_height(height_norm: torch.Tensor, min_z: float = -4.0,
                  max_z: float = 5.4) -> torch.Tensor:
    """Decode maximum height in metres without hiding prediction overshoot."""
    return min_z + (height_norm + 1.0) * .5 * (max_z - min_z)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor,
                *, both_empty: float = 1.0) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.full_like(numerator, both_empty),
    )


def _mean_on(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return torch.zeros((), device=values.device, dtype=values.dtype)


def _one_bev_metrics(pred_bev: torch.Tensor, gt_bev: torch.Tensor,
                     input_bev: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    """Calculate all metrics for one processed prediction ``[3,H,W]``."""
    pred = process_generated_bev(pred_bev.unsqueeze(0))[0]
    gt = gt_bev
    inp = input_bev
    valid = valid_mask.bool()
    gt_occ = (gt[2] > 0) & valid
    pred_occ = (pred[2] > 0) & valid
    input_occ = (inp[2] > 0) & valid

    tp = (gt_occ & pred_occ).sum().float()
    fp = ((~gt_occ) & pred_occ & valid).sum().float()
    fn = (gt_occ & (~pred_occ)).sum().float()
    precision = _safe_ratio(tp, tp + fp, both_empty=1.0)
    recall = _safe_ratio(tp, tp + fn, both_empty=1.0)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall, both_empty=1.0)
    iou = _safe_ratio(tp, tp + fp + fn, both_empty=1.0)

    union = (gt_occ | pred_occ) & valid
    intersection = gt_occ & pred_occ
    pred_density = pred[0]
    gt_density = gt[0]
    density_log_mae_union = _mean_on((pred_density - gt_density).abs(), union)
    density_log_mae_intersection = _mean_on((pred_density - gt_density).abs(), intersection)

    pred_raw_density = decode_density(pred_density) * pred_occ
    gt_raw_density = decode_density(gt_density) * gt_occ
    density_raw_mae_intersection = _mean_on(
        (pred_raw_density - gt_raw_density).abs(), intersection
    )
    density_raw_rmse_intersection = torch.sqrt(_mean_on(
        (pred_raw_density - gt_raw_density).square(), intersection
    ))
    pred_mass = pred_raw_density[valid].sum()
    gt_mass = gt_raw_density[valid].sum()
    if float(pred_mass) == 0.0 and float(gt_mass) == 0.0:
        density_mass_tv = torch.zeros((), device=pred.device)
    elif float(pred_mass) == 0.0 or float(gt_mass) == 0.0:
        density_mass_tv = torch.ones((), device=pred.device)
    else:
        density_mass_tv = .5 * (
            pred_raw_density[valid] / pred_mass - gt_raw_density[valid] / gt_mass
        ).abs().sum()
    density_mass_ratio = _safe_ratio(pred_mass, gt_mass, both_empty=1.0)

    pred_height = decode_height(pred[1])
    gt_height = decode_height(gt[1])
    height_mae_gtocc = _mean_on((pred_height - gt_height).abs(), gt_occ)
    height_mae_intersection = _mean_on((pred_height - gt_height).abs(), intersection)
    height_rmse_intersection = torch.sqrt(_mean_on(
        (pred_height - gt_height).square(), intersection
    ))
    height_bias_intersection = _mean_on(pred_height - gt_height, intersection)

    completion_gt = gt_occ & (~input_occ)
    completion_pred = pred_occ & (~input_occ)
    completion_tp = (completion_gt & completion_pred).sum().float()
    completion_fp = (completion_pred & (~completion_gt) & valid).sum().float()
    completion_fn = (completion_gt & (~completion_pred)).sum().float()
    completion_precision = _safe_ratio(completion_tp, completion_tp + completion_fp, both_empty=1.0)
    completion_recall = _safe_ratio(completion_tp, completion_tp + completion_fn, both_empty=1.0)
    completion_f1 = _safe_ratio(
        2.0 * completion_precision * completion_recall,
        completion_precision + completion_recall,
        both_empty=1.0,
    )
    completion_density_mae = _mean_on(
        (pred_density - gt_density).abs(), completion_gt
    )
    completion_height_mae = _mean_on(
        (pred_height - gt_height).abs(), completion_gt
    )

    values = {
        "density_mass_tv": density_mass_tv,
        "density_log_mae_union": density_log_mae_union,
        "density_log_mae_intersection": density_log_mae_intersection,
        "density_raw_mae_intersection": density_raw_mae_intersection,
        "density_raw_rmse_intersection": density_raw_rmse_intersection,
        "density_mass_ratio": density_mass_ratio,
        "height_mae_gtocc_m": height_mae_gtocc,
        "height_mae_intersection_m": height_mae_intersection,
        "height_rmse_intersection_m": height_rmse_intersection,
        "height_bias_intersection_m": height_bias_intersection,
        "occupancy_iou": iou,
        "occupancy_f1": f1,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "completion_precision": completion_precision,
        "completion_recall": completion_recall,
        "completion_f1": completion_f1,
        "completion_density_log_mae": completion_density_mae,
        "completion_height_mae_m": completion_height_mae,
        "valid_cells": valid.sum().float(),
        "gt_occupied_cells": gt_occ.sum().float(),
        "pred_occupied_cells": pred_occ.sum().float(),
        "completion_gt_cells": completion_gt.sum().float(),
    }
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def compute_bev_metrics(pred_bev: torch.Tensor, gt_bev: torch.Tensor,
                        input_bev: torch.Tensor, valid_mask: torch.Tensor) -> List[Dict[str, float]]:
    """Compute one metrics dictionary per batch item.

    ``pred_bev`` is raw flow output.  It is postprocessed internally using the
    same strict occupancy threshold as FPSGen inference.
    """
    if pred_bev.shape != gt_bev.shape or pred_bev.shape != input_bev.shape:
        raise ValueError("pred_bev, gt_bev, and input_bev must have identical shape")
    if pred_bev.ndim != 4 or pred_bev.shape[1] != 3:
        raise ValueError("BEV tensors must have shape [B,3,H,W]")
    if valid_mask.shape != pred_bev.shape[-2:]:
        raise ValueError("valid_mask must have BEV spatial shape")
    return [_one_bev_metrics(pred_bev[i], gt_bev[i], input_bev[i], valid_mask)
            for i in range(pred_bev.shape[0])]


def compute_range_metrics(pred_bev: torch.Tensor, gt_bev: torch.Tensor,
                          input_bev: torch.Tensor, *, pc_range: float = 50.0,
                          radial_bins: Sequence[Sequence[float]] = ((0., 20.), (20., 35.), (35., 50.))
                          ) -> List[Dict[str, float]]:
    """Return the same metrics within each valid radial annulus."""
    grid_size = pred_bev.shape[-1]
    disk = make_valid_disk_mask(grid_size, pc_range, device=pred_bev.device)
    centres = ((torch.arange(grid_size, device=pred_bev.device, dtype=torch.float32) + .5)
               / grid_size * (2.0 * pc_range) - pc_range)
    x, y = torch.meshgrid(centres, centres, indexing="ij")
    radius = torch.sqrt(x.square() + y.square())
    rows: List[Dict[str, float]] = []
    for low, high in radial_bins:
        annulus = disk & (radius >= float(low)) & (radius < float(high))
        for result in compute_bev_metrics(pred_bev, gt_bev, input_bev, annulus):
            result.update({"range_min_m": float(low), "range_max_m": float(high)})
            rows.append(result)
    return rows

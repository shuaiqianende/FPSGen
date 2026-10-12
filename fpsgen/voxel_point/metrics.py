"""Independent 3D density and point geometry diagnostics; no dense pair cost."""

import torch

from fpsgen.ops.endpoint_refinement.knn import knn
from .density_loss import count_sum_pool


def chamfer_sqrt(predicted, target, *, backend="keops"):
    # Sparse KNN indices are nondifferentiable, selected coordinate distances
    # are differentiable. This is symmetric sqrt-CD, with the PWM 0.5 factor.
    p, g = predicted.float(), target.float()
    _, pg = knn(p, g, k=1, backend=backend)
    _, gp = knn(g, p, k=1, backend=backend)
    return .5 * ((p - g[pg[:, 0]]).norm(dim=-1).mean() + (g - p[gp[:, 0]]).norm(dim=-1).mean())


@torch.no_grad()
def point_metrics(predicted, target, *, backend="keops", spec=None):
    p, g = predicted.float(), target.float()
    if not torch.isfinite(p).all() or not torch.isfinite(g).all():
        raise ValueError("Nonfinite point metrics input")
    dp, _ = knn(p, g, 1, backend)
    dg, _ = knn(g, p, 1, backend)
    spacing, _ = knn(p, p, 2, backend)
    spacing = spacing[:, 1].clamp_min(0).sqrt()
    result = {"cd": float(.5 * (dp.sqrt().mean() + dg.sqrt().mean())),
              "spacing_mean": float(spacing.mean()),
              "spacing_cv": float(spacing.std(unbiased=False) / spacing.mean().clamp_min(1e-12))}
    for threshold in (.1, .2):
        precision = float((dp < threshold ** 2).float().mean())
        coverage = float((dg < threshold ** 2).float().mean())
        result[f"fscore_{threshold}"] = 2 * precision * coverage / max(precision + coverage, 1e-12)
        result[f"coverage_{threshold}"] = coverage
    if spec is not None:
        result["out_of_range_ratio"] = float((~spec.valid_points(p)).float().mean())
    return result


@torch.no_grad()
def density_metrics(predicted, counts, spec, *, threshold):
    p = predicted.float().reshape(spec.shape)
    g = counts.float().reshape(spec.shape)
    valid = spec.valid_circle_mask(device=p.device)
    if not torch.isfinite(p).all() or bool((p < 0).any()):
        raise ValueError("Invalid density metrics input")
    p = p * valid
    po, go = (p >= threshold) & valid, (g > 0) & valid
    intersection, union = (po & go).sum(), (po | go).sum()
    mass_p, mass_g = p.sum(), g.sum()
    cp, cg = count_sum_pool(p[None, None]), count_sum_pool(g[None, None])
    return {"occupancy_iou": float(intersection / union.clamp_min(1)),
        "occupancy_f1": float(2 * intersection / (po.sum() + go.sum()).clamp_min(1)),
        "occupied_count_mae": float((p[go] - g[go]).abs().mean()) if go.any() else 0.,
        "log_count_mae": float((p[valid].log1p() - g[valid].log1p()).abs().mean()),
        "mass_tv": float(.5 * (p / mass_p.clamp_min(1e-12) - g / mass_g.clamp_min(1e-12)).abs().sum()),
        "coarse_mass_tv": float(.5 * (cp / mass_p.clamp_min(1e-12) - cg / mass_g.clamp_min(1e-12)).abs().sum()),
        "mass_calibration_error": float((mass_p - mass_g).abs() / mass_g.clamp_min(1)),
        "density_sum": float(mass_p), "occupied_voxel_fraction": float(po.sum() / valid.sum())}

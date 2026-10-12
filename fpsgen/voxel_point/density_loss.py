"""FP32 density-only losses and density-derived geometric run supervision."""

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = dict(pos=1., bg=.1, coarse=.5, dist=.1, mass=.1, direction=.1, auxiliary=.3)


def count_sum_pool(counts):
    return F.avg_pool3d(counts.float(), kernel_size=2, stride=2) * 8


def occupied_run_lengths(counts, valid):
    occupied = (counts > 0) & valid
    result = []
    for axis in range(1, 4):
        for sign in (1, -1):
            run = torch.zeros_like(counts, dtype=torch.float32)
            order = range(counts.shape[axis] - 1, -1, -1) if sign == 1 else range(counts.shape[axis])
            previous = None
            for index in order:
                here = [slice(None)] * 4
                here[axis] = index
                here = tuple(here)
                value = occupied[here].float()
                if previous is not None:
                    value = value * (run[previous] + 1)
                run[here] = value
                previous = here
            result.append(run / counts.shape[axis])
    return torch.stack(result, 1)


def density_terms(predicted, counts, valid):
    p, n = predicted.float(), counts.float()
    if p.shape != n.shape or p.ndim != 5 or p.shape[1] != 1:
        raise ValueError("Density and counts must be [B,1,X,Y,Z]")
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("Density must be finite and nonnegative")
    valid = valid.expand_as(p)
    p = p * valid
    positive, background = (n > 0) & valid, (n == 0) & valid
    error = F.smooth_l1_loss(torch.log1p(p), torch.log1p(n), reduction="none")
    terms = {"pos": (error * positive).sum() / positive.sum().clamp_min(1),
             "bg": (error * background).sum() / background.sum().clamp_min(1)}
    cp, cn = count_sum_pool(p), count_sum_pool(n * valid)
    cv = count_sum_pool(valid.float()) > 0
    terms["coarse"] = (F.smooth_l1_loss(torch.log1p(cp), torch.log1p(cn), reduction="none") * cv).sum() / cv.sum().clamp_min(1)
    axes = (1, 2, 3, 4)
    pm, nm = p.sum(axes, keepdim=True), (n * valid).sum(axes, keepdim=True)
    terms["dist"] = .5 * (p / pm.clamp_min(1e-12) - n * valid / nm.clamp_min(1e-12)).abs().sum(axes).mean()
    terms["mass"] = ((pm - nm).abs() / nm.clamp_min(1)).mean()
    return terms


def density_loss(outputs, counts, spec, weights=None):
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        if set(weights) != set(w):
            raise ValueError("All seven density-loss weights must be configured")
        w.update(weights)
    valid = spec.valid_circle_mask(device=counts.device)[None, None]
    terms = density_terms(outputs["pred_density"], counts, valid)
    target = occupied_run_lengths(counts[:, 0], valid[:, 0])
    offset = outputs["offset"].float()
    target = F.interpolate(target, size=offset.shape[-3:], mode="trilinear", align_corners=False)
    occupied = F.adaptive_max_pool3d(((counts > 0) & valid).float(), offset.shape[-3:]) > 0
    # Average over all six directions on occupied cells only.
    terms["direction"] = ((offset - target).abs() * occupied).sum() / (6 * occupied.sum()).clamp_min(1)
    aux = density_terms(outputs["aux_density"], counts, valid)
    terms["auxiliary"] = sum(w[name] * aux[name] for name in ("pos", "bg", "coarse", "dist", "mass"))
    terms["total"] = sum(w[name] * terms[name] for name in w)
    return terms

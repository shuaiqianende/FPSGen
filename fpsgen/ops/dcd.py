"""Training-safe Density-aware Chamfer Distance for FPSGen.

This module intentionally depends only on PyTorch and FPSGen's pinned CUDA
Chamfer extension.  It mirrors ``fpsgen.utils.metrics.calc_dcd`` exactly, but
does not import the evaluation stack (Open3D, SciPy, torchist, ...).
"""

from __future__ import annotations

import torch

from fpsgen.ops.chamfer import Chamfer3DDist


_chamfer_dist = Chamfer3DDist()


def density_aware_chamfer(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    n_lambda: float = 1.0,
    non_reg: bool = False,
    return_raw: bool = False,
):
    """Compute the Density-aware Chamfer Distance (DCD).

    Args:
        pred: Predicted point cloud, shaped ``[B, N, 3]``.
        target: Target point cloud, shaped ``[B, M, 3]``.
        alpha: Exponential distance coefficient. Distances from
            :class:`Chamfer3DDist` are already squared Euclidean distances.
        n_lambda: Assignment-multiplicity exponent.
        non_reg: Preserve the historical DCD cardinality-ratio option.
        return_raw: Also return the Chamfer distances and nearest-neighbour
            indices, with the same return contract as ``calc_dcd``.

    Returns:
        A list ``[loss, cd_p, cd_t]`` of batch-wise tensors, optionally
        followed by ``dist1, dist2, idx1, idx2``.  The order intentionally
        matches ``fpsgen.utils.metrics.calc_dcd`` for reproducibility.
    """
    if pred.ndim != 3 or target.ndim != 3 or pred.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError("DCD expects pred and target tensors shaped [B, N, 3].")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("DCD pred and target must have the same batch size.")
    if pred.shape[1] == 0 or target.shape[1] == 0:
        raise ValueError("DCD does not accept empty point clouds.")
    if alpha <= 0:
        raise ValueError("DCD alpha must be positive.")

    pred = pred.float()
    target = target.float()
    _, n_pred, _ = pred.shape
    _, n_target, _ = target.shape

    if non_reg:
        frac_pred_to_target = max(1.0, n_pred / n_target)
        frac_target_to_pred = max(1.0, n_target / n_pred)
    else:
        frac_pred_to_target = n_pred / n_target
        frac_target_to_pred = n_target / n_pred

    # The historical calc_dcd calls calc_cd(pred, target), whose helper calls
    # chamfer(target, pred). Keep that ordering exactly: dist1 is target->pred
    # and idx1 indexes pred; dist2 is pred->target and idx2 indexes target.
    dist1, dist2, idx1, idx2 = _chamfer_dist(target, pred)
    cd_p = (torch.sqrt(dist1).mean(dim=1) + torch.sqrt(dist2).mean(dim=1)) / 2
    cd_t = dist1.mean(dim=1) + dist2.mean(dim=1)

    # dist1/dist2 are squared Euclidean distances: do not apply sqrt before
    # the DCD exponential term.
    exp_dist1 = torch.exp(-alpha * dist1)
    exp_dist2 = torch.exp(-alpha * dist2)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach().pow(n_lambda)
    weight1 = (weight1 + 1e-6).reciprocal() * frac_target_to_pred
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach().pow(n_lambda)
    weight2 = (weight2 + 1e-6).reciprocal() * frac_pred_to_target
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    result = [(loss1 + loss2) / 2, cd_p, cd_t]
    if return_raw:
        result.extend([dist1, dist2, idx1, idx2])
    return result

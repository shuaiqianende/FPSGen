"""Teacher-endpoint local sparse balanced-OT construction and refinement."""

from __future__ import annotations

import torch

from .knn import knn
from .sinkhorn import sparse_sinkhorn


def _coalesce_edges(source: torch.Tensor, target: torch.Tensor,
                    cost: torch.Tensor, num_target: int):
    """Deduplicate sparse edges, retaining the least-cost copy."""
    keys = source.to(torch.int64) * num_target + target.to(torch.int64)
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    coalesced_cost = torch.full((unique_keys.numel(),), torch.inf,
                                device=cost.device, dtype=cost.dtype)
    coalesced_cost.scatter_reduce_(0, inverse, cost, reduce="amin", include_self=True)
    return (unique_keys // num_target).long(), (unique_keys % num_target).long(), coalesced_cost


def _sparse_knn_graph(endpoint: torch.Tensor, target: torch.Tensor, k: int,
                      backend: str):
    """Build query KNN edges plus reciprocal target KNN edges for coverage."""
    n_source, n_target = endpoint.shape[0], target.shape[0]
    d2_st, idx_st = knn(endpoint, target, k=k, backend=backend)
    src_st = torch.arange(n_source, device=endpoint.device).repeat_interleave(k)
    tgt_st = idx_st.reshape(-1)
    cost_st = d2_st.reshape(-1)

    # A query-only graph can omit target cells entirely, making balanced OT
    # impossible. Reciprocal KNN edges ensure every target has a feasible edge.
    d2_ts, idx_ts = knn(target, endpoint, k=k, backend=backend)
    src_ts = idx_ts.reshape(-1)
    tgt_ts = torch.arange(n_target, device=endpoint.device).repeat_interleave(k)
    cost_ts = d2_ts.reshape(-1)
    return _coalesce_edges(torch.cat((src_st, src_ts)),
                           torch.cat((tgt_st, tgt_ts)),
                           torch.cat((cost_st, cost_ts)), n_target)


@torch.no_grad()
def refine_endpoint(endpoint: torch.Tensor, target: torch.Tensor, *, k: int = 16,
                    alpha: float = 1.0, epsilon: float = 0.05,
                    iterations: int = 100, backend: str = "keops"):
    """Refine an endpoint toward sparse balanced-OT barycentres.

    ``alpha=0`` exactly returns ``endpoint``.  This intentionally interpolates
    from the teacher endpoint (not P0), isolating post-teacher refinement.
    Returns the refined points and diagnostics, with no dense ``N x M`` cost.
    """
    if endpoint.ndim != 2 or endpoint.shape[1] != 3:
        raise ValueError("endpoint must have shape [N, 3]")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target must have shape [M, 3]")
    if endpoint.device != target.device or endpoint.dtype != target.dtype:
        raise ValueError("endpoint and target must share device and dtype")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        return endpoint.clone(), {"row_error": 0.0, "col_error": 0.0,
                                  "entropy": 0.0, "num_edges": 0}

    source, target_index, cost = _sparse_knn_graph(endpoint, target, k, backend)
    plan = sparse_sinkhorn(source, target_index, cost, endpoint.shape[0],
                           target.shape[0], epsilon, iterations)
    row_mass = torch.zeros(endpoint.shape[0], device=endpoint.device, dtype=endpoint.dtype)
    row_mass.scatter_add_(0, source, plan)
    bary_sum = torch.zeros_like(endpoint)
    bary_sum.index_add_(0, source, plan[:, None] * target[target_index])
    barycentre = bary_sum / row_mass[:, None].clamp_min(torch.finfo(endpoint.dtype).tiny)
    refined = endpoint + alpha * (barycentre - endpoint)

    col_mass = torch.zeros(target.shape[0], device=endpoint.device, dtype=endpoint.dtype)
    col_mass.scatter_add_(0, target_index, plan)
    expected_row = 1.0 / endpoint.shape[0]
    expected_col = 1.0 / target.shape[0]
    entropy = -(plan * torch.log(plan.clamp_min(torch.finfo(plan.dtype).tiny))).sum()
    return refined, {
        "row_error": float((row_mass - expected_row).abs().max().item()),
        "col_error": float((col_mass - expected_col).abs().max().item()),
        "entropy": float(entropy.item()),
        "num_edges": int(plan.numel()),
    }


@torch.no_grad()
def knn_barycentre(endpoint: torch.Tensor, target: torch.Tensor, *, k: int = 16,
                   backend: str = "keops"):
    """Unbalanced cheap baseline: mean of each endpoint's K nearest GT points."""
    _, indices = knn(endpoint, target, k=k, backend=backend)
    return target[indices].mean(dim=1)

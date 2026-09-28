"""Log-domain Sinkhorn on a sparse bipartite transport graph."""

from __future__ import annotations

import torch


def _segment_logsumexp(values: torch.Tensor, index: torch.Tensor,
                       size: int) -> torch.Tensor:
    """Equivalent to ``logsumexp(values[index == i])`` for every segment."""
    maximum = torch.full((size,), -torch.inf, device=values.device,
                         dtype=values.dtype)
    maximum.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    shifted = torch.exp(values - maximum[index])
    total = torch.zeros((size,), device=values.device, dtype=values.dtype)
    total.scatter_add_(0, index, shifted)
    return maximum + torch.log(total.clamp_min(torch.finfo(values.dtype).tiny))


def sparse_sinkhorn(source_index: torch.Tensor, target_index: torch.Tensor,
                    cost: torch.Tensor, num_source: int, num_target: int,
                    epsilon: float = 0.05, iterations: int = 100):
    """Solve uniform balanced OT on edges only, returning an edge transport.

    The graph must give every source and target at least one edge. Costs are
    squared distances in metres squared. The result has total mass one.
    """
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    if source_index.numel() == 0:
        raise ValueError("Sparse OT graph must contain at least one edge")
    if source_index.min() < 0 or source_index.max() >= num_source:
        raise ValueError("source indices are outside graph bounds")
    if target_index.min() < 0 or target_index.max() >= num_target:
        raise ValueError("target indices are outside graph bounds")
    if torch.unique(source_index).numel() != num_source:
        raise ValueError("Every source must have at least one sparse OT edge")
    if torch.unique(target_index).numel() != num_target:
        raise ValueError("Every target must have at least one sparse OT edge")

    log_kernel = -cost / epsilon
    log_a = -torch.log(torch.tensor(float(num_source), device=cost.device,
                                    dtype=cost.dtype))
    log_b = -torch.log(torch.tensor(float(num_target), device=cost.device,
                                    dtype=cost.dtype))
    log_u = torch.zeros(num_source, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(num_target, device=cost.device, dtype=cost.dtype)

    for _ in range(iterations):
        log_u = log_a - _segment_logsumexp(log_kernel + log_v[target_index],
                                            source_index, num_source)
        log_v = log_b - _segment_logsumexp(log_kernel + log_u[source_index],
                                            target_index, num_target)
    log_plan = log_kernel + log_u[source_index] + log_v[target_index]
    return torch.exp(log_plan)

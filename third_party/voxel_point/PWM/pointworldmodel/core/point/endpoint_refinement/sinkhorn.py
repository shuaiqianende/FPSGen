"""Sparse log-domain balanced Sinkhorn on a fixed edge graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SinkhornResult:
    plan: torch.Tensor
    row_mass: torch.Tensor
    col_mass: torch.Tensor
    row_error: float
    col_error: float
    max_row_probability: float
    mean_row_entropy: float
    iterations: int


def coalesce_edges(
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    cost: torch.Tensor,
    num_dst: int,
):
    """Coalesce duplicate (source, target) edges using the minimum cost."""
    if not (src_idx.ndim == dst_idx.ndim == cost.ndim == 1):
        raise ValueError("sparse edge tensors must be one-dimensional")
    if not (src_idx.numel() == dst_idx.numel() == cost.numel()):
        raise ValueError("sparse edge tensors must have equal lengths")
    if src_idx.numel() == 0:
        raise ValueError("cannot coalesce an empty edge graph")
    key = src_idx.long() * int(num_dst) + dst_idx.long()
    unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
    if unique_key.numel() == key.numel():
        return src_idx.long(), dst_idx.long(), cost.float()
    unique_cost = torch.full(
        (unique_key.numel(),),
        torch.inf,
        dtype=torch.float32,
        device=cost.device,
    )
    unique_cost.scatter_reduce_(
        0,
        inverse,
        cost.float(),
        reduce="amin",
        include_self=True,
    )
    return (
        torch.div(unique_key, int(num_dst), rounding_mode="floor").long(),
        (unique_key % int(num_dst)).long(),
        unique_cost,
    )


def _group_logsumexp(values: torch.Tensor, groups: torch.Tensor, group_count: int):
    """Compute logsumexp by sparse integer group IDs with scatter operations."""
    if values.ndim != 1 or groups.ndim != 1 or values.numel() != groups.numel():
        raise ValueError("values and groups must be aligned one-dimensional tensors")
    negative_inf = torch.tensor(-torch.inf, device=values.device, dtype=values.dtype)
    group_max = torch.full(
        (int(group_count),),
        negative_inf.item(),
        device=values.device,
        dtype=values.dtype,
    )
    group_max.scatter_reduce_(
        0,
        groups.long(),
        values,
        reduce="amax",
        include_self=True,
    )
    shifted = torch.exp(values - group_max[groups.long()])
    group_sum = torch.zeros(
        (int(group_count),), device=values.device, dtype=values.dtype
    )
    group_sum.scatter_add_(0, groups.long(), shifted)
    return group_max + torch.log(group_sum.clamp_min(torch.finfo(values.dtype).tiny))


class SparseBalancedSinkhorn:
    """Balanced unit-marginal Sinkhorn using only sparse graph edges."""

    def __init__(self, temperature: float = 0.5, iterations: int = 10):
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        if int(iterations) < 1:
            raise ValueError("iterations must be positive")
        self.temperature = float(temperature)
        self.iterations = int(iterations)

    @torch.no_grad()
    def solve(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        normalized_cost: torch.Tensor,
        num_src: int,
        num_dst: int,
    ) -> SinkhornResult:
        # scatter_reduce/scatter_add do not have deterministic CUDA kernels in
        # the supported PyTorch version.  Sinkhorn is an explicitly exploratory
        # raw-plan path here, so allow these sparse reductions only inside this
        # solver and restore the caller's setting afterwards.
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        except AttributeError:
            warn_only = False
        if deterministic:
            torch.use_deterministic_algorithms(False)
        try:
            return self._solve_impl(
                src_idx, dst_idx, normalized_cost, num_src, num_dst
            )
        finally:
            if deterministic:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)

    @torch.no_grad()
    def _solve_impl(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        normalized_cost: torch.Tensor,
        num_src: int,
        num_dst: int,
    ) -> SinkhornResult:
        src_idx = src_idx.long().reshape(-1)
        dst_idx = dst_idx.long().reshape(-1)
        cost = normalized_cost.float().reshape(-1)
        if src_idx.numel() == 0:
            raise ValueError("Sinkhorn graph is empty")
        if int(torch.unique(src_idx).numel()) != int(num_src):
            raise ValueError("every source row must have at least one sparse edge")
        if int(torch.unique(dst_idx).numel()) != int(num_dst):
            raise ValueError("every target column must have at least one sparse edge")

        log_kernel = -cost / self.temperature
        log_u = torch.zeros(int(num_src), device=cost.device, dtype=torch.float32)
        log_v = torch.zeros(int(num_dst), device=cost.device, dtype=torch.float32)
        for _ in range(self.iterations):
            row_lse = _group_logsumexp(
                log_kernel + log_v[dst_idx], src_idx, int(num_src)
            )
            log_u = -row_lse
            col_lse = _group_logsumexp(
                log_kernel + log_u[src_idx], dst_idx, int(num_dst)
            )
            log_v = -col_lse

        log_plan = log_kernel + log_u[src_idx] + log_v[dst_idx]
        plan = torch.exp(log_plan)
        row_mass = torch.zeros(int(num_src), device=cost.device, dtype=torch.float32)
        row_mass.scatter_add_(0, src_idx, plan)
        col_mass = torch.zeros(int(num_dst), device=cost.device, dtype=torch.float32)
        col_mass.scatter_add_(0, dst_idx, plan)
        row_error = float((row_mass - 1.0).abs().mean().item())
        col_error = float((col_mass - 1.0).abs().mean().item())

        row_max = torch.zeros_like(row_mass)
        row_max.scatter_reduce_(
            0, src_idx, plan, reduce="amax", include_self=True
        )
        entropy_terms = -(plan.clamp_min(torch.finfo(plan.dtype).tiny) * log_plan)
        row_entropy = torch.zeros_like(row_mass)
        row_entropy.scatter_add_(0, src_idx, entropy_terms)
        return SinkhornResult(
            plan=plan,
            row_mass=row_mass,
            col_mass=col_mass,
            row_error=row_error,
            col_error=col_error,
            max_row_probability=float(row_max.mean().item()),
            mean_row_entropy=float(row_entropy.mean().item()),
            iterations=self.iterations,
        )

"""Teacher-guided sparse global balanced OT endpoint refinement."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch

from .base import EndpointRefiner
from .knn import KeOpsGlobalKNN, KNNResult
from .sinkhorn import SparseBalancedSinkhorn, SinkhornResult, coalesce_edges


def _percentile_stats(values: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return {f"{prefix}{name}": 0.0 for name in ("mean", "p50", "p90", "p95", "p99")}
    quantiles = torch.tensor(
        [0.50, 0.90, 0.95, 0.99], device=values.device, dtype=values.dtype
    )
    q = torch.quantile(values, quantiles)
    return {
        f"{prefix}mean": float(values.mean().item()),
        f"{prefix}p50": float(q[0].item()),
        f"{prefix}p90": float(q[1].item()),
        f"{prefix}p95": float(q[2].item()),
        f"{prefix}p99": float(q[3].item()),
    }


class SparseGlobalBalancedOTRefiner(EndpointRefiner):
    """Refine a source-indexed Teacher endpoint with sparse global OT.

    The refiner intentionally has no trainable state.  All KNN, Sinkhorn and
    barycentric operations run under ``torch.no_grad`` so a frozen Teacher is
    never part of the Student autograd graph.
    """

    def __init__(
        self,
        *,
        k_max: int = 32,
        initial_k: int = 16,
        zero_degree_repair_k: int = 8,
        bad_column_repair_k: int = 16,
        spacing_k: int = 8,
        spacing_query_count: int = 8192,
        temperature: float = 0.5,
        sinkhorn_iterations: int = 10,
        barycentric_alpha: float = 1.0,
        marginal_accept_threshold: float = 0.02,
        marginal_failure_threshold: float = 0.05,
        fallback_on_invalid: bool = True,
    ):
        if not 1 <= initial_k <= k_max:
            raise ValueError("initial_k must lie in [1, k_max]")
        if not 1 <= zero_degree_repair_k <= k_max:
            raise ValueError("zero_degree_repair_k must lie in [1, k_max]")
        if not 1 <= bad_column_repair_k <= k_max:
            raise ValueError("bad_column_repair_k must lie in [1, k_max]")
        if spacing_k < 1 or spacing_query_count < 1:
            raise ValueError("spacing parameters must be positive")
        if not 0.0 <= float(barycentric_alpha) <= 1.0:
            raise ValueError("barycentric_alpha must lie in [0, 1]")
        self.k_max = int(k_max)
        self.initial_k = int(initial_k)
        self.zero_degree_repair_k = int(zero_degree_repair_k)
        self.bad_column_repair_k = int(bad_column_repair_k)
        self.spacing_k = int(spacing_k)
        self.spacing_query_count = int(spacing_query_count)
        self.temperature = float(temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.barycentric_alpha = float(barycentric_alpha)
        self.marginal_accept_threshold = float(marginal_accept_threshold)
        self.marginal_failure_threshold = float(marginal_failure_threshold)
        self.fallback_on_invalid = bool(fallback_on_invalid)
        self.knn = KeOpsGlobalKNN(max_k=self.k_max)
        self.sinkhorn = SparseBalancedSinkhorn(
            temperature=self.temperature,
            iterations=self.sinkhorn_iterations,
        )
        self._spacing_cache: Dict[str, float] = {}

    @staticmethod
    def _sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            k_max=config.get("k_max", 32),
            initial_k=config.get("initial_k", 16),
            zero_degree_repair_k=config.get("zero_degree_repair_k", 8),
            bad_column_repair_k=config.get("bad_column_repair_k", 16),
            spacing_k=config.get("spacing_k", 8),
            spacing_query_count=config.get("spacing_query_count", 8192),
            temperature=config.get("temperature", 0.5),
            sinkhorn_iterations=config.get("sinkhorn_iterations", 10),
            barycentric_alpha=config.get("barycentric_alpha", 1.0),
            marginal_accept_threshold=config.get("marginal_accept_threshold", 0.02),
            marginal_failure_threshold=config.get("marginal_failure_threshold", 0.05),
            fallback_on_invalid=config.get("fallback_on_invalid", True),
        )

    @staticmethod
    def _validate_points(source, teacher_endpoint, target):
        for name, points in (
            ("source", source),
            ("teacher_endpoint", teacher_endpoint),
            ("target", target),
        ):
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"{name} must have shape [N, 3]")
            if not torch.isfinite(points).all():
                raise ValueError(f"{name} contains non-finite coordinates")
        if source.shape[0] != teacher_endpoint.shape[0]:
            raise ValueError("source and teacher_endpoint must have equal point counts")
        if source.shape[0] != target.shape[0]:
            raise ValueError(
                "balanced OT requires source, Teacher endpoint and target to have equal counts"
            )
        if source.device != target.device or source.device != teacher_endpoint.device:
            raise ValueError("all endpoint-refiner tensors must share a device")

    @torch.no_grad()
    def estimate_reference_spacing(
        self,
        reference: torch.Tensor,
        *,
        cache_key: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Estimate median reference 8-NN spacing with global KeOps KNN."""
        if cache_key is not None and cache_key in self._spacing_cache:
            return self._spacing_cache[cache_key], reference.new_empty((0,))
        count = min(self.spacing_query_count, reference.shape[0])
        if generator is None:
            generator = torch.Generator(device=reference.device)
            generator.manual_seed(0)
        query_indices = torch.randperm(
            reference.shape[0], device=reference.device, generator=generator
        )[:count]
        # K=spacing_k+1 gives one extra slot for the query point itself.
        knn = self.knn.query(
            reference[query_indices], reference, self.spacing_k + 1
        )
        self_mask = knn.indices == query_indices[:, None]
        distances = torch.where(knn.dist2 >= 0, knn.dist2, knn.dist2)
        distances = torch.where(self_mask, torch.inf, distances)
        distances, _ = torch.sort(distances, dim=1)
        spacing = torch.sqrt(distances[:, self.spacing_k - 1].clamp_min(0.0))
        d_ref = float(torch.median(spacing).item())
        d_ref = max(d_ref, 1e-6)
        if cache_key is not None:
            self._spacing_cache[cache_key] = d_ref
        return d_ref, spacing

    @staticmethod
    def _forward_edges(knn: KNNResult, active_k: int, d_ref: float):
        rows = torch.arange(
            knn.indices.shape[0], device=knn.indices.device, dtype=torch.long
        )[:, None].expand(-1, active_k).reshape(-1)
        cols = knn.indices[:, :active_k].reshape(-1).long()
        cost = knn.dist2[:, :active_k].reshape(-1).float()
        cost = cost / (float(d_ref) ** 2 + 1e-12)
        return rows, cols, cost

    def _reverse_edges(
        self,
        query_points: torch.Tensor,
        source_points: torch.Tensor,
        destination_ids: torch.Tensor,
        repair_k: int,
        d_ref: float,
    ):
        if destination_ids.numel() == 0:
            return (
                source_points.new_empty((0,), dtype=torch.long),
                destination_ids.new_empty((0,), dtype=torch.long),
                source_points.new_empty((0,), dtype=torch.float32),
            )
        reverse = self.knn.query(
            query_points[destination_ids], source_points, repair_k
        )
        src = reverse.indices.reshape(-1).long()
        dst = destination_ids[:, None].expand(-1, repair_k).reshape(-1).long()
        cost = reverse.dist2.reshape(-1).float()
        cost = cost / (float(d_ref) ** 2 + 1e-12)
        return src, dst, cost

    @staticmethod
    def _coverage(dst_idx: torch.Tensor, num_dst: int):
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        except AttributeError:
            warn_only = False
        if deterministic:
            torch.use_deterministic_algorithms(False)
        try:
            coverage = torch.bincount(dst_idx, minlength=int(num_dst))
        finally:
            if deterministic:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
        return coverage

    def _make_graph(
        self,
        knn: KNNResult,
        *,
        active_k: int,
        d_ref: float,
        num_src: int,
        num_dst: int,
        target: torch.Tensor,
        source: torch.Tensor,
        reverse_k: int,
        only_zero_degree: bool = True,
        bad_columns: Optional[torch.Tensor] = None,
    ):
        src, dst, cost = self._forward_edges(knn, active_k, d_ref)
        incoming = self._coverage(dst, num_dst)
        if bad_columns is None:
            repair_columns = torch.nonzero(incoming == 0, as_tuple=False).flatten()
        elif only_zero_degree:
            repair_columns = bad_columns[incoming[bad_columns] == 0]
        else:
            repair_columns = bad_columns
        repair_src, repair_dst, repair_cost = self._reverse_edges(
            target,
            source,
            repair_columns,
            reverse_k,
            d_ref,
        )
        repair_edges = int(repair_src.numel())
        if repair_edges:
            src = torch.cat([src, repair_src], dim=0)
            dst = torch.cat([dst, repair_dst], dim=0)
            cost = torch.cat([cost, repair_cost], dim=0)
        src, dst, cost = coalesce_edges(src, dst, cost, num_dst)
        final_incoming = self._coverage(dst, num_dst)
        return src, dst, cost, incoming, final_incoming, repair_edges

    @staticmethod
    def _barycentric(
        source: torch.Tensor,
        target: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        plan: torch.Tensor,
        alpha: float,
    ):
        row_mass = torch.zeros(source.shape[0], device=source.device, dtype=torch.float32)
        row_mass.scatter_add_(0, src_idx, plan)
        numerator = torch.zeros(
            (source.shape[0], 3), device=source.device, dtype=torch.float32
        )
        numerator.index_add_(0, src_idx, plan[:, None] * target[dst_idx].float())
        barycentric = numerator / row_mass.clamp_min(1e-12)[:, None]
        return source.float() + float(alpha) * (barycentric - source.float())

    @torch.no_grad()
    def refine(self, source, teacher_endpoint, target, *, cache_key=None):
        self._validate_points(source, teacher_endpoint, target)
        original_dtype = teacher_endpoint.dtype
        source_f = source.float().contiguous()
        teacher_f = teacher_endpoint.float().contiguous()
        target_f = target.float().contiguous()
        num_points = source_f.shape[0]
        timing = {}
        total_start = time.perf_counter()

        spacing_start = time.perf_counter()
        self._sync(target_f.device)
        spacing_generator = torch.Generator(device=target_f.device)
        spacing_generator.manual_seed(1729)
        d_ref, spacing_samples = self.estimate_reference_spacing(
            target_f, cache_key=cache_key, generator=spacing_generator
        )
        self._sync(target_f.device)
        timing["spacing_seconds"] = time.perf_counter() - spacing_start

        knn_start = time.perf_counter()
        self._sync(target_f.device)
        knn32 = self.knn.query(teacher_f, target_f, self.k_max)
        self._sync(target_f.device)
        timing["keops_knn_seconds"] = time.perf_counter() - knn_start

        graph_start = time.perf_counter()
        self._sync(target_f.device)
        (
            src16,
            dst16,
            cost16,
            incoming16_before,
            incoming16_after,
            repair_edges16,
        ) = self._make_graph(
            knn32,
            active_k=self.initial_k,
            d_ref=d_ref,
            num_src=num_points,
            num_dst=target_f.shape[0],
            target=target_f,
            source=teacher_f,
            reverse_k=self.zero_degree_repair_k,
        )
        self._sync(target_f.device)
        timing["initial_graph_seconds"] = time.perf_counter() - graph_start

        sinkhorn_start = time.perf_counter()
        self._sync(target_f.device)
        k16_result = None
        k16_valid_graph = bool((incoming16_after > 0).all().item())
        if k16_valid_graph:
            k16_result = self.sinkhorn.solve(
                src16, dst16, cost16, num_points, target_f.shape[0]
            )
        self._sync(target_f.device)
        timing["k16_sinkhorn_seconds"] = time.perf_counter() - sinkhorn_start
        k16_error = max(
            k16_result.row_error if k16_result is not None else float("inf"),
            k16_result.col_error if k16_result is not None else float("inf"),
        )
        strict_accept = k16_result is not None and k16_error <= self.marginal_accept_threshold

        active_k = self.initial_k
        selected_src, selected_dst, selected_cost = src16, dst16, cost16
        selected_result = k16_result
        repair_edges = repair_edges16
        k32_result = None
        k32_repair_edges = 0
        bad_column_repair_edges = 0
        fallback_reason = ""

        if not strict_accept:
            active_k = self.k_max
            k32_graph_start = time.perf_counter()
            self._sync(target_f.device)
            (
                src32,
                dst32,
                cost32,
                incoming32_before,
                incoming32_after,
                k32_repair_edges,
            ) = self._make_graph(
                knn32,
                active_k=self.k_max,
                d_ref=d_ref,
                num_src=num_points,
                num_dst=target_f.shape[0],
                target=target_f,
                source=teacher_f,
                reverse_k=self.bad_column_repair_k,
            )
            self._sync(target_f.device)
            timing["k32_graph_seconds"] = time.perf_counter() - k32_graph_start
            k32_sinkhorn_start = time.perf_counter()
            self._sync(target_f.device)
            if bool((incoming32_after > 0).all().item()):
                k32_result = self.sinkhorn.solve(
                    src32, dst32, cost32, num_points, target_f.shape[0]
                )
            self._sync(target_f.device)
            timing["k32_sinkhorn_seconds"] = time.perf_counter() - k32_sinkhorn_start

            selected_src, selected_dst, selected_cost = src32, dst32, cost32
            selected_result = k32_result
            repair_edges = repair_edges16 + k32_repair_edges
            k32_error = max(
                k32_result.row_error if k32_result is not None else float("inf"),
                k32_result.col_error if k32_result is not None else float("inf"),
            )
            if k32_result is not None and k32_error > self.marginal_failure_threshold:
                bad_columns = torch.nonzero(
                    (k32_result.col_mass - 1.0).abs() > self.marginal_failure_threshold,
                    as_tuple=False,
                ).flatten()
                bad_start = time.perf_counter()
                self._sync(target_f.device)
                repair_src, repair_dst, repair_cost = self._reverse_edges(
                    target_f,
                    teacher_f,
                    bad_columns,
                    self.bad_column_repair_k,
                    d_ref,
                )
                self._sync(target_f.device)
                timing["bad_column_repair_knn_seconds"] = time.perf_counter() - bad_start
                bad_column_repair_edges = int(repair_src.numel())
                if bad_column_repair_edges:
                    selected_src = torch.cat([selected_src, repair_src], dim=0)
                    selected_dst = torch.cat([selected_dst, repair_dst], dim=0)
                    selected_cost = torch.cat([selected_cost, repair_cost], dim=0)
                    selected_src, selected_dst, selected_cost = coalesce_edges(
                        selected_src,
                        selected_dst,
                        selected_cost,
                        target_f.shape[0],
                    )
                    retry_start = time.perf_counter()
                    self._sync(target_f.device)
                    if bool(
                        (self._coverage(selected_dst, target_f.shape[0]) > 0).all().item()
                    ):
                        selected_result = self.sinkhorn.solve(
                            selected_src,
                            selected_dst,
                            selected_cost,
                            num_points,
                            target_f.shape[0],
                        )
                    else:
                        selected_result = None
                    self._sync(target_f.device)
                    timing["bad_column_repair_sinkhorn_seconds"] = (
                        time.perf_counter() - retry_start
                    )

        final_error = max(
            selected_result.row_error if selected_result is not None else float("inf"),
            selected_result.col_error if selected_result is not None else float("inf"),
        )
        accepted = selected_result is not None and final_error <= self.marginal_failure_threshold
        use_invalid_plan = (
            selected_result is not None
            and not accepted
            and not self.fallback_on_invalid
        )
        use_ot_plan = bool(accepted or use_invalid_plan)
        if not use_ot_plan:
            fallback_reason = "marginal_error_or_zero_degree"
            refined = teacher_endpoint
        else:
            if use_invalid_plan:
                fallback_reason = "marginal_error_allowed_raw_plan"
            bary_start = time.perf_counter()
            self._sync(target_f.device)
            refined = self._barycentric(
                source_f,
                target_f,
                selected_src,
                selected_dst,
                selected_result.plan,
                self.barycentric_alpha,
            ).to(dtype=original_dtype)
            self._sync(target_f.device)
            timing["barycentric_seconds"] = time.perf_counter() - bary_start

        self._sync(target_f.device)
        timing["total_ot_seconds"] = time.perf_counter() - total_start
        displacement = (refined.float() - teacher_f).norm(dim=1)
        normalized_displacement = displacement / float(d_ref)
        incoming_initial_ratio = float(
            (incoming16_before == 0).float().mean().item()
        )
        incoming_after_ratio = float(
            (incoming16_after == 0).float().mean().item()
        )
        # Per-row/column marginal diagnostics (report 5.6 / 8.B): the scalar
        # row/col errors above are global mean absolute errors and can hide a
        # small number of badly unsatisfied marginals.
        if selected_result is not None:
            row_abs_err = (selected_result.row_mass - 1.0).abs()
            col_abs_err = (selected_result.col_mass - 1.0).abs()
            max_row_error = float(row_abs_err.max().item())
            max_col_error = float(col_abs_err.max().item())
            bad_row_ratio = float(
                (row_abs_err > self.marginal_failure_threshold).float().mean().item()
            )
            bad_col_ratio = float(
                (col_abs_err > self.marginal_failure_threshold).float().mean().item()
            )
            plan_finite = bool(torch.isfinite(selected_result.plan).all().item())
            row_mass_min = float(selected_result.row_mass.min().item())
        else:
            max_row_error = float("inf")
            max_col_error = float("inf")
            bad_row_ratio = 1.0
            bad_col_ratio = 1.0
            plan_finite = False
            row_mass_min = 0.0
        result = {
            "refiner": "sparse_global_balanced_ot",
            "accepted": bool(accepted),
            "used_invalid_plan": bool(use_invalid_plan),
            "strict_accept": bool(strict_accept),
            "fallback": not use_ot_plan,
            "fallback_reason": fallback_reason,
            "num_points": int(num_points),
            "d_ref": float(d_ref),
            "active_k": int(active_k),
            "edge_count": int(selected_src.numel()),
            "k16_edge_count": int(src16.numel()),
            "k16_repair_edge_count": int(repair_edges16),
            "k32_repair_edge_count": int(k32_repair_edges),
            "bad_column_repair_edge_count": int(bad_column_repair_edges),
            "gt_zero_degree_ratio_k16_before_repair": incoming_initial_ratio,
            "gt_zero_degree_ratio_k16_after_repair": incoming_after_ratio,
            "k16_row_marginal_error": float(
                k16_result.row_error if k16_result is not None else float("inf")
            ),
            "k16_col_marginal_error": float(
                k16_result.col_error if k16_result is not None else float("inf")
            ),
            "row_marginal_error": float(
                selected_result.row_error if selected_result is not None else float("inf")
            ),
            "col_marginal_error": float(
                selected_result.col_error if selected_result is not None else float("inf")
            ),
            "max_row_marginal_error": max_row_error,
            "max_col_marginal_error": max_col_error,
            "bad_row_ratio": bad_row_ratio,
            "bad_col_ratio": bad_col_ratio,
            "plan_all_finite": plan_finite,
            "row_mass_min": row_mass_min,
            "max_row_probability": float(
                selected_result.max_row_probability
                if selected_result is not None
                else 0.0
            ),
            "mean_row_entropy": float(
                selected_result.mean_row_entropy if selected_result is not None else 0.0
            ),
            "temperature": self.temperature,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "barycentric_alpha": self.barycentric_alpha,
            **_percentile_stats(displacement, "teacher_to_ot_displacement_"),
            **_percentile_stats(normalized_displacement, "normalized_teacher_to_ot_displacement_"),
            **timing,
        }
        return refined, result

    @torch.no_grad()
    def spacing_statistics(self, points: torch.Tensor, *, seed: int = 0):
        """Return 8-NN spacing samples/statistics for one full point cloud."""
        count = min(self.spacing_query_count, points.shape[0])
        generator = torch.Generator(device=points.device)
        generator.manual_seed(int(seed))
        query_indices = torch.randperm(
            points.shape[0], device=points.device, generator=generator
        )[:count]
        knn = self.knn.query(points[query_indices], points, self.spacing_k + 1)
        distances = torch.where(
            knn.indices == query_indices[:, None], torch.inf, knn.dist2
        )
        distances, _ = torch.sort(distances, dim=1)
        spacing = torch.sqrt(distances[:, self.spacing_k - 1].clamp_min(0.0))
        q = torch.tensor(
            [0.10, 0.25, 0.50, 0.75, 0.90],
            device=spacing.device,
            dtype=spacing.dtype,
        )
        quantiles = torch.quantile(spacing, q)
        mean = spacing.mean()
        std = spacing.std(unbiased=False)
        return {
            "mean": float(mean.item()),
            "std": float(std.item()),
            "cv": float((std / mean.clamp_min(1e-12)).item()),
            "p10": float(quantiles[0].item()),
            "p25": float(quantiles[1].item()),
            "p50": float(quantiles[2].item()),
            "p75": float(quantiles[3].item()),
            "p90": float(quantiles[4].item()),
            "samples": spacing,
        }

    @torch.no_grad()
    def chamfer_statistics(self, points_a: torch.Tensor, points_b: torch.Tensor):
        """Compute one-sided and symmetric Chamfer with global KeOps KNN."""
        a_to_b = self.knn.query(points_a.float(), points_b.float(), 1)
        b_to_a = self.knn.query(points_b.float(), points_a.float(), 1)
        a_to_b_value = torch.sqrt(a_to_b.dist2.clamp_min(0.0)).mean()
        b_to_a_value = torch.sqrt(b_to_a.dist2.clamp_min(0.0)).mean()
        return {
            "a_to_b": float(a_to_b_value.item()),
            "b_to_a": float(b_to_a_value.item()),
            "symmetric": float(0.5 * (a_to_b_value + b_to_a_value).item()),
        }

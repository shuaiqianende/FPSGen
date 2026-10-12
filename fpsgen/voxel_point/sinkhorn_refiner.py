"""Two explicit acceptance policies around FPSGen's sparse log-domain OT."""

import time

import torch

from fpsgen.ops.endpoint_refinement.sparse_global_ot import _sparse_knn_graph
from fpsgen.ops.endpoint_refinement.sinkhorn import sparse_sinkhorn
from .metrics import point_metrics


def refine_endpoint(endpoint, target, *, k, epsilon, iterations, alpha, backend):
    """Reuse FPSGen graph/solver; expose actual edge-plan coverage diagnostics."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    source, target_id, cost = _sparse_knn_graph(endpoint, target, k, backend)
    plan = sparse_sinkhorn(source, target_id, cost.float(), len(endpoint), len(target),epsilon,iterations)
    row = endpoint.new_zeros(len(endpoint))
    col = target.new_zeros(len(target))
    row.scatter_add_(0,source,plan)
    col.scatter_add_(0,target_id,plan)
    barycenter = torch.zeros_like(endpoint)
    barycenter.index_add_(0,source,plan[:,None]*target[target_id])
    barycenter /= row[:,None].clamp_min(torch.finfo(torch.float32).tiny)
    row_error, col_error = (row-1/len(endpoint)).abs(), (col-1/len(target)).abs()
    return endpoint+alpha*(barycenter-endpoint), {
        "row_error":float(row_error.max()), "col_error":float(col_error.max()),
        "row_error_mean":float(row_error.mean()), "col_error_mean":float(col_error.mean()),
        "bad_row_ratio":float((row_error > .01/len(endpoint)).float().mean()),
        "bad_col_ratio":float((col_error > .01/len(target)).float().mean()),
        "entropy":float(-(plan*plan.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()),
        "num_edges":len(plan), "edge_plan_finite":bool(torch.isfinite(plan).all()),
        "zero_row_ratio":float((row <= 0).float().mean()), "zero_column_ratio":float((col <= 0).float().mean()),
        "graph_every_source_target_covered":True,
        "plan_every_source_target_covered":bool((row > 0).all() and (col > 0).all())}


@torch.no_grad()
def refine_with_policy(teacher_endpoint, target, spec, *, mode="sinkhorn_guided_geometry",
                       k=16, epsilon=.01, iterations=100, alpha=1., backend="keops",
                       marginal_relative_tolerance=.01, coverage_drop_tolerance=.01,
                       spacing_min_ratio=.8, spacing_cv_max_ratio=1.25,
                       max_displacement=10., reset_peak_memory=True):
    if mode not in {"strict_balanced_ot", "sinkhorn_guided_geometry"}:
        raise ValueError("Unknown OT acceptance mode")
    teacher, target = teacher_endpoint.float(), target.float()
    if teacher.shape != target.shape or teacher.shape[1:] != (3,):
        raise ValueError("Equal source/target XYZ counts required")
    if not torch.isfinite(teacher).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite Teacher/target; cannot construct fallback")
    if teacher.is_cuda:
        torch.cuda.synchronize(teacher.device)
        if reset_peak_memory:
            torch.cuda.reset_peak_memory_stats(teacher.device)
    start = time.perf_counter()
    before = point_metrics(teacher, target, backend=backend, spec=spec)
    diagnostics = {"mode": mode, "teacher": before, "balanced_ot_converged": False,
                   "geometry_refinement_accepted": False, "fallback_used": True,
                   "plan_finite": False, "every_source_target_covered": False,
                   "no_dense_nxn": True}
    refined = teacher
    try:
        candidate, transport = refine_endpoint(teacher, target, k=k, epsilon=epsilon,
                iterations=iterations, alpha=alpha, backend=backend)
        diagnostics.update(transport)
        finite = transport.get("edge_plan_finite", True) and bool(torch.isfinite(candidate).all()) and all(
            torch.isfinite(torch.tensor(transport[name])) for name in ("row_error", "col_error", "entropy"))
        diagnostics["plan_finite"] = finite
        # The reused sparse solver checks every source and target before solving.
        diagnostics["every_source_target_covered"] = transport.get("plan_every_source_target_covered", False)
        row_relative = transport["row_error"] * len(teacher)
        column_relative = transport["col_error"] * len(target)
        diagnostics.update(row_relative_error=row_relative, column_relative_error=column_relative)
        balanced = (finite and diagnostics["every_source_target_covered"]
            and row_relative <= marginal_relative_tolerance and column_relative <= marginal_relative_tolerance)
        diagnostics["balanced_ot_converged"] = balanced
        if finite:
            after = point_metrics(candidate, target, backend=backend, spec=spec)
            diagnostics["candidate"] = after
            displacement = float((candidate - teacher).norm(dim=-1).max())
            diagnostics["max_refinement_displacement"] = displacement
            geometry_ok = (after["cd"] < before["cd"]
                and after["coverage_0.1"] >= before["coverage_0.1"] - coverage_drop_tolerance
                and after["spacing_mean"] >= before["spacing_mean"] * spacing_min_ratio
                and after["spacing_cv"] <= max(before["spacing_cv"], 1e-6) * spacing_cv_max_ratio
                and after["out_of_range_ratio"] <= before["out_of_range_ratio"] + .001
                and displacement <= max_displacement)
            accepted = geometry_ok and (balanced or mode == "sinkhorn_guided_geometry")
            diagnostics["geometry_refinement_accepted"] = accepted
            diagnostics["fallback_used"] = not accepted
            diagnostics["reason"] = "accepted" if accepted else "marginal or geometry gate failed"
            if accepted:
                refined = candidate
        else:
            diagnostics["reason"] = "nonfinite plan/endpoint"
    except (ValueError, FloatingPointError) as exc:
        diagnostics["reason"] = str(exc)
    if teacher.is_cuda:
        torch.cuda.synchronize(teacher.device)
        diagnostics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(teacher.device)
    diagnostics["runtime_seconds"] = time.perf_counter() - start
    return refined, diagnostics

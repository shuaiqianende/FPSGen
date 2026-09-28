"""K-nearest-neighbour backends that never materialize a full cost matrix."""

from __future__ import annotations

import torch


def _validate(points: torch.Tensor, name: str) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {tuple(points.shape)}")
    if not points.is_floating_point():
        raise TypeError(f"{name} must be floating point")


@torch.no_grad()
def _keops_knn(query: torch.Tensor, support: torch.Tensor, k: int):
    try:
        from pykeops.torch import LazyTensor
    except ImportError as exc:
        raise RuntimeError(
            "PyKeOps is required for --knn-backend keops. Install the project "
            "environment before running large endpoint diagnostics."
        ) from exc

    # LazyTensor keeps the N x M distance expression symbolic; only N x K
    # distances and indices are materialized by Kmin_argKmin.
    q_i = LazyTensor(query[:, None, :])
    s_j = LazyTensor(support[None, :, :])
    squared_distance = ((q_i - s_j) ** 2).sum(-1)
    return squared_distance.Kmin_argKmin(k, dim=1)


@torch.no_grad()
def _scipy_knn(query: torch.Tensor, support: torch.Tensor, k: int):
    """CPU diagnostic fallback based on a KD-tree (also not dense)."""
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("SciPy is required for --knn-backend scipy") from exc
    tree = cKDTree(support.detach().cpu().numpy())
    distance, indices = tree.query(query.detach().cpu().numpy(), k=k, workers=-1)
    if k == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    d2 = torch.as_tensor(np.square(distance), device=query.device, dtype=query.dtype)
    idx = torch.as_tensor(indices, device=query.device, dtype=torch.long)
    return d2, idx


@torch.no_grad()
def knn(query: torch.Tensor, support: torch.Tensor, k: int,
        backend: str = "keops"):
    """Return squared distances and support indices with shapes ``[N, K]``.

    ``keops`` is the intended GPU backend. ``scipy`` is provided only for
    small CPU/offline diagnostics; both avoid an ``N x M`` allocation.
    """
    _validate(query, "query")
    _validate(support, "support")
    if query.device != support.device:
        raise ValueError("query and support must be on the same device")
    if not 1 <= k <= support.shape[0]:
        raise ValueError(f"k must be in [1, {support.shape[0]}], got {k}")
    if backend == "keops":
        return _keops_knn(query, support, k)
    if backend == "scipy":
        return _scipy_knn(query, support, k)
    raise ValueError(f"Unknown KNN backend {backend!r}; choose 'keops' or 'scipy'")

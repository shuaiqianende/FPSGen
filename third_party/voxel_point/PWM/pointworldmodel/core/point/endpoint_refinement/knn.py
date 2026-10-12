"""Global KNN primitives backed by PyKeOps LazyTensor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pykeops.torch import LazyTensor


@dataclass
class KNNResult:
    """Squared distances and reference indices for each query row."""

    dist2: torch.Tensor
    indices: torch.Tensor


class KeOpsGlobalKNN:
    """Exact global KNN without materialising a query-by-reference matrix."""

    def __init__(self, max_k: int = 32):
        if int(max_k) < 1:
            raise ValueError("max_k must be positive")
        self.max_k = int(max_k)

    @torch.no_grad()
    def query(self, queries: torch.Tensor, references: torch.Tensor, k: int) -> KNNResult:
        if queries.ndim != 2 or references.ndim != 2:
            raise ValueError("queries and references must have shape [N, D]")
        if queries.shape[1] != references.shape[1]:
            raise ValueError("queries and references must have the same feature dimension")
        if queries.shape[0] == 0 or references.shape[0] == 0:
            raise ValueError("KNN inputs must be non-empty")
        k = int(k)
        if k < 1 or k > self.max_k:
            raise ValueError(f"k must be in [1, {self.max_k}]")
        if k > references.shape[0]:
            raise ValueError("k cannot exceed the number of references")
        if queries.device != references.device:
            raise ValueError("queries and references must be on the same device")

        # Force float32 for stable squared-distance accumulation and make the
        # operation explicit: no torch.cdist or dense NxN tensor is allowed.
        queries = queries.float().contiguous()
        references = references.float().contiguous()
        query_i = LazyTensor(queries[:, None, :])
        reference_j = LazyTensor(references[None, :, :])
        distance2 = ((query_i - reference_j) ** 2).sum(-1)
        dist2, indices = distance2.Kmin_argKmin(k, axis=1)
        return KNNResult(
            dist2=dist2.float().contiguous(),
            indices=indices.long().contiguous(),
        )

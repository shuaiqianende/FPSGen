"""Sparse, training-only endpoint refinement for FPSGen teacher pairs."""

from .knn import knn
from .sparse_global_ot import refine_endpoint

__all__ = ["knn", "refine_endpoint"]

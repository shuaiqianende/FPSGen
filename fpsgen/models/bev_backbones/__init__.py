"""Dense Stage-1 BEV backbone implementations.

Backbones are intentionally selected through :func:`build_bev_backbone` so a
legacy PyTorch-1.13 installation never imports the PyTorch-2-only PixelU code.
"""

from .factory import build_bev_backbone

__all__ = ["build_bev_backbone"]

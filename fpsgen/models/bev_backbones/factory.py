"""Lazy factory for interchangeable Stage-1 BEV velocity backbones."""

from __future__ import annotations


def _model_config(cfg):
    """Support both plain mappings and Lightning's hyperparameter mapping."""
    return cfg.get("model", {})


def build_bev_backbone(cfg):
    """Build the configured BEV backbone without eagerly importing PixelU.

    Historical checkpoints do not contain ``model.backbone`` and therefore
    deterministically select ``legacy``.
    """
    model_cfg = _model_config(cfg)
    name = str(model_cfg.get("backbone", "legacy")).lower()
    if name == "legacy":
        from fpsgen.models.image_flow_net import BEVFlowTransNet
        return BEVFlowTransNet(base_ch=32, time_dim=256, cls=0, layout_ch=2)
    if name == "dic_s":
        from .dic_bev import BEVDiCS
        return BEVDiCS(model_cfg)
    if name == "pixelu_s":
        from .pixelu_bev import BEVPixelUS
        return BEVPixelUS(model_cfg)
    if name == "hdit_s":
        from .hdit_bev import BEVHDiTS
        return BEVHDiTS(model_cfg)
    raise ValueError(
        f"Unknown BEV backbone {name!r}; expected one of "
        "'legacy', 'dic_s', 'pixelu_s', or 'hdit_s'."
    )

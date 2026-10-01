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
    if name == "dip_s":
        from .dip_bev import BEVDiPS
        return BEVDiPS(model_cfg)
    if name == "ncsnpp_s":
        from .ncsnpp_bev import BEVNCSNppS
        return BEVNCSNppS(model_cfg)
    if name == "sid2_s":
        from .sid2_bev import BEVSiD2S
        return BEVSiD2S(model_cfg)
    if name == "pixeldit_s":
        from .pixeldit_bev import BEVPixelDiTS
        return BEVPixelDiTS(model_cfg)
    if name in {"synflow_bev_s", "unet_generic_bev_s"}:
        from .synflow_bev import BEVSynFlowS
        return BEVSynFlowS(model_cfg)
    if name == "cracksegflow_bev_s":
        from .cracksegflow_bev import BEVCrackSegFlowS
        return BEVCrackSegFlowS(model_cfg)
    if name in {"pixeldit_generic_bev_s", "pixelcontrol_bev_s"}:
        from .pixelcontrol_bev import BEVPixelControlS
        return BEVPixelControlS(model_cfg)
    raise ValueError(
        f"Unknown BEV backbone {name!r}; expected one of "
        "'legacy', 'dic_s', 'pixelu_s', 'hdit_s', 'dip_s', 'ncsnpp_s', "
        "'sid2_s', 'pixeldit_s', 'unet_generic_bev_s', 'synflow_bev_s', "
        "'cracksegflow_bev_s', 'pixeldit_generic_bev_s', or 'pixelcontrol_bev_s'."
    )

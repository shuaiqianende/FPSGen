"""Topology and lazy-import guards for Stage-1 backbone preparation."""

import sys
import types
import yaml
import pytest

from fpsgen.models.bev_backbones.dic_core import DiCCore


def test_dic_s_architecture_guard():
    core = DiCCore(input_size=32)
    assert core.hidden_size == 96
    assert core.depth == (6, 6, 5, 6, 6)
    assert core.mult_channels == (1, 2, 4, 2, 1)
    assert core.skip_stride == 3
    assert len(core.enc_blocks[0]) == len(core.enc_blocks[1]) == 6
    assert len(core.lat_blocks) == 5
    assert len(core.dec_blocks[0]) == len(core.dec_blocks[1]) == 6


def test_pixelu_s_config_topology_guard():
    cfg = yaml.safe_load(open("configs/research_v2/train_bev_pixelu_s_gt_possion.yaml"))
    pixelu = cfg["model"]["pixelu"]
    assert pixelu["patch_size"] == 16
    assert pixelu["depth"] == [4, 4, 4]
    assert pixelu["hidden_size"] == 384
    assert pixelu["num_heads"] == 6
    assert pixelu["bottleneck_dim"] == 64
    assert pixelu["skip"] is True and pixelu["skip_all_blocks"] is True


def test_legacy_factory_does_not_import_pixelu(monkeypatch):
    sys.modules.pop("fpsgen.models.bev_backbones.pixelu_core", None)
    sys.modules.pop("fpsgen.models.bev_backbones.pixelu_bev", None)
    # This is an import-boundary test, not a PyKeOps environment test.  Stub
    # the legacy module so CPU-only CI without PyKeOps can still prove that
    # factory's historical branch never imports PixelU.
    legacy_module = types.ModuleType("fpsgen.models.image_flow_net")
    class BEVFlowTransNet:  # noqa: N801 - preserve historical class name
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    legacy_module.BEVFlowTransNet = BEVFlowTransNet
    monkeypatch.setitem(sys.modules, "fpsgen.models.image_flow_net", legacy_module)
    from fpsgen.models.bev_backbones.factory import build_bev_backbone
    model = build_bev_backbone({"model": {}})
    assert model.__class__.__name__ == "BEVFlowTransNet"
    assert "fpsgen.models.bev_backbones.pixelu_core" not in sys.modules
    assert "fpsgen.models.bev_backbones.pixelu_bev" not in sys.modules


@pytest.mark.skipif(not hasattr(__import__("torch").nn.functional, "scaled_dot_product_attention"),
                    reason="PixelU requires the PyTorch-2 SDPA API")
def test_pixelu_s_architecture_guard():
    from fpsgen.models.bev_backbones.pixelu_core import UiTCore, RMSNorm, SwiGLUFFN
    core = UiTCore(input_size=64, hidden_size=96, num_heads=6, bottleneck_dim=16)
    assert core.depth == (4, 4, 4)
    assert core.patch_size == 16
    assert len(core.blocks0) == len(core.blocks1) == len(core.blocks2) == 4
    assert isinstance(core.blocks0[0].norm1, RMSNorm)
    assert isinstance(core.blocks0[0].mlp, SwiGLUFFN)

"""Topology and lazy-import guards for Stage-1 backbone preparation."""

import sys
import types
from pathlib import Path
import yaml
import pytest

from fpsgen.models.bev_backbones.dic_core import DiCCore
from fpsgen.models.bev_backbones.hdit_core import HDiTCore, TokenMerge, TokenSplit


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
    sys.modules.pop("fpsgen.models.bev_backbones.hdit_core", None)
    sys.modules.pop("fpsgen.models.bev_backbones.hdit_bev", None)
    sys.modules.pop("fpsgen.models.bev_backbones.dip_core", None)
    sys.modules.pop("fpsgen.models.bev_backbones.dip_bev", None)
    sys.modules.pop("fpsgen.models.bev_backbones.ncsnpp_core", None)
    sys.modules.pop("fpsgen.models.bev_backbones.ncsnpp_bev", None)
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
    assert "fpsgen.models.bev_backbones.hdit_core" not in sys.modules
    assert "fpsgen.models.bev_backbones.dip_core" not in sys.modules
    assert "fpsgen.models.bev_backbones.ncsnpp_core" not in sys.modules


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


def test_hdit_s_config_topology_guard():
    cfg = yaml.safe_load(open("configs/research_v2/train_bev_hdit_s_gt_possion.yaml"))
    hdit = cfg["model"]["hdit"]
    assert hdit["patch_size"] == [4, 4]
    assert hdit["widths"] == [128, 256, 512]
    assert hdit["depths"] == [2, 2, 4]
    assert hdit["d_ffs"] == [384, 768, 1536]
    assert [layer["type"] for layer in hdit["self_attns"]] == ["shifted_window", "shifted_window", "global"]
    assert hdit["self_attns"][0]["window_size"] == hdit["self_attns"][1]["window_size"] == 8


def test_hdit_s_core_topology_guard():
    core = HDiTCore(input_size=64, patch_size=4, widths=(128, 256, 512), depths=(2, 2, 4),
                    d_ffs=(384, 768, 1536), mapping_width=256, mapping_d_ff=768)
    assert len(core.enc0) == len(core.dec0) == 2
    assert len(core.enc1) == len(core.dec1) == 2
    assert len(core.middle) == 4
    assert isinstance(core.merge0, TokenMerge) and isinstance(core.merge1, TokenMerge)
    assert isinstance(core.split0, TokenSplit) and isinstance(core.split1, TokenSplit)


@pytest.mark.skipif(not hasattr(__import__("torch").nn.functional, "scaled_dot_product_attention"),
                    reason="DiP requires the PyTorch-2 SDPA API")
def test_dip_s_architecture_guard():
    from fpsgen.models.bev_backbones.dip_core import DiPCore, FlattenDiTBlock, LocalDetailer
    cfg = yaml.safe_load(open("configs/research_v2/train_bev_dip_s_gt_possion.yaml"))
    dip = cfg["model"]["dip"]
    assert dip["patch_size"] == 16 and dip["hidden_size"] == 384 and dip["num_groups"] == 6
    assert dip["hidden_size"] // dip["num_groups"] == 64
    assert dip["local_channels"] == [64, 128, 256, 512]
    core = DiPCore(input_size=64, patch_size=16, hidden_size=192, num_groups=3, num_cond_blocks=2)
    assert isinstance(core.blocks[0], FlattenDiTBlock)
    assert isinstance(core.detailer, LocalDetailer)
    assert core.detailer.patch_size == 16


def test_ncsnpp_s_architecture_guard():
    from fpsgen.models.bev_backbones.ncsnpp_core import AttnBlockpp, BigGANResBlock, FIRResample, NCSNppCore
    cfg = yaml.safe_load(open("configs/research_v2/train_bev_ncsnpp_s_gt_possion.yaml"))
    ncsn = cfg["model"]["ncsnpp"]
    assert ncsn["ch_mult"] == [1, 1, 2, 2, 2, 2, 2]
    assert ncsn["num_res_blocks"] == 2 and ncsn["attn_resolutions"] == [16]
    assert ncsn["resblock_type"] == "biggan" and ncsn["fir"] is True
    assert ncsn["progressive"] == "output_skip" and ncsn["progressive_input"] == "input_skip"
    core = NCSNppCore(input_size=64, nf=16)
    assert len(core.enc_blocks) == len(core.dec_blocks) == 7
    assert all(len(stage) == 2 for stage in core.enc_blocks)
    assert isinstance(core.enc_blocks[0][0], BigGANResBlock)
    assert isinstance(core.downsamplers[0], FIRResample)
    assert any(isinstance(module, AttnBlockpp) for stage in core.enc_attn for module in stage)


def test_sid2_s_topology_guard():
    from fpsgen.models.bev_backbones.sid2_core import AdaResBlock, AdaTransformerBlock, SiD2Core
    cfg = yaml.safe_load(open("configs/research_v2/train_bev_sid2_s_gt_possion.yaml"))
    sid2 = cfg["model"]["sid2"]
    assert sid2["patch_size"] == 2 and sid2["channels"] == [64, 128, 256, 384]
    assert sid2["num_updown_blocks"] == [3, 3, 3] and sid2["num_mid_blocks"] == 16
    assert sid2["block_types"] == ["resblock", "resblock", "transformer", "transformer"]
    core = SiD2Core(input_size=64, channels=(16, 32, 64, 64), num_mid_blocks=2, head_dim=16, time_dim=32)
    assert len(core.enc0) == len(core.enc1) == len(core.enc2) == 3
    assert len(core.dec0) == len(core.dec1) == len(core.dec2) == 3 and len(core.mid) == 2
    assert isinstance(core.enc0[0], AdaResBlock) and isinstance(core.enc2[0], AdaTransformerBlock)
    # Residual U-ViT must not retain traditional per-block concatenative skips.
    assert not any("skip" in name.lower() for name, _ in core.named_modules())


@pytest.mark.skipif(not hasattr(__import__("torch").nn.functional, "scaled_dot_product_attention"),
                    reason="PixelDiT requires the PyTorch-2 SDPA API")
def test_pixeldit_s_topology_guard():
    from fpsgen.models.bev_backbones.pixeldit_core import AugmentedDiTBlock, PiTBlock, PixelDiTCore
    cfg = yaml.safe_load(open("configs/research_v2/train_bev_pixeldit_s_gt_possion.yaml"))
    pixeldit = cfg["model"]["pixeldit"]
    assert pixeldit["patch_size"] == 16 and pixeldit["hidden_size"] == 384 and pixeldit["num_groups"] == 6
    assert pixeldit["patch_depth"] == 8 and pixeldit["pixel_hidden_size"] == 8 and pixeldit["pixel_depth"] == 4
    assert pixeldit["pit_adaln_post_modulation"] is True and pixeldit["time_scale"] == 1.0 and pixeldit["repa"] is False
    core = PixelDiTCore(input_size=64, hidden_size=128, num_groups=2, patch_depth=2, pixel_hidden_size=8, pixel_depth=2)
    assert isinstance(core.patch_blocks[0], AugmentedDiTBlock) and isinstance(core.pixel_blocks[0], PiTBlock)
    assert len(core.patch_blocks) == 2 and len(core.pixel_blocks) == 2
    assert core.pixel_blocks[0].norm1.__class__.__name__ == "RMSNorm"
    assert core.pixel_blocks[0].attn.__class__.__name__ == "Attention"
    assert core.pixel_blocks[0].pixel_mlp.__class__.__name__ == "SwiGLUFFN"


def test_new_pixel_backbones_do_not_depend_on_vae_or_latents():
    root = Path("fpsgen/models/bev_backbones")
    for name in ("hdit_core.py", "hdit_bev.py", "dip_core.py", "dip_bev.py", "ncsnpp_core.py", "ncsnpp_bev.py", "sid2_core.py", "sid2_bev.py", "pixeldit_core.py", "pixeldit_bev.py"):
        source = (root / name).read_text().lower()
        assert "autoencoderkl" not in source
        assert "import vae" not in source


def test_spatial_control_config_topology_guards():
    syn = yaml.safe_load(open("configs/research_v2/spatial_control/synflow_bev_s_b8.yaml"))
    crack = yaml.safe_load(open("configs/research_v2/spatial_control/cracksegflow_bev_s_b8.yaml"))
    pixel = yaml.safe_load(open("configs/research_v2/spatial_control/pixelcontrol_bev_s_b8.yaml"))
    assert syn["model"]["synflow"]["channel_mult"] == [1, 1, 2, 2, 4, 4]
    assert syn["model"]["condition"]["injection"] == {"type": "spade", "encoder": True, "middle": True, "decoder": True}
    assert crack["model"]["condition"]["injection"] == {"type": "spade", "encoder": False, "middle": False, "decoder": True}
    assert pixel["model"]["pixeldit"]["patch_size"] == 16
    assert pixel["model"]["pixelcontrol"]["inject_every"] == 1
    assert pixel["model"]["pixelcontrol"]["zero_proj"] is True


def test_ncsnpp_spade_config_keeps_the_historical_core_topology():
    n0 = yaml.safe_load(open("configs/research_v2/ncsnpp_spade/n0_generic_seed42.yaml"))
    n1 = yaml.safe_load(open("configs/research_v2/ncsnpp_spade/n1_spade_seed42.yaml"))
    assert n0["model"]["ncsnpp"] == n1["model"]["ncsnpp"]
    assert n0["model"]["condition"]["fusion"] == n1["model"]["condition"]["fusion"] == "shared"
    assert n1["model"]["condition"]["spade"] == {"enabled": True, "cond_dim": 8, "boundary_gate": False}

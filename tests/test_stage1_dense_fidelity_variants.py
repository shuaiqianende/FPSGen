from torch import nn
import torch
import yaml
from pathlib import Path

from fpsgen.hier_occ_flow.fidelity_dense_blocks import (
    AsymmetricFactorized3D, LightweightSkip3D, replace_fidelity_stage0,
)
from fpsgen.hier_occ_flow.dense_encoder_variants import forward_variant_backbone


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, *, padding=1,
                 norm="bn", activation=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)
        self.norm_name = "bn" if norm == "bn" else "gn"
        self.add_module(self.norm_name, nn.BatchNorm3d(out_channels)
                        if norm == "bn" else nn.GroupNorm(32, out_channels))
        self.activate = nn.ReLU(inplace=False) if activation else nn.Identity()

    def forward(self, x):
        return self.activate(getattr(self, self.norm_name)(self.conv(x)))


class OriginalBlock(nn.Module):
    def __init__(self, skip_norm="bn", projected=True):
        super().__init__()
        self.conv1 = ConvModule(128, 128, activation=True)
        self.conv2 = ConvModule(128, 128)
        self.downsample = ConvModule(128, 128, norm=skip_norm) if projected else None
        self.drop_path = nn.Identity()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.drop_path(self.conv2(self.conv1(x))) + identity)


class Backbone(nn.Module):
    def __init__(self, skip_norm="bn"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(OriginalBlock(skip_norm), OriginalBlock(projected=False)),
            nn.Sequential(nn.Identity()),
        )
        self.backbone_output_ids = (0, 1)


def test_asymmetric3d_preserves_dense_shape_and_has_finite_gradients():
    block = AsymmetricFactorized3D(OriginalBlock(), rank=64)
    x = torch.randn(1, 128, 5, 5, 3, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    for name in ("reduce.weight", "spatial.weight", "expand.weight"):
        assert torch.isfinite(dict(block.named_parameters())[name].grad).all()


def test_lightweight_skip_uses_pointwise_projection_and_retains_norm_type():
    for norm in ("bn", "gn"):
        original = OriginalBlock(skip_norm=norm)
        block = LightweightSkip3D(original)
        assert block.skip_conv.kernel_size == (1, 1, 1)
        assert isinstance(block.skip_norm, nn.BatchNorm3d if norm == "bn" else nn.GroupNorm)
        x = torch.randn(1, 128, 5, 5, 3, requires_grad=True)
        y = block(x)
        y.mean().backward()
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert torch.isfinite(block.skip_conv.weight.grad).all()


def test_fidelity_replacement_is_stage0_local_and_runner_handles_mixed_blocks():
    backbone = Backbone(skip_norm="gn")
    original_regular = backbone.layers[0][1]
    assert replace_fidelity_stage0(backbone, "lightweight_skip") == 1
    x = torch.randn(1, 128, 5, 5, 3)
    outputs = forward_variant_backbone(backbone, x, None)
    assert outputs[0].shape == x.shape and outputs[1].shape == x.shape
    assert backbone.layers[0][1] is original_regular
    assert isinstance(backbone.layers[0][0], LightweightSkip3D)


def test_bottleneck96_and_asymmetric_replace_all_stage0_blocks():
    from fpsgen.hier_occ_flow.dense_encoder_variants import DenseBottleneck3D
    for variant in ("bottleneck96", "asymmetric3d"):
        backbone = Backbone()
        count = replace_fidelity_stage0(backbone, variant)
        assert count == 2
        y = forward_variant_backbone(backbone, torch.randn(1, 128, 5, 5, 3), None)[0]
        assert y.shape == (1, 128, 5, 5, 3)
        if variant == "bottleneck96":
            assert isinstance(backbone.layers[0][0], DenseBottleneck3D)
            assert backbone.layers[0][0].reduce.conv.out_channels == 96
        else:
            assert isinstance(backbone.layers[0][0], AsymmetricFactorized3D)


def test_registered_fidelity_configs_keep_the_same_20k_protocol():
    from fpsgen.hier_occ_flow.stage1_training import validate_config
    root = Path(__file__).parents[1] / "configs/research_v2/hier_occ_flow"
    for name in ("fidelity_v2_e1_bottleneck96.yaml",
                 "fidelity_v2_e2_asymmetric3d.yaml",
                 "fidelity_v2_e3_lightweight_skip.yaml"):
        path = root / name
        cfg = yaml.safe_load(path.read_text())
        cfg["geometry"] = yaml.safe_load((root / cfg["geometry"]).read_text())
        geometry = validate_config(cfg)
        assert geometry.parent_resolution == (384, 384, 32)
        assert cfg["training"]["successful_frame_exposures"] == 20_000
        assert cfg["training"]["successful_updates"] == 5_000
        assert cfg["training"]["seed"] == 20261021
        assert cfg["training"]["physical_gpus"] == [2, 3]
        assert cfg["training"]["batch_per_gpu"] == 2
        assert cfg["validation"]["threshold"] == 0.5
        assert cfg["validation"]["nfes"] == [2]

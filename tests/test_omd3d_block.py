import torch
from torch import nn

from fpsgen.hier_occ_flow.dense_encoder_variants import (
    DenseBottleneck3D,
    ObservationModulatedDense3D,
    forward_variant_backbone,
    replace_stage0_blocks,
)
from fpsgen.hier_occ_flow.observation_context import build_observation_context


def _context(batch, shape, device="cpu"):
    indices = []
    for b in range(batch):
        indices.append(torch.tensor([[1, min(2, shape[1] - 1), min(3, shape[2] - 1)],
                                     [1, min(2, shape[1] - 1), min(3, shape[2] - 1)],
                                     [shape[0] - 1, 0, shape[2] - 1]],
                                    device=device))
    return build_observation_context(indices, shape, device=device)


def test_observation_context_xyz_layout_duplicate_and_coverage():
    ctx = _context(2, (8, 9, 5))
    assert ctx.mask.shape == (2, 1, 8, 9, 5)
    assert ctx.mask[0, 0, 1, 2, 3] == 1
    assert ctx.mask.sum() == 4
    assert ctx.rho_x.shape == ctx.mask.shape
    assert ctx.rho_y.shape == ctx.mask.shape
    assert ctx.rho_z.shape == ctx.mask.shape
    assert ctx.rho_x[0, 0, 1, 2, 3] > 0
    assert torch.isfinite(ctx.features).all()


def test_omd_runs_with_empty_observation_and_preserves_shape():
    block = ObservationModulatedDense3D(
        channels=32, internal_channels=16,
        axial_kernels=(3, 3, 3), skip=nn.Identity())
    x = torch.randn(2, 32, 8, 9, 5)
    empty = build_observation_context(
        [torch.empty(0, 3, dtype=torch.long) for _ in range(2)],
        (8, 9, 5))
    y = block(x, empty)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_omd_local_axial_and_gate_receive_gradients():
    block = ObservationModulatedDense3D(
        channels=32, internal_channels=16,
        axial_kernels=(3, 3, 3), skip=nn.Identity())
    ctx = _context(2, (8, 9, 5))
    x = torch.randn(2, 32, 8, 9, 5, requires_grad=True)
    block(x, ctx).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("local1.conv.weight", "axial_x.conv.weight", "gate.weight",
                 "context_scale"):
        value = dict(block.named_parameters())[name]
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_bottleneck_and_omd_support_amp_and_shape():
    x = torch.randn(1, 128, 8, 8, 3)
    context = _context(1, (8, 8, 3))
    for block in (DenseBottleneck3D(skip=nn.Identity()),
                  ObservationModulatedDense3D(skip=nn.Identity(),
                                              axial_kernels=(3, 3, 3))):
        result = block(x, context)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


def test_stage0_replacement_is_local_and_keeps_skip_and_droppath_objects():
    class OriginalBlock(nn.Module):
        def __init__(self, channels=128):
            super().__init__()
            self.downsample = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1),
                                            nn.GroupNorm(32, channels))
            self.drop_path = nn.Identity()

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Sequential(OriginalBlock(), OriginalBlock()),
                                        nn.Sequential(nn.Identity()),
                                        nn.Sequential(nn.Identity()))
            self.backbone_output_ids = (0, 1, 2)

        def forward(self, x):
            result = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.backbone_output_ids:
                    result.append(x)
            return result

    backbone = Backbone()
    skip = backbone.layers[0][0].downsample
    drop_path = backbone.layers[0][0].drop_path
    assert replace_stage0_blocks(backbone, "bottleneck", internal_channels=64) == 2
    assert backbone.layers[1][0].__class__ is nn.Identity
    assert backbone.layers[0][0].skip is skip
    assert backbone.layers[0][0].drop_path is drop_path
    x = torch.randn(1, 128, 8, 8, 3)
    ctx = _context(1, (8, 8, 3))
    outputs = forward_variant_backbone(backbone, x, ctx)
    assert [tuple(v.shape) for v in outputs] == [tuple(x.shape)] * 3

"""Dense Stage0 block variants for fidelity-oriented architecture pilots.

The vendor ResNet3D implementation stays untouched. All variants retain the
same 128-channel input/output, Stage0 spatial shape, original residual add and
original DropPath policy.
"""
from torch import nn

from .dense_encoder_variants import DenseBottleneck3D


class AsymmetricFactorized3D(nn.Module):
    """Full first 3x3x3 layer, factorized second 3x3x3 layer.

    The second convolution is represented as 1x1x1 -> 3x3x3 -> 1x1x1
    without intermediate normalization or activation. The original second
    ConvModule's BatchNorm remains after the composed factorized convolution.
    """

    def __init__(self, original, rank=64):
        super().__init__()
        if rank < 1 or rank > 128:
            raise ValueError("asymmetric rank must be in [1, 128]")
        self.conv1 = original.conv1
        norm_name = getattr(original.conv2, "norm_name", None)
        norm = getattr(original.conv2, norm_name, None) if norm_name else None
        if norm is None:
            norm = getattr(original.conv2, "bn", None)
        if norm is None:
            raise ValueError("original second ConvModule must contain BatchNorm")
        self.reduce = nn.Conv3d(128, rank, kernel_size=1, bias=False)
        self.spatial = nn.Conv3d(rank, rank, kernel_size=3, padding=1, bias=False)
        self.expand = nn.Conv3d(rank, 128, kernel_size=1, bias=False)
        self.norm2 = norm
        self.skip = original.downsample if original.downsample is not None else nn.Identity()
        self.drop_path = original.drop_path
        self.relu = original.relu
        self._fidelity_dense_block = True

    def forward(self, x, observation_context=None):
        identity = self.skip(x)
        y = self.conv1(x)
        y = self.norm2(self.expand(self.spatial(self.reduce(y))))
        return self.relu(self.drop_path(y) + identity)


class LightweightSkip3D(nn.Module):
    """Original main branch with a 1x1x1 projection in the first skip."""

    def __init__(self, original):
        super().__init__()
        self.conv1 = original.conv1
        self.conv2 = original.conv2
        old = original.downsample
        if old is None:
            raise ValueError("lightweight skip is only defined for projected blocks")
        old_conv = getattr(old, "conv", None)
        if old_conv is None:
            raise ValueError("expected ConvModule skip projection")
        norm_name = getattr(old, "norm_name", None)
        old_norm = getattr(old, norm_name, None) if norm_name else None
        if old_norm is None:
            old_norm = getattr(old, "bn", None)
        if old_norm is None:
            old_norm = getattr(old, "gn", None)
        if old_norm is None:
            raise ValueError("expected normalized ConvModule skip projection")
        self.skip_conv = nn.Conv3d(
            old_conv.in_channels, old_conv.out_channels, kernel_size=1,
            stride=old_conv.stride, padding=0, dilation=old_conv.dilation,
            groups=old_conv.groups, bias=old_conv.bias is not None,
        )
        self.skip_norm = old_norm
        self.drop_path = original.drop_path
        self.relu = original.relu
        self._fidelity_dense_block = True

    def forward(self, x, observation_context=None):
        identity = self.skip_norm(self.skip_conv(x))
        y = self.conv2(self.conv1(x))
        return self.relu(self.drop_path(y) + identity)


def replace_fidelity_stage0(backbone, variant, *, internal_channels=96,
                            asymmetric_rank=64):
    """Replace only Stage0 blocks and return the number changed."""
    if variant not in ("original", "bottleneck96", "asymmetric3d", "lightweight_skip"):
        raise ValueError(f"unknown fidelity encoder variant: {variant}")
    if variant == "original":
        return 0
    blocks = list(backbone.layers[0])
    if variant == "lightweight_skip":
        # Only the first block has a learned projection; later Stage0 blocks
        # retain identity skip paths and remain byte-for-byte the same modules.
        backbone.layers[0][0] = LightweightSkip3D(blocks[0])
        return 1
    replacements = []
    for block in blocks:
        if variant == "bottleneck96":
            replacement = DenseBottleneck3D.from_original(
                block, internal_channels=internal_channels)
        else:
            replacement = AsymmetricFactorized3D(block, rank=asymmetric_rank)
        replacements.append(replacement)
    backbone.layers[0] = nn.Sequential(*replacements)
    return len(replacements)

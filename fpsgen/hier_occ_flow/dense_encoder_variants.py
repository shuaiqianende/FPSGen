"""Stage0-only dense 3-D residual block variants for the OMD3D pilot.

The pinned VoxDet vendor implementation remains untouched. New blocks retain
the original first-block skip projection module (including its configured
normalization) and its DropPath module.
"""
import torch
from torch import nn

from .observation_context import ObservationContext


class ConvNormAct3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *, padding=0,
                 groups=1, norm="bn", activation=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              padding=padding, groups=groups, bias=False)
        if norm == "bn":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == "gn":
            self.norm = nn.GroupNorm(32, out_channels)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unsupported norm {norm!r}")
        self.act = nn.ReLU(inplace=False) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DenseBottleneck3D(nn.Module):
    """R1 local 3-D bottleneck; accepts but does not use an observation context."""
    def __init__(self, channels=128, internal_channels=64, *, skip=None,
                 drop_path=None):
        super().__init__()
        if channels % 32 or internal_channels <= 0:
            raise ValueError("channels must support GN-compatible skip modules")
        self.reduce = ConvNormAct3d(channels, internal_channels, 1, norm="bn",
                                    activation=True)
        self.local1 = ConvNormAct3d(internal_channels, internal_channels, 3,
                                    padding=1, norm="bn", activation=True)
        self.local2 = ConvNormAct3d(internal_channels, internal_channels, 3,
                                    padding=1, norm="bn", activation=False)
        self.expand = ConvNormAct3d(internal_channels, channels, 1, norm="bn")
        self.skip = skip if skip is not None else nn.Identity()
        self.drop_path = drop_path if drop_path is not None else nn.Identity()
        self.relu = nn.ReLU(inplace=False)

    @classmethod
    def from_original(cls, original, *, internal_channels=64, **kwargs):
        return cls(channels=128, internal_channels=internal_channels,
                   skip=original.downsample, drop_path=original.drop_path,
                   **kwargs)

    def forward(self, x, observation_context=None):
        u = self.reduce(x)
        local = self.local2(self.local1(u))
        out = self.drop_path(self.expand(local))
        return self.relu(out + self.skip(x))


class ObservationModulatedDense3D(DenseBottleneck3D):
    """R2 bottleneck plus axis-wise depthwise context and input-only gating."""
    def __init__(self, channels=128, internal_channels=64, *, skip=None,
                 drop_path=None, axial_kernels=(7, 7, 3), gamma_init=1e-3,
                 gate_mode="observation"):
        super().__init__(channels, internal_channels, skip=skip,
                         drop_path=drop_path)
        if len(axial_kernels) != 3 or any(k <= 0 or k % 2 == 0 for k in axial_kernels):
            raise ValueError("axial_kernels must contain three positive odd sizes")
        if gate_mode not in ("observation", "feature"):
            raise ValueError("gate_mode must be observation or feature")
        self.axial_kernels = tuple(int(k) for k in axial_kernels)
        self.gate_mode = gate_mode
        c = internal_channels
        kx, ky, kz = self.axial_kernels
        self.axial_x = ConvNormAct3d(c, c, (kx, 1, 1),
                                     padding=(kx // 2, 0, 0), groups=c,
                                     norm="bn")
        self.axial_y = ConvNormAct3d(c, c, (1, ky, 1),
                                     padding=(0, ky // 2, 0), groups=c,
                                     norm="bn")
        self.axial_z = ConvNormAct3d(c, c, (1, 1, kz),
                                     padding=(0, 0, kz // 2), groups=c,
                                     norm="bn")
        gate_in = c + (4 if gate_mode == "observation" else 0)
        self.gate = nn.Conv3d(gate_in, 3, kernel_size=1)
        self.context_scale = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x, observation_context=None):
        if self.gate_mode == "observation" and observation_context is None:
            raise ValueError("OMD-3D observation gate requires input context")
        u = self.reduce(x)
        local = self.local2(self.local1(u))
        ax = self.axial_x(u)
        ay = self.axial_y(u)
        az = self.axial_z(u)
        if self.gate_mode == "observation":
            context = observation_context.features.to(dtype=u.dtype)
            if context.shape[0] != u.shape[0] or context.shape[2:] != u.shape[2:]:
                raise ValueError("observation context shape does not match feature grid")
            gate_input = torch.cat((u, context), dim=1)
        else:
            gate_input = u
        gx, gy, gz = self.gate(gate_input).sigmoid().chunk(3, dim=1)
        context = gx * ax + gy * ay + gz * az
        out = self.drop_path(self.expand(local + self.context_scale * context))
        return self.relu(out + self.skip(x))


def replace_stage0_blocks(backbone, variant, *, internal_channels=64,
                          axial_kernels=(7, 7, 3), gamma_init=1e-3):
    """Replace only stage-0 blocks, returning count; original is a no-op."""
    if variant not in ("original", "bottleneck", "omd3d"):
        raise ValueError("variant must be original, bottleneck, or omd3d")
    if variant == "original":
        return 0
    stage = backbone.layers[0]
    replacements = []
    for block in stage:
        if variant == "bottleneck":
            replacement = DenseBottleneck3D.from_original(
                block, internal_channels=internal_channels)
        else:
            replacement = ObservationModulatedDense3D.from_original(
                block, internal_channels=internal_channels,
                axial_kernels=axial_kernels, gamma_init=gamma_init)
        replacements.append(replacement)
    backbone.layers[0] = nn.Sequential(*replacements)
    return len(replacements)


def forward_variant_backbone(backbone, x, observation_context):
    """Run a variant backbone, supplying context only to its Stage0 blocks."""
    first = backbone.layers[0][0]
    if not (isinstance(first, DenseBottleneck3D) or
            getattr(first, "_fidelity_dense_block", False)):
        return backbone(x)
    features = []
    value = x
    for block in backbone.layers[0]:
        if (isinstance(block, DenseBottleneck3D) or
                getattr(block, "_fidelity_dense_block", False)):
            value = block(value, observation_context)
        else:
            value = block(value)
    if 0 in backbone.backbone_output_ids:
        features.append(value)
    for level, layer in enumerate(backbone.layers[1:], start=1):
        value = layer(value)
        if level in backbone.backbone_output_ids:
            features.append(value)
    return features


class VariantBackboneRunner(nn.Module):
    """Adapter so the existing whole-backbone checkpoint policy still applies."""
    def __init__(self, backbone, observation_context):
        super().__init__()
        self.backbone = backbone
        self.observation_context = observation_context

    def forward(self, x):
        return forward_variant_backbone(self.backbone, x,
                                        self.observation_context)

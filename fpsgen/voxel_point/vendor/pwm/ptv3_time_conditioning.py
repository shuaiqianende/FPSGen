"""DiT-style flow-time conditioning for the official PTv3 blocks.

The official PTv3 model is kept unchanged.  This module subclasses only its
``Block`` and temporarily installs that subclass while the official
``PointTransformerV3`` constructor creates encoder and decoder blocks.
Scene-level time embeddings travel through PTv3's existing ``context`` field,
which official pooling/unpooling already preserves.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fpsgen.voxel_point.vendor.pwm.ptv3_reference import model as ptv3


class TimeConditionedBlock(ptv3.Block):
    """Official PTv3 Block with AdaLN-Zero on attention and MLP branches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_channels = 32
        self.time_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.time_channels, 6 * self.channels)
        )
        # Preserve the initial unconditioned PTv3 computation.
        nn.init.constant_(self.time_modulation[-1].weight, 0)
        nn.init.constant_(self.time_modulation[-1].bias, 0)

    def forward(self, point):
        if "context" not in point:
            return super().forward(point)

        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        context = point.context[point.batch]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.time_modulation(context).chunk(6, dim=1)
        )

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point.feat = point.feat * (1 + scale_attn) + shift_attn
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + gate_attn * point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point.feat = point.feat * (1 + scale_mlp) + shift_mlp
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + gate_mlp * point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


def build_official_ptv3_with_time_context(build_fn):
    """Build official PTv3 with only its Block class replaced temporarily."""
    original_block = ptv3.Block
    ptv3.Block = TimeConditionedBlock
    try:
        return build_fn()
    finally:
        ptv3.Block = original_block

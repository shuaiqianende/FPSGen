"""PWM source-only displacement contract; GT context exists only in training."""

from contextlib import nullcontext

import torch
from torch import nn

from .point_flow import build_ptv3
from .metrics import chamfer_sqrt


class MappingTeacher(nn.Module):
    def __init__(self, spec, network=None):
        super().__init__()
        self.spec = spec
        self.net = build_ptv3(spec, teacher=True, network=network)

    def forward(self, source, voxel_ids, gt_context):
        if source.shape != (180000, 3) or gt_context.shape != (180000, 3):
            raise ValueError("Teacher requires 180000 source + 180000 GT context")
        target_ids, _ = self.spec.voxelize_points(gt_context)
        points = torch.cat((source, gt_context))
        centers = self.spec.voxel_centers(torch.cat((voxel_ids, target_ids))).to(points)
        flags = torch.cat((torch.zeros(len(source), device=source.device, dtype=torch.bool),
                           torch.ones(len(gt_context), device=source.device, dtype=torch.bool)))
        # SpConv 2.3.6's eval path skips the autocast wrapper used by its
        # autograd path. Keep sparse eval FP32; official FlashAttention casts
        # its QKV to FP16 internally. Training retains validated external AMP.
        precision = nullcontext() if self.training else torch.autocast(points.device.type,enabled=False)
        with precision:
            displacement = self.net.forward_conditional(points.float(), centers.float(), points.new_zeros(1).float(), known_flags=flags)
        if displacement.shape != points.shape or not torch.isfinite(displacement).all():
            raise ValueError("Invalid source-indexed Teacher displacement")
        return source + displacement[:len(source)]

    def loss(self, source, voxel_ids, gt_context, *, backend="keops"):
        return chamfer_sqrt(self(source, voxel_ids, gt_context), gt_context, backend=backend)

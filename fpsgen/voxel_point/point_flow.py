"""PWM official PTv3 velocity regression and source-indexed Euler integration."""

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F


def build_ptv3(spec, *, teacher=False, network=None):
    from .vendor.pwm.ptv3_reference_adapter import OfficialPTV3PointFlowNet
    options = dict(network or {})
    net = OfficialPTV3PointFlowNet(in_channels=7 if teacher else 6,
        coordinate_scale=tuple((b-a)/2 for a,b in zip(spec.origin, spec.maximum)),
        flow_voxel_size=spec.size, **options)
    from .ptv3_checkpoint import enable_block_checkpointing
    return enable_block_checkpointing(net)


def flow_matching_pair(source, endpoint, t):
    if source.shape != endpoint.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Source-indexed endpoint must match source XYZ")
    if t.numel() != 1 or not 0 <= float(t) <= 1:
        raise ValueError("One flow time in [0,1] per B1 scene")
    if not torch.isfinite(source).all() or not torch.isfinite(endpoint).all():
        raise ValueError("Nonfinite flow pair")
    endpoint = endpoint.detach()
    return (1-t) * source + t * endpoint, endpoint - source


class PointFlow(nn.Module):
    def __init__(self, spec, network=None):
        super().__init__()
        self.spec = spec
        self.net = build_ptv3(spec, network=network)

    def forward(self, points, t, voxel_ids):
        if points.shape != (len(voxel_ids), 3):
            raise ValueError("Point/voxel ID alignment mismatch")
        centers = self.spec.voxel_centers(voxel_ids).to(points)
        # Sparse eval FP32 avoids SpConv's mixed weight/feature algorithm
        # selection failure; FlashAttention still uses its native FP16 QKV.
        precision = nullcontext() if self.training else torch.autocast(points.device.type,enabled=False)
        with precision:
            return self.net(points.float(), centers.float(), t.reshape(1).float())

    def velocity_loss(self, source, endpoint, voxel_ids, t):
        state, target = flow_matching_pair(source, endpoint, t)
        prediction = self(state, t, voxel_ids)
        if prediction.shape != source.shape or not torch.isfinite(prediction).all():
            raise ValueError("Invalid PointFlow velocity")
        return F.mse_loss(prediction.float(), target.float())


@torch.no_grad()
def euler_rollout(velocity_net, source, voxel_ids, *, steps=8):
    if steps not in (4, 8, 16):
        raise ValueError("Registered Euler settings are 4/8/16")
    state = source.clone()
    for step in range(steps):
        t = state.new_tensor([step / steps])
        velocity = velocity_net(state, t, voxel_ids)
        if velocity.shape != state.shape or not torch.isfinite(velocity).all():
            raise ValueError("Invalid inference velocity")
        state = state + velocity.float() / steps
    if not torch.isfinite(state).all():
        raise ValueError("Nonfinite PointFlow rollout")
    return state.float()

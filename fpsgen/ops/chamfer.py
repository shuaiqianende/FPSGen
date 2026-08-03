"""Stable FPSGen wrapper for the ChamferDistancePytorch CUDA extension.

The autograd wrapper is adapted from the upstream MIT-licensed
ChamferDistancePytorch project.  Keeping it in FPSGen makes the public import
path independent of the third-party submodule's source-tree layout; the CUDA
extension itself is still built directly from that pinned submodule.
"""

import torch
from torch import nn
from torch.autograd import Function

try:
    import chamfer_3D
except ImportError as error:
    raise ImportError(
        "Chamfer distance is not installed. Run `pip install -e "
        "fpsgen/models/ChamferDistancePytorch/chamfer3D` from the FPSGen root."
    ) from error


class _Chamfer3DFunction(Function):
    """Autograd bridge for the pinned ChamferDistancePytorch CUDA operator."""

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batch_size, num_first, channels = xyz1.size()
        if channels != 3 or xyz2.size(2) != 3:
            raise ValueError("Chamfer distance expects point tensors with shape [B, N, 3].")
        num_second = xyz2.size(1)
        device = xyz1.device

        dist1 = torch.zeros(batch_size, num_first, device=device)
        dist2 = torch.zeros(batch_size, num_second, device=device)
        idx1 = torch.zeros(batch_size, num_first, dtype=torch.int32, device=device)
        idx2 = torch.zeros(batch_size, num_second, dtype=torch.int32, device=device)

        torch.cuda.set_device(device)
        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, _grad_idx1, _grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1 = torch.zeros_like(xyz1)
        grad_xyz2 = torch.zeros_like(xyz2)
        chamfer_3D.backward(
            xyz1,
            xyz2,
            grad_xyz1,
            grad_xyz2,
            grad_dist1.contiguous(),
            grad_dist2.contiguous(),
            idx1,
            idx2,
        )
        return grad_xyz1, grad_xyz2


class Chamfer3DDist(nn.Module):
    """Compute bidirectional squared nearest-neighbor distances on CUDA."""

    def forward(self, input1, input2):
        return _Chamfer3DFunction.apply(input1.contiguous(), input2.contiguous())

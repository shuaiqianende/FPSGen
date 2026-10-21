"""Input-only observation statistics for dense 3-D encoder variants.

All coordinates are coarse-cell integer XYZ indices.  Feature tensors use the
VoxDet layout [B,C,X,Y,Z], so the spatial axes here deliberately follow XYZ.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ObservationContext:
    mask: torch.Tensor
    rho_x: torch.Tensor
    rho_y: torch.Tensor
    rho_z: torch.Tensor

    @property
    def features(self):
        return torch.cat((self.mask, self.rho_x, self.rho_y, self.rho_z), dim=1)


def build_observation_context(indices_by_batch, grid_shape, *, device=None,
                             dtype=torch.float32):
    """Build one binary occupied-cell mask and three zero-padded coverages.

    ``indices_by_batch`` contains the exact integer voxel coordinates already
    produced for LiDAR tokenization. Duplicate returns remain one occupied
    voxel, and empty observations are supported.
    """
    shape = tuple(int(x) for x in grid_shape)
    if len(shape) != 3 or any(x <= 0 for x in shape):
        raise ValueError("grid_shape must be positive XYZ dimensions")
    if not indices_by_batch:
        raise ValueError("at least one batch item is required")
    if device is None:
        device = indices_by_batch[0].device
    mask = torch.zeros((len(indices_by_batch), 1, *shape), device=device,
                       dtype=dtype)
    for batch_index, coords in enumerate(indices_by_batch):
        coords = torch.as_tensor(coords, device=device, dtype=torch.long)
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError("each index array must have shape [N,3] XYZ")
        if coords.numel() == 0:
            continue
        limits = torch.tensor(shape, device=device, dtype=torch.long)
        if bool(((coords < 0) | (coords >= limits)).any()):
            raise ValueError("observation index outside the encoder grid")
        mask[batch_index, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    # F.avg_pool3d treats dimensions as D,H,W. For [X,Y,Z], use the
    # corresponding kernels and symmetric zero padding. count_include_pad=True
    # defines coverage relative to the full window, including at borders.
    def coverage(kernel, padding):
        # avg_pool3d rejects a window larger than a spatial axis. The
        # equivalent zero-padded convolution keeps the intended fixed-window
        # denominator for small synthetic tests as well as production shapes.
        if all(n >= k for n, k in zip(shape, kernel)):
            return F.avg_pool3d(mask, kernel, stride=1, padding=padding,
                                count_include_pad=True)
        weight = torch.ones((1, 1, *kernel), device=device, dtype=dtype)
        return F.conv3d(mask, weight / float(kernel[0] * kernel[1] * kernel[2]),
                        padding=padding)

    rho_x = coverage((7, 1, 1), (3, 0, 0))
    rho_y = coverage((1, 7, 1), (0, 3, 0))
    rho_z = coverage((1, 1, 3), (0, 0, 1))
    return ObservationContext(mask, rho_x, rho_y, rho_z)

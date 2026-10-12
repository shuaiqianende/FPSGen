"""One geometry contract. Dense tensors use XYZ; flat IDs use ZYX order."""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class VoxelGridSpec:
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -4.0
    z_max: float = 4.4
    nx: int = 256
    ny: int = 256
    nz: int = 32
    xy_radius: float = 50.0
    circle_mask: bool = True

    def __post_init__(self):
        if min(self.shape) < 1 or any(a >= b for a, b in zip(self.origin, self.maximum)):
            raise ValueError("Invalid voxel grid")
        if self.xy_radius <= 0:
            raise ValueError("xy_radius must be positive")

    @property
    def shape(self):
        return self.nx, self.ny, self.nz

    @property
    def origin(self):
        return self.x_min, self.y_min, self.z_min

    @property
    def maximum(self):
        return self.x_max, self.y_max, self.z_max

    @property
    def size(self):
        return tuple((b - a) / n for a, b, n in zip(self.origin, self.maximum, self.shape))

    @property
    def num_voxels(self):
        return self.nx * self.ny * self.nz

    @property
    def coarse_grid_size(self):
        if any(n % 2 for n in self.shape):
            raise ValueError("VoxDet coarse_ratio=2 requires even grid dimensions")
        return tuple(n // 2 for n in self.shape)

    def voxel_position_grid_coarse(self, *, device=None):
        # Official preprocessing's positional tensor, explicitly XYZ ordered.
        # The LiDAR trunk does not consume it; expose it for coordinate audits.
        axes = [torch.arange(n,device=device,dtype=torch.float64) for n in self.coarse_grid_size]
        indices = torch.stack(torch.meshgrid(*axes,indexing="ij"),-1)
        return (indices.new_tensor(self.origin)+(indices+.5)*indices.new_tensor(self.size)*2).float()

    def to_dict(self):
        return asdict(self)

    def valid_points(self, xyz):
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("XYZ must have shape [N,3]; labels are not model inputs")
        p = xyz.double()
        valid = torch.isfinite(p).all(-1)
        valid &= (p > p.new_tensor(self.origin)).all(-1)
        valid &= (p < p.new_tensor(self.maximum)).all(-1)
        if self.circle_mask:
            valid &= p[:, :2].square().sum(-1) < self.xy_radius ** 2
        return valid

    def voxelize_points(self, xyz, *, strict=True):
        valid = self.valid_points(xyz)
        if strict and not bool(valid.all()):
            raise ValueError(f"Invalid GT: {int((~valid).sum())} nonfinite/out-of-ROI points")
        p = xyz[valid].double()
        indices = torch.floor((p - p.new_tensor(self.origin)) / p.new_tensor(self.size)).long()
        return self.flatten_voxel_ids(indices), valid

    def flatten_voxel_ids(self, indices):
        if indices.shape[-1] != 3 or indices.is_floating_point():
            raise ValueError("Integer XYZ indices required")
        if bool(((indices < 0) | (indices >= indices.new_tensor(self.shape))).any()):
            raise ValueError("Voxel index outside grid")
        ix, iy, iz = indices.unbind(-1)
        return (iz * self.ny + iy) * self.nx + ix

    def unflatten_voxel_ids(self, ids):
        if ids.is_floating_point() or bool(((ids < 0) | (ids >= self.num_voxels)).any()):
            raise ValueError("Integer voxel IDs inside grid required")
        return torch.stack((ids % self.nx, (ids // self.nx) % self.ny,
                            ids // (self.nx * self.ny)), -1)

    def voxel_centers(self, ids):
        indices = self.unflatten_voxel_ids(ids).double()
        return (indices.new_tensor(self.origin) + (indices + .5) * indices.new_tensor(self.size)).float()

    def valid_circle_mask(self, *, device=None):
        # Closest point of the closed XY cell to the origin. Tangency alone
        # has no overlap with the open disk and is invalid. Centers are NOT used.
        x = self.x_min + torch.arange(self.nx, dtype=torch.float64, device=device) * self.size[0]
        y = self.y_min + torch.arange(self.ny, dtype=torch.float64, device=device) * self.size[1]
        closest_x = torch.maximum(x, torch.zeros_like(x)) + torch.minimum(x + self.size[0], torch.zeros_like(x))
        closest_y = torch.maximum(y, torch.zeros_like(y)) + torch.minimum(y + self.size[1], torch.zeros_like(y))
        mask = closest_x[:, None].square() + closest_y[None, :].square() < self.xy_radius ** 2
        if not self.circle_mask:
            mask = torch.ones_like(mask)
        return mask[:, :, None].expand(self.shape)

    def flat_to_dense(self, values):
        if values.shape[-1] != self.num_voxels:
            raise ValueError("Wrong flat voxel count")
        return values.reshape(*values.shape[:-1], self.nz, self.ny, self.nx).transpose(-3, -1).contiguous()

    def dense_to_flat(self, values):
        if tuple(values.shape[-3:]) != self.shape:
            raise ValueError("Wrong dense XYZ shape")
        return values.transpose(-3, -1).contiguous().flatten(-3)

    def counts(self, xyz, *, expected_points=180000):
        if xyz.shape != (expected_points, 3):
            raise ValueError(f"GT must contain exactly {expected_points} XYZ, got {tuple(xyz.shape)}")
        ids, _ = self.voxelize_points(xyz)
        counts = torch.bincount(ids, minlength=self.num_voxels)
        if int(counts.sum()) != expected_points:
            raise ValueError("GT density mass mismatch")
        return self.flat_to_dense(counts)


def voxelize_points(points, spec=None, **kwargs):
    return (spec or VoxelGridSpec()).voxelize_points(points, **kwargs)


def flatten_voxel_ids(indices, spec=None):
    return (spec or VoxelGridSpec()).flatten_voxel_ids(indices)


def unflatten_voxel_ids(ids, spec=None):
    return (spec or VoxelGridSpec()).unflatten_voxel_ids(ids)


def voxel_centers(ids, spec=None):
    return (spec or VoxelGridSpec()).voxel_centers(ids)


def valid_circle_mask(spec=None, **kwargs):
    return (spec or VoxelGridSpec()).valid_circle_mask(**kwargs)

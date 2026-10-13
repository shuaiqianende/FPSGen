"""XYZ external grids, ZYX Conv3d fields, cylindrical representation/sphere GT."""

from dataclasses import dataclass
import torch

from fpsgen.voxel_point.voxel_spec import VoxelGridSpec


@dataclass(frozen=True)
class ShortcutGeometry(VoxelGridSpec):
    sphere_radius: float = 50.
    point_budget: int = 180000

    def __post_init__(self):
        super().__post_init__()
        if self.sphere_radius <= 0 or self.point_budget != 180000:
            raise ValueError("Positive GT sphere radius and exactly180000 output points required")

    @property
    def field_shape(self):
        return self.nz, self.ny, self.nx

    def valid_points(self, xyz):
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("Only XYZ is a geometry input")
        p = xyz.double()
        valid = torch.isfinite(p).all(-1)
        valid &= (p >= p.new_tensor(self.origin)).all(-1)
        valid &= (p < p.new_tensor(self.maximum)).all(-1)
        if self.circle_mask:
            valid &= p[:,:2].square().sum(-1) < self.xy_radius**2
        return valid & (p.square().sum(-1) < self.sphere_radius**2)

    def representation_mask(self, *, device=None):
        return super().valid_circle_mask(device=device)

    def valid_circle_mask(self, *, device=None):
        # Minimum distance from each closed cell to the sphere origin; centers
        # outside the sphere can still contain valid GT. Tangency is excluded.
        closest = []
        for start, size, count in zip(self.origin,self.size,self.shape):
            lo = start + torch.arange(count,device=device,dtype=torch.float64)*size
            closest.append(torch.maximum(lo,torch.zeros_like(lo)) + torch.minimum(lo+size,torch.zeros_like(lo)))
        x,y,z = closest
        sphere = x[:,None,None].square()+y[None,:,None].square()+z[None,None,:].square() < self.sphere_radius**2
        return self.representation_mask(device=device) & sphere

    def to_field(self, xyz_grid):
        if tuple(xyz_grid.shape[-3:]) != self.shape:
            raise ValueError("Expected XYZ voxel grid")
        return xyz_grid.transpose(-3,-1).contiguous()

    def to_xyz(self, field):
        if tuple(field.shape[-3:]) != self.field_shape:
            raise ValueError("Expected ZYX Conv3d field")
        return field.transpose(-3,-1).contiguous()

    def condition_to_field(self,feature):
        shapes=[tuple(n//(2**level) for n in self.shape) for level in (1,2,3)]
        if feature.ndim!=5 or tuple(feature.shape[-3:]) not in shapes:
            raise ValueError("Expected VoxDet coarse XYZ condition pyramid")
        return feature.transpose(-3,-1).contiguous()

    def field_mask(self, *, device=None):
        return self.to_field(self.valid_circle_mask(device=device))[None,None]

    def observed_mask(self, input_xyz):
        ids,_ = self.voxelize_points(input_xyz,strict=False)
        occupied = torch.bincount(ids,minlength=self.num_voxels)>0
        return self.to_field(self.flat_to_dense(occupied))[None,None]

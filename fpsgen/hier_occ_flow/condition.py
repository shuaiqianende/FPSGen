"""World-space sampling of frozen LiDAR features, plus input-only hit masks."""
import torch
from torch import nn
import torch.nn.functional as F
from .voxelization import voxelize_surface,lookup_masks,unpack_children


def sample_pyramid(pyramid,world_centers,origin,maximum,*,chunk=65536):
    if not pyramid: raise ValueError('A real LiDAR feature pyramid is required')
    lo=world_centers.new_tensor(origin); extent=world_centers.new_tensor(maximum)-lo
    grid=(2*(world_centers-lo)/extent-1).float()
    sampled=[]
    for feature in pyramid:
        if feature.ndim!=5 or feature.shape[0]!=1: raise ValueError('ZYX B1 feature volume required')
        values=[]
        for start in range(0,len(grid),chunk):
            local=grid[start:start+chunk].reshape(1,1,1,-1,3)
            value=F.grid_sample(feature.float(),local,mode='bilinear',padding_mode='border',align_corners=False)
            values.append(value[0,:,0,0].T)
        sampled.append(torch.cat(values))
    return torch.cat(sampled,dim=-1)


def observed_children(input_xyz,candidate_ids,geometry):
    xyz=input_xyz.detach().cpu().numpy() if isinstance(input_xyz,torch.Tensor) else input_xyz
    fine,valid=geometry.voxelize_points(xyz)
    if not len(fine):
        import numpy as np
        return np.zeros((len(candidate_ids),geometry.k),dtype=bool)
    ids,masks,_=voxelize_surface(xyz,geometry)
    return unpack_children(lookup_masks(ids,masks,candidate_ids),geometry)


class FrozenVoxDetCondition(nn.Module):
    """One encoder pass; strict checkpoint geometry; optional original density head.

    Official VoxDet stores dense features in XYZ. Only this adapter converts
    them to grid_sample's ZYX. A different hierarchy can sample this encoder in
    world coordinates, but its density cannot be reused as a different grid.
    """
    def __init__(self,checkpoint):
        super().__init__()
        from fpsgen.voxel_point.voxel_spec import VoxelGridSpec
        from fpsgen.voxel_shortcut.voxdet_encoder import VoxDetEncoder
        value=torch.load(checkpoint,map_location='cpu')
        if value.get('stage')!='stage1' or value.get('mechanics_smoke_only',True):
            raise ValueError('A validated Stage1 density checkpoint is required')
        self.spec=VoxelGridSpec(**value['geometry'])
        self.encoder=VoxDetEncoder(self.spec,activation_checkpointing=False)
        self.encoder.load_state_dict(value['model'],strict=True)
        self.encoder.requires_grad_(False); self.encoder.eval()

    def train(self,mode=True):
        super().train(False); return self

    @torch.no_grad()
    def forward(self,lidar_xyz,*,with_density=False):
        _,cls,geo,pyramid=self.encoder.encode_features(lidar_xyz)
        features=[v.permute(0,1,4,3,2).contiguous() for v in pyramid]
        density=None
        if with_density:
            from fpsgen.voxel_point.density_head import nonnegative_density
            output=self.encoder.main_head([[cls[0]],[geo[0]]])
            density=nonnegative_density(output['output_voxels'],self.encoder.valid)[0,0]
        return features,density

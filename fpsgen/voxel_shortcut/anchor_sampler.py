"""Gaussian sampling in cell/cylinder/sphere intersections; no coordinate clamp."""

import math
import torch


def sample_spherical_anchors(geometry,counts,*,generator,noise_std_ratio=.25,max_rounds=10000):
    if generator is None or noise_std_ratio<=0 or counts.shape!=(geometry.num_voxels,) or counts.is_floating_point() or bool((counts<0).any()):
        raise ValueError('Explicit generator, positive noise and nonnegative flat integer counts required')
    valid=geometry.dense_to_flat(geometry.valid_circle_mask(device=counts.device))
    if bool((counts[~valid]!=0).any()):raise ValueError('Counts outside spherical voxel intersections')
    ids=torch.repeat_interleave(torch.arange(geometry.num_voxels,device=counts.device),counts)
    indices=geometry.unflatten_voxel_ids(ids).double();size=indices.new_tensor(geometry.size)
    lower=indices.new_tensor(geometry.origin)+indices*size;upper=lower+size
    center=lower+size*.5
    closest=torch.maximum(lower,torch.zeros_like(lower))+torch.minimum(upper,torch.zeros_like(upper))
    squares=closest.square()
    axis_max=(geometry.sphere_radius**2-(squares.sum(-1,keepdim=True)-squares)).clamp_min(0).sqrt()
    lower=torch.maximum(lower,-axis_max);upper=torch.minimum(upper,axis_max)
    if geometry.circle_mask:
        circle_max=(geometry.xy_radius**2-closest[:,:2].flip(-1).square()).clamp_min(0).sqrt()
        lower[:,:2]=torch.maximum(lower[:,:2],-circle_max);upper[:,:2]=torch.minimum(upper[:,:2],circle_max)
    sigma=size*noise_std_ratio
    cdf_lo=.5*(1+torch.erf((lower-center)/(sigma*math.sqrt(2))))
    cdf_hi=.5*(1+torch.erf((upper-center)/(sigma*math.sqrt(2))))
    if bool((cdf_hi<=cdf_lo).any()):raise ValueError('Numerically empty Gaussian truncation; no point duplication fallback')
    result=center.new_empty(center.shape,dtype=torch.float32);pending=torch.arange(len(ids),device=ids.device)
    for _ in range(max_rounds):
        if not pending.numel():return result,ids
        u=torch.rand((len(pending),3),device=ids.device,dtype=torch.float64,generator=generator)
        q=cdf_lo[pending]+u*(cdf_hi[pending]-cdf_lo[pending])
        candidates=(center[pending]+sigma*math.sqrt(2)*torch.erfinv(2*q-1)).float()
        candidate_ids,inside=geometry.voxelize_points(candidates,strict=False)
        accepted=inside.clone();accepted[inside]&=candidate_ids==ids[pending][inside]
        result[pending[accepted]]=candidates[accepted];pending=pending[~accepted]
    raise RuntimeError(f'Spherical truncated sampling failed for{len(pending)} points')

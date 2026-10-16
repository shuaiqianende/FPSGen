"""GT-free sparse integration. LiDAR encoding is supplied once, outside NFE loop."""
import numpy as np
import torch
from .voxelization import pack_children,occupied_fine_coords


@torch.no_grad()
def sample(model,parent_ids,condition,*,nfe=4,generator=None,threshold=0.):
    if model.training: raise ValueError('Inference requires eval mode')
    if model.objective=='bce':
        endpoint=model(parent_ids,**condition)
        return endpoint,(endpoint>threshold)&condition['valid_child_mask'],1
    if nfe not in (1,2,4,8) or generator is None: raise ValueError('NFE must be 1/2/4/8 with explicit RNG')
    valid=condition['valid_child_mask']; x=torch.randn(valid.shape,device=valid.device,generator=generator)*valid
    for step in range(nfe):
        time=x.new_tensor(step/nfe)
        velocity=model(parent_ids,state=x,time=time,**condition)
        if not bool(torch.isfinite(velocity).all()): raise ValueError('Nonfinite velocity')
        x=x+velocity.float()/nfe
    if not bool(torch.isfinite(x).all()): raise ValueError('Nonfinite endpoint')
    return x,(x>threshold)&valid,nfe


def export_occupancy(directory,parent_ids,occupancy,geometry):
    """Surface-hit centers have variable point count; no 180k claim/point allocator."""
    from pathlib import Path
    from .storage import atomic_json
    out=Path(directory); out.mkdir(parents=True,exist_ok=False)
    ids=np.asarray(parent_ids,dtype='<u4'); occ=np.asarray(occupancy,dtype=bool)
    occ &= geometry.valid_children(ids.astype(np.int64))
    masks=pack_children(occ,geometry)
    np.savez_compressed(out/'fine_occupancy.npz',parent_ids=ids,child_occ_mask=masks)
    centers=geometry.centers(occupied_fine_coords(ids,masks,geometry),fine=True).astype('<f4')
    with (out/'fine_surface_hit_centers.ply').open('wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(centers)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n').encode())
        f.write(centers.tobytes())
    atomic_json(out/'manifest.json',{'geometry':geometry.to_dict(),'geometry_sha256':geometry.sha256,
        'representation':'surface-hit occupancy centers, not final subvoxel point geometry','points':len(centers)})

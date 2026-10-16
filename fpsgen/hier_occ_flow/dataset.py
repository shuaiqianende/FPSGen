"""Scene assembly keeps conditioning separate from occupancy supervision."""
from pathlib import Path
import numpy as np
import torch
from .condition import observed_children,sample_pyramid
from .candidate_selection import oracle_candidates,select_candidates
from .voxelization import lookup_masks,unpack_children


def input_xyz(path):
    raw=np.load(path,allow_pickle=False)
    if raw.ndim!=2 or raw.shape[1]<3: raise ValueError('Partial LiDAR must contain XYZ')
    xyz=np.array(raw[:,:3],dtype=np.float32,copy=True)
    if not np.isfinite(xyz).all(): raise ValueError('Nonfinite partial LiDAR')
    return xyz


def assemble_scene(record,config,geometry,device,encoder,seed):
    path=Path(config['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
    xyz=input_xyz(path)
    policy=config['candidates']; pyramid=None; density=None
    if encoder is not None:
        pyramid,density=encoder(torch.as_tensor(xyz,device=device),with_density=policy['mode']=='predicted')
    if policy['mode']=='oracle':
        ids,scores,source=oracle_candidates(record['parent_ids'],geometry,np.random.default_rng(seed),
            negative_fraction=policy.get('negative_fraction',.25))
    elif policy['mode']=='predicted':
        if density is None: raise ValueError('Predicted candidates require a real Stage1 forward')
        ids,scores=select_candidates(density.cpu().numpy(),geometry,**policy['selection']); source={'mode':'predicted'}
    else: raise ValueError('Unknown candidate policy')
    if not len(ids): raise ValueError('No active parents after selection')
    if len(ids)>config['training']['max_parent_tokens']:
        raise RuntimeError(f'Full scene has {len(ids)} parents, exceeds validated token budget; no random truncation is permitted')
    parent_ids=torch.as_tensor(ids,dtype=torch.long,device=device)
    valid=torch.as_tensor(geometry.valid_children(ids),device=device)
    observed=observed_children(xyz,ids,geometry)
    condition={'parent_score':torch.as_tensor(scores,device=device),
               'observed_child':torch.as_tensor(observed,device=device),'valid_child_mask':valid}
    if 'pyramid' in config['model']['condition_mode']:
        if pyramid is None: raise ValueError('Real frozen LiDAR encoder required')
        centers=geometry.centers(geometry.unflatten(parent_ids)).float()
        condition['lidar_features']=sample_pyramid(pyramid,centers,encoder.spec.origin,encoder.spec.maximum).detach()
    target=unpack_children(lookup_masks(record['parent_ids'],record['child_occ_mask'],ids),geometry)
    return parent_ids,condition,torch.as_tensor(target,device=device),source

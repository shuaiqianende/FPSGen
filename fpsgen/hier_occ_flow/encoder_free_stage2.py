"""Independent Stage2 ablation: no dense Encoder or pyramid feature branch.

Oracle support is a training/diagnostic input. Deployment supplies predicted
Parent IDs externally; neither case constructs or runs Stage1 here.
"""
import copy
import numpy as np
import torch
from torch import nn
from .condition import observed_children
from .stage2_candidates import oracle_parent_candidates
from .voxelization import lookup_masks,unpack_children


def make_encoder_free_copy(source,*,keep_observed=True):
    if (source.condition_mode!='pyramid_observed' or source.objective!='fm'
        or source.context_strategy!='dynamic' or source.child_bridge!='none'
        or source.state_condition_interaction!='none' or source.local_state_encoding!='none'):
        raise ValueError('Original dynamic FM with pyramid+observed conditions required')
    model=copy.deepcopy(source)
    linear=model.fuse[0]
    # Canonical concat: position32, score16, lidarW, observed32, stateW, time32.
    widths=[model.position[-1].out_features,model.score[-1].out_features,
        model.lidar[-1].out_features,model.observed[-1].out_features,
        model.local[-1].out_features,model.time[-1].out_features]
    if sum(widths)!=linear.in_features:raise ValueError('Unexpected fusion order/shape')
    keep=[];cursor=0
    for index,width in enumerate(widths):
        if index!=2 and (index!=3 or keep_observed):keep.extend(range(cursor,cursor+width))
        cursor+=width
    # All replacement weights are copied. Do not consume a training RNG draw.
    devices=[linear.weight.device.index or 0] if linear.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        replacement=nn.Linear(len(keep),linear.out_features,bias=linear.bias is not None,
            device=linear.weight.device,dtype=linear.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(linear.weight[:,keep])
        if linear.bias is not None:replacement.bias.copy_(linear.bias)
    replacement.train(linear.training);model.fuse[0]=replacement
    del model.lidar
    if not keep_observed:del model.observed
    model.condition_mode='observed' if keep_observed else 'none'
    old=source.state_dict();new=model.state_dict()
    removed=[n for n in old if n not in new]
    if any(not (n.startswith('lidar.') or not keep_observed and n.startswith('observed.')) for n in removed):
        raise RuntimeError('Unexpected parameter removal')
    if any(not torch.equal(v,old[n]) for n,v in new.items() if n!='fuse.0.weight'):
        raise RuntimeError('Shared parameters changed during branch removal')
    return model,dict(condition_mode=model.condition_mode,removed_tensors=removed,
        fusion_input_before=linear.in_features,fusion_input_after=len(keep),
        fusion_column_indices=keep,shared_tensors_exact=True,random_initialization=False,
        encoder_instantiated=False,lidar_projection_present=False)


def encoder_free_conditions(parent_ids,xyz,geometry,*,device='cpu',keep_observed=True):
    ids=np.asarray(parent_ids,dtype=np.int64)
    if ids.ndim!=1 or not len(ids) or np.any(ids[1:]<=ids[:-1]):
        raise ValueError('Nonempty sorted unique Parent IDs required')
    geometry.unflatten(ids)
    condition=dict(parent_score=torch.ones(len(ids),device=device),
        valid_child_mask=torch.as_tensor(geometry.valid_children(ids),device=device))
    if keep_observed:
        if xyz is None:raise ValueError('Observed mode requires original Partial LiDAR XYZ')
        condition['observed_child']=torch.as_tensor(observed_children(xyz,ids,geometry),device=device)
    return torch.as_tensor(ids,device=device),condition


def assemble_oracle_encoder_free_scene(record,xyz,geometry,*,seed,device='cpu',
                                        keep_observed=True,negative_fraction=0.):
    ids,_,provenance=oracle_parent_candidates(record['parent_ids'],geometry,seed=seed,
        negative_fraction=negative_fraction,neutral_score=1.)
    parents,condition=encoder_free_conditions(ids,xyz,geometry,device=device,keep_observed=keep_observed)
    # GT fine occupancy is queried only for targets, not injected as a feature.
    targets=unpack_children(lookup_masks(record['parent_ids'],record['child_occ_mask'],ids),geometry)
    provenance.update(encoder_forwards=0,stage1_flow_forwards=0,
        lidar_pyramid_used=False,observed_mask_input_only=keep_observed,
        gt_fine_is_condition=False,oracle_support_not_deployable=True)
    return parents,condition,torch.as_tensor(targets,device=device),provenance

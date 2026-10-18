"""Input-only Stage1-to-parent conditions; no GT membership channel."""
import hashlib
import numpy as np
import torch
from .condition import sample_pyramid,observed_children
from .stage2_candidates import select_occupancy_candidates,oracle_parent_candidates


def scene_seed(sequence,frame,base):
    return (base+int.from_bytes(hashlib.sha256(f'{sequence}/{int(frame):06d}'.encode()).digest()[:4],'little'))%(2**31)


@torch.no_grad()
def input_conditions(xyz,geometry,adapter,policy,*,seed,oracle_ids=None):
    """GT IDs are usable solely to choose oracle support, not feature values.

    The deployment caller never passes oracle_ids. Predicted selection has no
    GT argument and runs before target lookup in stage2_dataset.
    """
    mode=policy['mode'];score_mode=policy.get('oracle_score','neutral')
    if mode not in ('oracle','predicted'):raise ValueError('Unknown parent mode')
    if mode=='predicted' and oracle_ids is not None:raise ValueError('Predicted candidates must not receive GT support')
    need_scores=mode=='predicted' or score_mode=='stage1'
    pyramid,score=adapter(xyz,nfe=policy['stage1_nfe'],seed=seed,with_scores=need_scores)
    if need_scores:
        if score is None or tuple(score.shape)!=geometry.parent_resolution[::-1]:
            raise ValueError('Actual Stage1 score field in ZYX layout required')
        if not bool(torch.isfinite(score).all() and ((score>=0)&(score<=1)).all()):
            raise ValueError('Stage1 endpoint score must be finite and in[0,1]')
    if mode=='predicted':
        ids,scores,provenance=select_occupancy_candidates(score.cpu().numpy(),geometry,**policy['selection'])
    else:
        if oracle_ids is None:raise ValueError('Oracle mode needs labeled support')
        if score_mode not in ('neutral','stage1'):raise ValueError('Unknown oracle score policy')
        ids,scores,provenance=oracle_parent_candidates(oracle_ids,geometry,seed=seed,
            negative_fraction=policy.get('negative_fraction',.25),neutral_score=policy.get('neutral_score',1.))
        if need_scores:
            scores=score.reshape(-1)[torch.as_tensor(ids,device=score.device)].detach()
            provenance['score_policy']='actual_Stage1_score_for_every_positive_and_negative'
            provenance.pop('neutral_score',None)
    if not len(ids):raise ValueError('Empty candidate set; no GT/uniform fallback')
    parent_ids=torch.from_numpy(ids).to(xyz.device)
    centers=geometry.centers(geometry.unflatten(parent_ids)).float()
    # Use identical sampled-feature precision for training, cache, and export.
    condition={'lidar_features':sample_pyramid(pyramid,centers,geometry.origin,geometry.maximum).detach().half(),
        'parent_score':torch.as_tensor(scores,device=xyz.device),
        'observed_child':torch.as_tensor(observed_children(xyz,ids,geometry),device=xyz.device),
        'valid_child_mask':torch.as_tensor(geometry.valid_children(ids),device=xyz.device)}
    if condition['lidar_features'].shape!=(len(ids),384):raise ValueError('Expected sampled128x3 actual LiDAR features')
    if any(t.requires_grad for t in condition.values()):raise RuntimeError('Frozen Stage1 graph leaked into Stage2')
    provenance.update(stage1_nfe=policy['stage1_nfe'] if need_scores else 0,
        stage1_encoder_forwards=1,stage1_seed=seed,stage1=adapter.provenance,
        conditioning='parent_centered384D,normalizedXYZ,score,input_only_observed8,scene_time32')
    provenance['parent_score_source']='actual_stage1_endpoint' if need_scores else 'constant_neutral'
    provenance['stage1_score_is_calibrated_probability']=False
    return parent_ids,condition,provenance

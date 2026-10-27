"""Input-only ordinary-FM joint inference. No GT support or parent expansion API."""
import hashlib
import torch
from .condition import observed_children,sample_pyramid

POLICY=dict(candidate_mode='predicted',stage1_nfe=2,threshold=.5,expansion=0,
            extra_parent_fraction=0.,gt_support=False,observed_parent_union=False,
            stage2_nfes=[1,2,4,8],primary_stage2_nfe=4,source_seed=20261031)


def validate_policy(policy):
    if policy!=POLICY:raise ValueError('Registered predicted-only/no-extra-parent policy required')


@torch.no_grad()
def predicted_support(score,valid,geometry):
    if score.numel()!=geometry.num_parents or valid.numel()!=geometry.num_parents:
        raise ValueError('Full original Parent grid required')
    if not bool(torch.isfinite(score).all() and ((score>=0)&(score<=1)).all()):
        raise ValueError('Finite Stage1 scores in[0,1] required')
    ids=torch.nonzero((score.reshape(-1)>.5)&valid.reshape(-1).bool()).flatten()
    if not len(ids):raise RuntimeError('Empty predicted support; no GT/neighbor fallback')
    values=score.reshape(-1)[ids].float()
    return ids,values


@torch.no_grad()
def input_only_conditions(ids,scores,xyz,geometry,*,pyramid=None):
    if ids.ndim!=1 or not len(ids) or ids.shape!=scores.shape:
        raise ValueError('Explicit predicted IDs and their actual Stage1 scores required')
    if len(ids)>1 and not bool((ids[1:]>ids[:-1]).all()):raise ValueError('Sorted unique Parent IDs required')
    ids_cpu=ids.cpu().numpy()
    condition=dict(parent_score=scores.float(),
        observed_child=torch.as_tensor(observed_children(xyz,ids_cpu,geometry),device=ids.device),
        valid_child_mask=torch.as_tensor(geometry.valid_children(ids_cpu),device=ids.device))
    if pyramid is not None:
        centers=geometry.centers(geometry.unflatten(ids)).float()
        features=sample_pyramid(pyramid,centers,geometry.origin,geometry.maximum).half()
        if features.shape!=(len(ids),384):raise ValueError('Original three128D pyramid levels required')
        condition['lidar_features']=features
    return condition


def ids_sha(ids):return hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()

"""High-recall parent selection and explicitly labeled oracle parent diagnostics."""
import itertools
import numpy as np
from .voxelization import lookup_masks,unpack_children


def valid_ids(ids,geometry):
    return np.concatenate([ids[j:j+32768][geometry.valid_children(ids[j:j+32768]).any(1)] for j in range(0,len(ids),32768)]) if len(ids) else ids


def expand_neighbors(ids,geometry,radius=1):
    coord=geometry.unflatten(np.asarray(ids,dtype=np.int64)); items=[]
    for delta in itertools.product(range(-radius,radius+1),repeat=3):
        shifted=coord+np.asarray(delta)
        in_grid=((shifted>=0)&(shifted<geometry.parent_resolution)).all(1)
        items.append(geometry.flatten(shifted[in_grid]))
    return valid_ids(np.unique(np.concatenate(items)),geometry)


def select_candidates(density_xyz,geometry,*,strategy='threshold',threshold=.5,top_k=100000,expansion=0,budget=None):
    if tuple(density_xyz.shape)!=geometry.parent_resolution: raise ValueError('Stage1 density and target parent grid differ; do not reinterpret IDs')
    if not np.isfinite(density_xyz).all() or (density_xyz<0).any(): raise ValueError('Invalid Stage1 count density')
    flat=density_xyz.transpose(2,1,0).reshape(-1)
    if strategy=='threshold': ids=np.flatnonzero(flat>=threshold)
    elif strategy=='topk':
        if not 1<=top_k<=len(flat): raise ValueError('Invalid candidate top-K')
        # Stable tie-break by ID to make configured seed irrelevant to support.
        ids=np.lexsort((np.arange(len(flat)),-flat))[:top_k]
    else: raise ValueError('Unknown selection strategy')
    ids=valid_ids(ids.astype(np.int64),geometry)
    if expansion: ids=expand_neighbors(ids,geometry,expansion)
    if budget is not None and len(ids)>budget: ids=ids[np.lexsort((ids,-flat[ids]))[:budget]]
    ids=np.sort(ids)
    return ids,flat[ids]


def oracle_candidates(label_ids,geometry,generator,*,negative_fraction=.25):
    if not 0<=negative_fraction<=1: raise ValueError('Invalid nearby-negative fraction')
    neighbors=expand_neighbors(label_ids.astype(np.int64),geometry)
    empty=np.setdiff1d(neighbors,label_ids,assume_unique=True)
    n=min(len(empty),int(len(label_ids)*negative_fraction))
    chosen=generator.choice(empty,size=n,replace=False)
    ids=np.union1d(label_ids,chosen).astype(np.int64)
    # A score derived from GT membership would disclose empty-vs-nonempty
    # parent targets. Every oracle candidate receives the SAME neutral score.
    return ids,np.ones(len(ids),np.float32),{'mode':'oracle_not_deployable','nearby_negatives':n,'negative_fraction_relative_to_gt_parents':negative_fraction,'score_policy':'constant_1_for_all_candidates'}


def candidate_metrics(candidates,label_ids,label_masks,geometry):
    matched=np.isin(label_ids,candidates)
    tp=int(matched.sum()); fp=len(candidates)-tp; fn=len(label_ids)-tp
    counts=unpack_children(label_masks,geometry).sum(1)
    return {'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
        'iou':tp/max(tp+fp+fn,1),'false_negative_rate':fn/max(tp+fn,1),'active_parent_count':len(candidates),
        'fine_occupancy_recall_ceiling':int(counts[matched].sum())/max(int(counts.sum()),1)}

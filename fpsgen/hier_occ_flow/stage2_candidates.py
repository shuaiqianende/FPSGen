"""384-grid score selection, explicit budgets, and membership-free oracle scores."""
from functools import lru_cache
import numpy as np
from scipy.ndimage import binary_dilation


@lru_cache(maxsize=4)
def parent_valid_field(geometry):
    closest=[]
    for n,origin,size in zip(geometry.parent_resolution,geometry.origin,geometry.parent_size):
        lo=origin+np.arange(n)*size;hi=lo+size
        closest.append((np.maximum(lo,0)+np.minimum(hi,0))**2)
    x,y,z=closest
    valid=(z[:,None,None]+y[None,:,None]+x[None,None,:])<geometry.sphere_radius**2
    valid.flags.writeable=False
    return valid


def expand_parent_ids(ids,geometry,radius=1):
    if type(radius) is not int or radius<0:raise ValueError('Nonnegative integer expansion radius required')
    ids=np.asarray(ids,dtype=np.int64);geometry.unflatten(ids)
    mask=np.zeros(geometry.num_parents,dtype=bool);mask[ids]=True
    mask=mask.reshape(geometry.parent_resolution[::-1])
    if radius:mask=binary_dilation(mask,structure=np.ones((3,3,3),dtype=bool),iterations=radius)
    return np.flatnonzero(mask.ravel()&parent_valid_field(geometry).ravel())


def select_occupancy_candidates(score_zyx,geometry,*,strategy='threshold',threshold=.5,top_k=None,
                                expansion=0,budget=None,overflow='error'):
    score=np.asarray(score_zyx,dtype=np.float32)
    if score.shape!=geometry.parent_resolution[::-1] or not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError('Expected finite occupancy score ZYX in[0,1], not old XYZ density')
    flat=score.ravel();valid=parent_valid_field(geometry).ravel()
    if strategy=='threshold':
        if not 0<=threshold<=1:raise ValueError('Invalid score threshold')
        ids=np.flatnonzero((flat>threshold)&valid)
    elif strategy=='topk':
        if type(top_k) is not int or not 0<top_k<=int(valid.sum()):raise ValueError('Invalid valid-ROI topK')
        choices=np.flatnonzero(valid);ids=choices[np.lexsort((choices,-flat[choices]))[:top_k]]
    else:raise ValueError('Unknown candidate strategy')
    before_expansion=len(ids)
    if expansion:ids=expand_parent_ids(ids,geometry,expansion)
    uncapped=len(ids)
    if overflow not in ('error','topk_score'):raise ValueError('Budget overflow must be explicit')
    if budget is not None and (type(budget) is not int or budget<=0):raise ValueError('Invalid token budget')
    if budget is not None and len(ids)>budget:
        if overflow=='error':raise RuntimeError(f'Candidate budget exceeded:{len(ids)}>{budget}; no truncation performed')
        ids=ids[np.lexsort((ids,-flat[ids]))[:budget]]
    ids=np.sort(ids)
    return ids,flat[ids],{'mode':'predicted','score_definition':'uncalibrated Stage1 endpoint proxy',
        'before_expansion':before_expansion,'before_budget':uncapped,'selected':len(ids),
        'budget_dropped':uncapped-len(ids),'budget_policy':overflow,'expansion':expansion}


def oracle_parent_candidates(label_ids,geometry,*,seed,negative_fraction=.25,neutral_score=1.):
    ids=np.asarray(label_ids,dtype=np.int64)
    if not 0<=negative_fraction<=1:raise ValueError('Invalid negative fraction')
    nearby=expand_parent_ids(ids,geometry,1)
    empty=np.setdiff1d(nearby,ids,assume_unique=True)
    n=min(len(empty),int(len(ids)*negative_fraction))
    negatives=np.random.default_rng(seed).choice(empty,n,replace=False)
    candidates=np.union1d(ids,negatives)
    return candidates,np.full(len(candidates),neutral_score,np.float32),{
        'mode':'oracle_not_deployable','nearby_empty_pool':len(empty),'nearby_negatives':n,
        'negative_fraction_relative_to_gt':negative_fraction,'negative_sampling_weight':1.,
        'score_policy':'constant_neutral_for_positive_and_negative','neutral_score':neutral_score}

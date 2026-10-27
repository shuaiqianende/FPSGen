"""Input-only Stage1-to-parent conditions; no GT membership channel."""
import hashlib
import numpy as np
import torch
from .condition import sample_pyramid,observed_children
from .stage2_candidates import select_occupancy_candidates,oracle_parent_candidates


def scene_seed(sequence,frame,base):
    return (base+int.from_bytes(hashlib.sha256(f'{sequence}/{int(frame):06d}'.encode()).digest()[:4],'little'))%(2**31)


def observed_child_geometry(xyz, candidate_ids, geometry):
    """Input-only hit/count/centroid/spread statistics for each candidate Child."""
    points = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if ids.ndim != 1 or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1])):
        raise ValueError('Child statistics require sorted unique candidate IDs')
    output = np.zeros((len(ids), geometry.k, 8), dtype=np.float32)
    if not len(ids) or not len(points):
        return output
    fine, valid = geometry.voxelize_points(points)
    if not np.any(valid):
        return output
    valid_points = np.asarray(points)[valid].astype(np.float64, copy=False)
    parent, child = geometry.split_fine(fine)
    parent_id = geometry.flatten(parent)
    row = np.searchsorted(ids, parent_id)
    inside = row < len(ids)
    possible = np.flatnonzero(inside)
    inside[possible] = ids[row[possible]] == parent_id[possible]
    if not np.any(inside):
        return output
    row, child = row[inside], child[inside]
    group = row * geometry.k + child
    fine_coord = fine[valid][inside]
    center = geometry.centers(fine_coord, fine=True)
    half = np.asarray(geometry.fine_size, dtype=np.float64) * 0.5
    local = (valid_points[inside] - center) / half
    size = len(ids) * geometry.k
    count = np.bincount(group, minlength=size).astype(np.float64)
    mean = np.zeros((size, 3), dtype=np.float64)
    second = np.zeros_like(mean)
    for axis in range(3):
        mean[:, axis] = np.bincount(group, weights=local[:, axis], minlength=size)
        second[:, axis] = np.bincount(group, weights=local[:, axis] ** 2, minlength=size)
    hit = count > 0
    mean[hit] /= count[hit, None]
    second[hit] = np.maximum(second[hit] / count[hit, None] - mean[hit] ** 2, 0.)
    flat = output.reshape(size, 8)
    flat[:, 0] = hit.astype(np.float32)
    flat[:, 1] = (np.log1p(count) / max(float(np.log1p(count.max())), 1.)).astype(np.float32)
    flat[:, 2:5] = mean.astype(np.float32)
    flat[:, 5:8] = np.sqrt(second).astype(np.float32)
    return output


@torch.no_grad()
def input_conditions(xyz,geometry,adapter,policy,*,seed,oracle_ids=None,
                     include_child_geometry=False):
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
            negative_fraction=policy.get('negative_fraction',0.),neutral_score=policy.get('neutral_score',1.))
        if need_scores:
            scores=score.reshape(-1)[torch.as_tensor(ids,device=score.device)].detach()
            provenance['score_policy']='actual_Stage1_score_for_every_positive_and_negative'
            provenance.pop('neutral_score',None)
    if not len(ids):raise ValueError('Empty candidate set; no GT/uniform fallback')
    parent_ids=torch.from_numpy(ids).to(xyz.device)
    centers=geometry.centers(geometry.unflatten(parent_ids)).float()
    # Both input-only geometric conditions share one D2H copy of the raw
    # points; do not independently transfer/convert the same scan twice.
    xyz_cpu=xyz.detach().cpu().numpy()
    # Use identical sampled-feature precision for training, cache, and export.
    condition={'lidar_features':sample_pyramid(pyramid,centers,geometry.origin,geometry.maximum).detach().half(),
        'parent_score':torch.as_tensor(scores,device=xyz.device),
        'observed_child':torch.as_tensor(observed_children(xyz_cpu,ids,geometry),device=xyz.device),
        'valid_child_mask':torch.as_tensor(geometry.valid_children(ids),device=xyz.device)}
    if include_child_geometry:
        child = observed_child_geometry(xyz_cpu, ids, geometry)
        condition['child_geometry'] = torch.as_tensor(child, device=xyz.device).half()
    if condition['lidar_features'].shape!=(len(ids),384):raise ValueError('Expected sampled128x3 actual LiDAR features')
    if any(t.requires_grad for t in condition.values()):raise RuntimeError('Frozen Stage1 graph leaked into Stage2')
    provenance.update(stage1_nfe=policy['stage1_nfe'] if need_scores else 0,
        stage1_encoder_forwards=1,stage1_seed=seed,stage1=adapter.provenance,
        conditioning='parent_centered384D,normalizedXYZ,score,input_only_observed8,scene_time32')
    provenance['parent_score_source']='actual_stage1_endpoint' if need_scores else 'constant_neutral'
    provenance['stage1_score_is_calibrated_probability']=False
    return parent_ids,condition,provenance

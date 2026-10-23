"""Input-only fixed-budget support diagnostics. GT is evaluation-only."""
import hashlib
import numpy as np


def stable_topk(scores, ids, k):
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    if len(np.unique(ids)) != len(ids) or k <= 0:
        raise ValueError('Unique candidate IDs and positive K required')
    values = np.asarray(scores)[ids]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite candidate rank score')
    k = min(k, len(ids))
    cut = np.partition(values, len(values)-k)[len(values)-k]
    above, tied = ids[values>cut], ids[values==cut]
    return np.sort(np.concatenate((above, tied[:k-len(above)])))


def six_neighbors(ids, geometry, valid):
    coords = geometry.unflatten(np.asarray(ids))
    groups = []
    bounds = np.asarray(geometry.parent_resolution)
    for axis in range(3):
        for shift in (-1, 1):
            other = coords.copy(); other[:,axis] += shift
            inside = ((other>=0)&(other<bounds)).all(1)
            groups.append(geometry.flatten(other[inside]))
    found = np.unique(np.concatenate(groups))
    return found[np.asarray(valid).reshape(-1)[found]]


def local_max6(score_zyx):
    score = np.asarray(score_zyx)
    result = np.zeros_like(score)
    for axis in range(3):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(1,None), slice(None,-1)
        result[tuple(left)] = np.maximum(result[tuple(left)], score[tuple(right)])
        result[tuple(right)] = np.maximum(result[tuple(right)], score[tuple(left)])
    return result


def is_calibration(sequence, frame, *, block_frames=100):
    key = str(sequence) + ':' + str(int(frame)//block_frames) + ':desescene-m3-split-v1'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little') % 5 == 0


def ceiling(ids, gt_ids, gt_masks):
    counts = np.unpackbits(np.asarray(gt_masks, dtype=np.uint8)[:,None], axis=1).sum(1)
    total = int(counts.sum())
    return float(counts[np.isin(gt_ids, ids)].sum()/max(total,1))


def oracle_pool_upper_bound(pool, gt_ids, gt_masks, k):
    # Never returned as an inference selector. This uses GT ONLY as an offline
    # upper bound on the number of Fine positives the fixed pool could cover.
    reward = np.zeros(int(max(np.max(pool), np.max(gt_ids)))+1, dtype=np.int16)
    reward[gt_ids] = np.unpackbits(np.asarray(gt_masks,dtype=np.uint8)[:,None],axis=1).sum(1)
    ids = stable_topk(reward, pool, k)
    return ceiling(ids, gt_ids, gt_masks)

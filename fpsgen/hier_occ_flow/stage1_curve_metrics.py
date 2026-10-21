"""Exact threshold and fixed-budget metrics for Stage1 parent scores."""
import numpy as np


def _arrays(score, record, valid):
    values = np.asarray(score, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if values.shape != valid.shape:
        raise ValueError('score and valid mask must have the same flattened shape')
    if not np.isfinite(values[valid]).all():
        raise ValueError('valid Stage1 scores must be finite')
    gt_ids = np.asarray(record['parent_ids'], dtype=np.int64).reshape(-1)
    if np.any(gt_ids < 0) or np.any(gt_ids >= len(values)) or np.any(~valid[gt_ids]):
        raise ValueError('GT parent IDs must be in-range and valid')
    masks = np.asarray(record['child_occ_mask'], dtype=np.uint64).reshape(-1)
    if masks.shape != gt_ids.shape:
        raise ValueError('GT parent IDs and child masks must have equal length')
    # ``int.bit_count`` is unavailable in the pinned Python 3.9 runtime.
    child_counts = np.fromiter((bin(int(x)).count('1') for x in masks), dtype=np.int16, count=len(masks))
    return values, valid, gt_ids, child_counts


def threshold_curve_metrics(score, record, valid, thresholds):
    """Compute exact strict-score-threshold metrics without rescanning the grid."""
    values, valid, gt_ids, child_counts = _arrays(score, record, valid)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if not np.isfinite(thresholds).all():
        raise ValueError('thresholds must be finite')
    valid_scores = np.sort(values[valid])
    gt_scores = np.sort(values[gt_ids])
    gt_child_counts = child_counts[np.argsort(values[gt_ids], kind='stable')]
    gt_suffix_children = np.cumsum(gt_child_counts[::-1], dtype=np.int64)[::-1]
    total_children = max(int(child_counts.sum()), 1)
    rows = []
    for threshold in thresholds:
        # Match the original NumPy/PyTorch comparison: scalar thresholds are
        # represented in the score dtype before applying the strict ``>``.
        threshold = np.asarray(threshold, dtype=values.dtype).item()
        # Strict > matches the frozen Stage1 candidate selector.
        active = len(valid_scores) - int(np.searchsorted(valid_scores, threshold, side='right'))
        first_gt = int(np.searchsorted(gt_scores, threshold, side='right'))
        tp = len(gt_scores) - first_gt
        fp = active - tp
        fn = len(gt_scores) - tp
        precision = tp / max(active, 1)
        recall = tp / max(len(gt_scores), 1)
        rows.append({
            'threshold': float(threshold), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall,
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'iou': tp / max(tp + fp + fn, 1),
            'active_parents': active,
            'fine_recall_ceiling': (int(gt_suffix_children[first_gt]) if first_gt < len(gt_suffix_children) else 0) / total_children,
        })
    return rows


def fixed_budget_metrics(score, record, valid, budgets):
    """Top-K parent metrics; equal-score ties prefer lower flattened parent IDs."""
    values, valid, gt_ids, child_counts = _arrays(score, record, valid)
    valid_ids = np.flatnonzero(valid)
    valid_scores = values[valid_ids]
    gt_scores = values[gt_ids]
    total_children = max(int(child_counts.sum()), 1)
    result = []
    for requested in budgets:
        requested = int(requested)
        if requested <= 0:
            raise ValueError('candidate budgets must be positive')
        k = min(requested, len(valid_ids))
        if k == 0:
            raise ValueError('cannot select candidates from an empty valid grid')
        cutoff = np.partition(valid_scores, len(valid_scores) - k)[len(valid_scores) - k]
        greater = int(np.count_nonzero(valid_scores > cutoff))
        remaining_ties = k - greater
        tied_ids = valid_ids[valid_scores == cutoff]  # valid_ids is ascending
        tie_rank = np.searchsorted(tied_ids, gt_ids, side='left')
        selected_gt = (gt_scores > cutoff) | ((gt_scores == cutoff) & (tie_rank < remaining_ties))
        tp = int(selected_gt.sum())
        fp = k - tp
        fn = len(gt_ids) - tp
        result.append({
            'requested_k': requested, 'actual_k': k, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / max(k, 1), 'recall': tp / max(len(gt_ids), 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'iou': tp / max(tp + fp + fn, 1),
            'fine_recall_ceiling': int(child_counts[selected_gt].sum()) / total_children,
            'score_cutoff': float(cutoff), 'tie_rule': 'lower_flattened_parent_id_first',
        })
    return result


def summarize_fixed_budget(rows):
    if not rows:
        raise ValueError('cannot summarize empty fixed-budget rows')
    total_tp = sum(r['tp'] for r in rows)
    total_fp = sum(r['fp'] for r in rows)
    total_fn = sum(r['fn'] for r in rows)
    return {
        'frames': len(rows),
        'macro': {k: float(np.mean([r[k] for r in rows])) for k in
                  ('precision', 'recall', 'f1', 'iou', 'fine_recall_ceiling', 'actual_k')},
        'micro': {
            'precision': total_tp / max(total_tp + total_fp, 1),
            'recall': total_tp / max(total_tp + total_fn, 1),
            'f1': 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1),
            'iou': total_tp / max(total_tp + total_fp + total_fn, 1),
        },
    }

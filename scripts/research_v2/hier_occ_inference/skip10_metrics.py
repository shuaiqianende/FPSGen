"""Bounded CPU metric workers. They never initialize CUDA or receive GT in a model."""
import time
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from eval_seq08_combinations import physical_grid_ids, raw_chamfer, support_counts
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.storage import file_sha

_RAW = None
_GEOMETRY = None
_REFERENCE = None


def initialize(data_root, geometry, reference):
    global _RAW, _GEOMETRY, _REFERENCE
    import torch
    torch.set_num_threads(1)
    _GEOMETRY = SparseHierarchyGeometry(**geometry)
    _REFERENCE = Path(reference)
    _RAW = RawMapSource(data_root, fast_viewpoint=True)
    # Each process owns its map and viewpoint adapter; no monkey-patch races.
    _RAW._load('08')


def load_target(record):
    """CPU map/pose extraction only; actual selected metrics run on GPU."""
    import json
    begin = time.perf_counter()
    if file_sha(Path(record['input'])) != record['input_sha256']:
        raise ValueError('Input changed during evaluation')
    raw = _RAW.load('08', record['frame'])
    coords, valid = _GEOMETRY.voxelize_points(raw)
    del coords
    target = raw[valid]
    filtered = len(raw)-len(target)
    target_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(target)).cast('B')).hexdigest()
    checked = False
    if record['frame'] % 100 == 0 and record['frame'] <= 4000:
        cache = _REFERENCE/'targets'/f'08_{record["frame"]:06d}.npz'
        metadata = json.loads(cache.with_suffix('.json').read_text())
        assert metadata['input_sha256'] == record['input_sha256']
        assert metadata['valid_points'] == len(target) and metadata['filtered_points'] == filtered
        assert metadata['geometry_sha256'] == _GEOMETRY.sha256
        with np.load(cache, allow_pickle=False) as saved:
            assert np.array_equal(saved['half_meter_ids'], physical_grid_ids(target, _GEOMETRY, target_points=True))
        checked = True
    return target, dict(raw_gt_points=len(target), raw_candidate_points_before_roi=len(raw),
        filtered_points=filtered, target_sha256=target_sha256,
        previous_skip100_gt_cache_checked=checked,
        raw_target_load_seconds=time.perf_counter()-begin)


def evaluate(payload):
    import json
    record, predicted, parent_ids, inference = payload
    begin = time.perf_counter()
    if file_sha(Path(record['input'])) != record['input_sha256']:
        raise ValueError('Input changed during evaluation')
    raw = _RAW.load('08', record['frame'])
    fine, valid = _GEOMETRY.voxelize_points(raw)
    del fine
    target = raw[valid]
    target_ids = physical_grid_ids(target, _GEOMETRY, target_points=True)
    original_points = len(raw)
    filtered_points = original_points-len(target)
    del raw
    cache_check = False
    if record['frame'] % 100 == 0 and record['frame'] <= 4000:
        cache = _REFERENCE/'targets'/f'08_{record["frame"]:06d}.npz'
        metadata = json.loads(cache.with_suffix('.json').read_text())
        assert metadata['input_sha256'] == record['input_sha256']
        assert metadata['valid_points'] == len(target) and metadata['filtered_points'] == filtered_points
        assert metadata['geometry_sha256'] == _GEOMETRY.sha256
        with np.load(cache, allow_pickle=False) as saved:
            assert np.array_equal(saved['half_meter_ids'], target_ids)
        cache_check = True
    tree = cKDTree(target, leafsize=32)
    metric = raw_chamfer(predicted, target, target_tree=tree, workers=1)
    counts = support_counts(physical_grid_ids(predicted, _GEOMETRY), target_ids)
    parent_points = _GEOMETRY.centers(_GEOMETRY.unflatten(parent_ids)).astype(np.float32)
    coarse_metric = raw_chamfer(parent_points, target, target_tree=tree, workers=1)
    coarse_counts = support_counts(physical_grid_ids(parent_points, _GEOMETRY), target_ids)
    return dict(sequence='08', frame=record['frame'], stage1_nfe=4, stage2_nfe=4,
        **metric, iou05=counts['iou'], **{k:counts[k] for k in ('tp', 'fp', 'fn')},
        predicted_fine_voxels=len(predicted), candidate_parents=len(parent_ids),
        raw_gt_points=len(target), raw_candidate_points_before_roi=original_points,
        filtered_points=filtered_points, previous_skip100_gt_cache_checked=cache_check,
        stage1_only_cd_m=coarse_metric['cd_m'], stage1_only_iou05=coarse_counts['iou'],
        stage1_only_pred_to_gt_m=coarse_metric['pred_to_gt_m'],
        stage1_only_gt_to_pred_m=coarse_metric['gt_to_pred_m'],
        stage1_only_tp=coarse_counts['tp'], stage1_only_fp=coarse_counts['fp'], stage1_only_fn=coarse_counts['fn'],
        metric_seconds=time.perf_counter()-begin, **inference)

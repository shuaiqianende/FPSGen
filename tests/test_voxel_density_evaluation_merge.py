"""Sharded test reports must cover the frozen manifest exactly once."""

import copy
import importlib.util
from pathlib import Path
import statistics

import pytest

source = Path(__file__).resolve().parents[1]/'scripts/research_v2/voxel_point/merge_density_evaluations.py'
module_spec = importlib.util.spec_from_file_location('density_merge',source)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)


def fixture_reports():
    frames = [{'sequence':'08','frame':i*100,'input':f'input/{i}.npy','gt':f'gt/{i}.ply'} for i in range(5)]
    rows = []
    for i,frame in enumerate(frames):
        rows.append({'frame':frame,'gt_validation':{'filtered_roi_points':0},
            'predicted':{'occupancy_iou':1/(1+i)},'partial_density':{'occupancy_iou':1/(1+i)},
            'gt_density':{'occupancy_iou':1.},
            'occupancy_confusion':{name:{'tp':1,'fp':i,'fn':0} for name in ('predicted','partial_density')}})
    common = {'status':'COMPLETE','split':'test','threshold':1.,'threshold_frozen':True,
              'config_sha256':'config','checkpoint_sha256':'weights','checkpoint_updates':11000,
              'device':'cpu','precision':'fp32','frame_stride':100,'num_shards':2,'selected_frame_count':5}
    reports = [{**common,'shard_index':0,'rows':rows[:3]}, {**common,'shard_index':1,'rows':rows[3:]}]
    return reports,frames


def test_unequal_shards_use_all_scene_rows_and_global_confusion():
    reports,frames = fixture_reports()
    result = module.merge_reports(reports,frames)
    assert result['evaluated_frames'] == 5
    expected_macro = statistics.mean(1/(1+i) for i in range(5))
    assert result['macro_means']['predicted']['occupancy_iou'] == pytest.approx(expected_macro)
    assert result['micro_occupancy']['predicted']['occupancy_iou'] == pytest.approx(5/15)


def test_point_voxel_and_density_metrics_remain_separate():
    reports,frames = fixture_reports()
    for report in reports:
        report.update(point_voxel_size_m=.5,point_voxel_iou_policy='floor, no morphology',anchor_points=180000)
        for row in report['rows']:
            row['point_voxel'] = {group:{'tp':2,'fp':1,'fn':1,'iou':.5,'f1':2/3}
                                  for group in ('predicted_anchors','partial_input','fine_grid_anchors')}
    result = module.merge_reports(reports,frames)
    assert result['point_voxel_macro_means']['predicted_anchors']['iou'] == .5
    assert result['point_voxel_micro']['predicted_anchors']['iou'] == .5
    assert result['macro_means']['predicted']['occupancy_iou'] != .5


@pytest.mark.parametrize('failure',['missing','duplicate','checkpoint','pairing'])
def test_merge_rejects_incomplete_or_inconsistent_test_reports(failure):
    reports,frames = fixture_reports()
    reports = copy.deepcopy(reports)
    if failure == 'missing':
        reports[0]['rows'].pop()
    elif failure == 'duplicate':
        reports[1]['rows'].append(reports[0]['rows'][0])
    elif failure == 'checkpoint':
        reports[1]['checkpoint_sha256'] = 'other weights'
    else:
        reports[0]['rows'][0]['frame']['gt'] = 'different target'
    with pytest.raises(ValueError):
        module.merge_reports(reports,frames)

"""Validate complete disjoint evaluation shards and report macro/micro metrics."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def merge_reports(reports, expected_frames):
    if not reports:
        raise ValueError('No completed evaluation shards')
    shared = ('split','threshold','threshold_frozen','config_sha256','checkpoint_sha256',
              'checkpoint_updates','device','precision','frame_stride','num_shards','selected_frame_count')
    reference = reports[0]
    for key in ('point_voxel_size_m','point_voxel_iou_policy','anchor_points'):
        if key in reference:
            shared += (key,)
    if len(reports) != reference['num_shards']:
        raise ValueError('Incomplete shard set')
    if {report['shard_index'] for report in reports} != set(range(reference['num_shards'])):
        raise ValueError('Duplicate/missing shard IDs')
    expected = {(frame['sequence'],frame['frame']):frame for frame in expected_frames}
    if len(expected) != len(expected_frames):
        raise ValueError('Duplicate expected frames')
    if reference['selected_frame_count'] != len(expected):
        raise ValueError('Protocol frame count mismatch')
    actual = {}
    for report in reports:
        if report['status'] != 'COMPLETE' or any(report[key] != reference[key] for key in shared):
            raise ValueError('Incomplete shard or inconsistent checkpoint/protocol')
        for row in report['rows']:
            frame = row['frame']
            identity = frame['sequence'],frame['frame']
            if identity in actual:
                raise ValueError('Duplicate evaluated frame')
            if identity not in expected or frame != expected[identity]:
                raise ValueError('Evaluated frame differs from frozen manifest')
            actual[identity] = row
    if set(actual) != set(expected):
        raise ValueError('Missing evaluated frames')
    rows = [actual[key] for key in sorted(actual)]
    means, spread, micro = {}, {}, {}
    for group in ('predicted','gt_density','partial_density'):
        means[group] = {key:statistics.mean(row[group][key] for row in rows) for key in rows[0][group]}
        spread[group] = {key:{'min':min(row[group][key] for row in rows),
                             'max':max(row[group][key] for row in rows),
                             'std':statistics.pstdev(row[group][key] for row in rows)}
                         for key in rows[0][group]}
        if group != 'gt_density':
            counts = {key:sum(row['occupancy_confusion'][group][key] for row in rows) for key in ('tp','fp','fn')}
            tp,fp,fn = counts['tp'],counts['fp'],counts['fn']
            micro[group] = {**counts,'occupancy_iou':tp/max(tp+fp+fn,1),
                           'occupancy_f1':2*tp/max(2*tp+fp+fn,1)}
    result = {**{key:reference[key] for key in shared},'status':'COMPLETE',
            'evaluated_frames':len(rows),'macro_means':means,'per_scene_spread':spread,
            'micro_occupancy':micro,'filtered_gt_points':sum(row['gt_validation']['filtered_roi_points'] for row in rows),
            'rows':rows}
    if reference.get('point_voxel_size_m',0):
        point_means,point_micro = {},{}
        for group in ('predicted_anchors','partial_input','fine_grid_anchors'):
            point_means[group] = {key:statistics.mean(row['point_voxel'][group][key] for row in rows) for key in ('iou','f1')}
            counts = {key:sum(row['point_voxel'][group][key] for row in rows) for key in ('tp','fp','fn')}
            tp,fp,fn = counts['tp'],counts['fp'],counts['fn']
            point_micro[group] = {**counts,'iou':tp/max(tp+fp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)}
        result['point_voxel_macro_means'] = point_means
        result['point_voxel_micro'] = point_micro
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input-root',required=True,type=Path)
    parser.add_argument('--expected-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    files = sorted(args.input_root.glob('shard*/density_evaluation.json'))
    reports = [json.loads(path.read_text()) for path in files]
    expected = json.loads(args.expected_manifest.read_text())
    result = merge_reports(reports,expected)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'density_evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.output/'per_frame_metrics.csv').open('w',newline='') as handle:
        writer = csv.writer(handle)
        keys = ['occupancy_iou','occupancy_f1','occupied_count_mae','log_count_mae',
                'mass_tv','coarse_mass_tv','mass_calibration_error','density_sum']
        point_keys = ['point_voxel_iou','point_voxel_f1','partial_point_voxel_iou','fine_grid_anchor_iou'] if result.get('point_voxel_size_m',0) else []
        writer.writerow(['sequence','frame']+keys+['partial_iou','filtered_gt_points']+point_keys)
        for row in result['rows']:
            writer.writerow([row['frame']['sequence'],row['frame']['frame']]+
                            [row['predicted'][key] for key in keys]+
                            [row['partial_density']['occupancy_iou'],row['gt_validation']['filtered_roi_points']]+
                            ([row['point_voxel']['predicted_anchors']['iou'],row['point_voxel']['predicted_anchors']['f1'],row['point_voxel']['partial_input']['iou'],row['point_voxel']['fine_grid_anchors']['iou']] if point_keys else []))
    print(json.dumps({key:result[key] for key in ('checkpoint_updates','evaluated_frames','macro_means','micro_occupancy')},indent=2))


if __name__ == '__main__':
    main()

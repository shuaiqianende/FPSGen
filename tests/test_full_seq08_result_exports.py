"""Portable integrity checks for the published small-result bundle."""
import csv
import json
import math
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / 'docs/research_v2/results/full_seq08_seed42_20261006'
MODELS = {'FPSGen', 'SiD2-style', 'DiP-S16'}
METRICS = ['occupancy_iou', 'occupancy_f1', 'completion_f1', 'height_mae_gtocc_m', 'density_mass_tv']


def read_csv(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def test_quality_csv_matches_complete_report():
    quality = read_csv('final_quality_full_seq08_seed42.csv')
    report = json.loads((RESULTS / 'final_report.json').read_text())
    indexed = {(r['method'], r['cfg']): r for r in report['quality']}
    assert {(r['method'], int(r['cfg'])) for r in quality} == {(m, c) for m in MODELS for c in [1, 2]}
    assert len(quality) == 6
    for row in quality:
        assert int(row['frames']) == 4071
        assert int(row['generation_seed']) == 42
        assert int(row['completed_epochs']) == 500
        assert int(row['optimizer_steps']) == 299000
        assert int(row['euler_steps']) == 10
        assert row['runtime_precision'] == 'fp16'
        assert row['dense_backend'] == ('eager' if row['cfg'] == '1' else 'inductor')
        original = indexed[row['method'], int(row['cfg'])]
        for metric in METRICS:
            value = float(row[metric])
            assert math.isfinite(value)
            assert value == original[metric]


def test_checkpoint_and_frame_provenance_matches_all_six_stages():
    provenance = json.loads((RESULTS / 'evaluation_provenance.json').read_text())
    queue = json.loads((RESULTS / 'queue_status.json').read_text())
    assert queue['status'] == provenance['status'] == 'COMPLETE'
    assert len(queue['completed']) == len(provenance['stages']) == 6
    assert len({s['frame_manifest_sha256'] for s in provenance['stages']}) == 1
    for model in MODELS:
        stages = [s for s in provenance['stages'] if s['method'] == model]
        assert len(stages) == 2
        assert len({s['checkpoint_sha256'] for s in stages}) == 1
        for stage in stages:
            assert stage['validation'] == 'PASS'
            assert stage['num_frames'] == 4071
            assert stage['finite_metric_values'] == 4071 * len(METRICS)
            assert stage['generation_seeds'] == [42]
            assert stage['gt_dir'] == 'gt_possion'
            assert stage['condition'] == '100'


def test_parameter_partition_and_training_losses():
    data = read_csv('training_and_parameters.csv')
    assert {r['method'] for r in data} == MODELS
    for row in data:
        assert sum(int(row[k]) for k in ['params_core', 'params_condition', 'params_pointpillar']) == int(row['params_total'])
        for key in ['train_loss_final500', 'train_loss_final200']:
            assert math.isfinite(float(row[key]))
    dip = next(r for r in data if r['method'] == 'DiP-S16')
    assert 'GPU3 x32' in dip['execution']
    assert 'RNG' in dip['limitation']

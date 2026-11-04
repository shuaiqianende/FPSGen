"""Summarize saved Stage1 predictions; block CIs do not measure training-seed variance."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/desescene_eval408_nextstage2_v1'
ARMS = [('U1-Adapted', 'u1_adapted_20k', '20K'), ('U1+A', 'u1a_100k', '100K'),
        ('U1+A+depth moment', 'depth_moment_100k', '100K')]
METRICS = ('cd_m', 'iou05', 'iou02', 'iou01', 'jsd_bev', 'jsd_3d')


def main():
    results = []
    for label, arm, exposure in ARMS:
        path = BASE / arm / 'seq08_skip10_408/results.json'
        data = json.loads(path.read_text())
        rows = data['rows']
        assert data['status'] == 'COMPLETE' and len(rows) == 408
        assert [r['frame'] for r in rows] == list(range(0, 4080, 10))
        results.append((label, exposure, rows))
    assert all([r['gt_points'] for r in results[0][2]] == [r['gt_points'] for r in item[2]] for item in results[1:])
    lines = ['# Stage1 seq08/skip10 408-frame results', '',
             'Original raw-map ground truth; Stage1 NFE2; native parent centers, no resampling.',
             'U1-Adapted is a 20K model; the other two are 100K. Its comparison is not budget matched.', '',
             '| Model | Exposure | CD m ↓ | IoU .5 ↑ | IoU .2 ↑ | IoU .1 ↑ | JSD BEV ↓ | JSD 3D ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, exposure, rows in results:
        means = [np.mean([r[k] for r in rows]) for k in METRICS]
        lines.append('| ' + ' | '.join([label, exposure] + [f'{x:.6f}' for x in means]) + ' |')
    lines += ['', '## Paired depth-moment minus U1+A', '',
              'A 95% percentile block bootstrap resamples contiguous 20-frame blocks (last block has 8 frames), 10,000 replicates, seed 20261105. These intervals describe variability within seq08 only, not uncertainty across scenes or independent training seeds.', '',
              '| Metric | Mean difference | 95% block-bootstrap interval |', '|---|---:|---:|']
    left, right = results[1][2], results[2][2]
    rng = np.random.default_rng(20261105)
    blocks = [np.arange(i, min(i + 20, 408)) for i in range(0, 408, 20)]
    draws = rng.integers(0, len(blocks), size=(10000, len(blocks)))
    for key in METRICS:
        diffs = np.array([b[key] - a[key] for a, b in zip(left, right)])
        boot = np.array([diffs[np.concatenate([blocks[j] for j in draw])].mean() for draw in draws])
        lo, hi = np.quantile(boot, [.025, .975])
        lines.append(f'| {key} | {diffs.mean():+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
    lines += ['', 'The evaluation timer in the existing per-frame records does not include all metric computation; it must not be interpreted as end-to-end evaluation latency.', '']
    target = ROOT / 'docs/research_v2/DESESCENE_STAGE1_408_PAIRED_RESULTS.md'
    target.write_text('\n'.join(lines))
    print(target)


if __name__ == '__main__':
    main()

"""Read-only inventory and paired results from immutable experiment artifacts."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def paired_delta(a, b, *, metric='iou', nested=False, seed=20261034, block_frames=100):
    """Cluster bootstrap of scene-paired deltas; repeats are not extra scenes."""
    key = lambda r: (r['sequence'], int(r['frame']), r.get('seed', 0))
    if [key(r) for r in a] != [key(r) for r in b]:
        raise ValueError('Paired frame/seed identity differs')
    per_scene = {}
    for left, right in zip(a, b):
        x, y = (left['fine'], right['fine']) if nested else (left, right)
        per_scene.setdefault((left['sequence'], int(left['frame'])), []).append(y[metric] - x[metric])
    groups = {}
    for (sequence, frame), values in per_scene.items():
        groups.setdefault((sequence, frame // block_frames), []).append(float(np.mean(values)))
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(4000, len(groups)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    deltas = np.array([np.mean(v) for v in per_scene.values()])
    order = np.argsort(deltas)
    identities = list(per_scene)
    return {'metric': metric, 'direction': 'candidate_minus_reference', 'scenes': len(per_scene),
            'mean': float(deltas.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'grouping': f'sequence and floor(frame/{block_frames})', 'groups': len(groups),
            'bootstrap_seed': seed, 'bootstrap_draws': len(boot),
            'worst_decile_mean': float(deltas[order[:max(1, len(order)//10)]].mean()),
            'worst_frames': [{'sequence': identities[i][0], 'frame': identities[i][1],
                              'delta': float(deltas[i])} for i in order[:10]]}


def inspect_run(root, name, result_path=None, stage=2):
    folder = root / name
    output = {'path': str(folder), 'status': 'NOT_FOUND'}
    if not folder.exists():
        return output
    config = folder / 'config.json'
    if config.exists():
        cfg = json.loads(config.read_text())
        output['config'] = cfg
        output['config_file_sha256'] = sha(config)
    launch = folder / 'launch.json'
    if launch.exists():
        output['launch'] = json.loads(launch.read_text())
    summary = folder / 'summary.json'
    if summary.exists():
        output['training'] = json.loads(summary.read_text())
        output['status'] = output['training'].get('status', 'SUMMARY_PRESENT')
    log = folder / 'train_rank0.jsonl'
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines() if line]
        output['latest'] = {k: rows[-1].get(k) for k in (
            'successful_updates', 'successful_frame_exposures', 'loss', 'finite', 'amp_skips')}
        if not summary.exists():
            output['status'] = 'RUNNING_OR_INTERRUPTED_CHECK_LIVE_PID'
    result = folder / result_path if result_path else None
    if result and result.exists():
        report = json.loads(result.read_text())
        output['evaluation'] = {'path': str(result), 'sha256': sha(result), 'nfes': {}}
        for nfe, value in report['nfe_results'].items():
            rows = value['rows']
            if stage == 2:
                metrics = value['summary']
            else:
                metrics = {'frames': len(rows), 'macro': {k: float(np.mean([r[k] for r in rows]))
                    for k in ('iou', 'f1', 'precision', 'recall', 'fine_recall_ceiling', 'active_parents')}}
            output['evaluation']['nfes'][nfe] = metrics
        output['evaluation']['continuous_geometry'] = 'NOT TESTED: native occupancy report has no CD/physical IoU/JSD'
        output['evaluation']['checkpoint_sha256'] = report.get('checkpoint_sha256')
        output['evaluation']['seconds'] = report.get('seconds')
    checkpoint = folder / 'checkpoints' / ('step010000.pt' if stage == 1 else 'resume_latest.pt')
    if checkpoint.exists():
        output['checkpoint'] = {'path': str(checkpoint), 'sha256': sha(checkpoint)}
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--c1-evaluation-root',type=Path)
    args = parser.parse_args()
    src = args.source_worktree.resolve()
    root = src / 'experiments/desescene_v_next'
    names = ('child_bridge_geometry_20k', 'child_bridge_position_20k',
             'static_context_fm_20k', 'static_context_bce_20k')
    runs = {name: inspect_run(root, name, 'eval900_seed20261031.json') for name in names}
    if args.c1_evaluation_root:
        for name in names[2:]:
            path=args.c1_evaluation_root.resolve()/(name+'_900.json')
            if path.exists():runs[name]=inspect_run(root,name,path)
    encoders = {name: inspect_run(src / 'experiments/dense_fidelity_v2_single_gpu', name,
                                'validation/update_010000.json', stage=1)
                for name in ('r0_gpu2_20k', 'e3_gpu3_20k', 'e1_gpu3_20k', 'e2_gpu2_20k')}
    paired = {}
    encoder_paired = {}
    reference_path=src/'experiments/dense_fidelity_v2_single_gpu/r0_gpu2_20k/validation/update_010000.json'
    if reference_path.exists():
        reference=json.loads(reference_path.read_text())['nfe_results']['2']['rows']
        for name in ('e1_gpu3_20k','e2_gpu2_20k','e3_gpu3_20k'):
            path=src/'experiments/dense_fidelity_v2_single_gpu'/name/'validation/update_010000.json'
            if path.exists():
                candidate=json.loads(path.read_text())['nfe_results']['2']['rows']
                encoder_paired[name]={metric:paired_delta(reference,candidate,metric=metric)
                                      for metric in ('iou','fine_recall_ceiling')}
    b1 = [root / name / 'eval900_seed20261031.json' for name in names[:2]]
    if all(p.exists() for p in b1):
        geometry, position = (json.loads(p.read_text()) for p in b1)
        for nfe in geometry['nfe_results']:
            paired[nfe] = paired_delta(position['nfe_results'][nfe]['rows'],
                                       geometry['nfe_results'][nfe]['rows'], nested=True)
    snapshot = {'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'source_worktree': str(src),
                'stage2_runs': runs, 'encoder_runs': encoders, 'b1_geometry_minus_position': paired,
                'encoders_minus_r0':encoder_paired,
                'matched_dynamic_fm_20k': 'NOT FOUND: B1 includes child decoder; C1 is static FM/BCE',
                'c1_comparison_limit': 'Both use predicted Stage1 NFE4, threshold0.5, scratch 20K; they do not isolate static vs dynamic PTv3'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'evidence.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    lines = ['# DeSeScene experiment evidence audit', '', snapshot['captured_at_utc'], '',
             'All values below are read from saved run artifacts. Existing runs and artifacts are unchanged.', '',
             '| Run | Training status | Updates | Exposures | Evaluation |', '|---|---|---:|---:|---|']
    for name, run in runs.items():
        latest = run.get('latest', {})
        lines.append(f"| {name} | {run['status']} | {latest.get('successful_updates', 'NOT FOUND')} | {latest.get('successful_frame_exposures', 'NOT FOUND')} | {'900-frame result present' if 'evaluation' in run else 'NOT COMPLETE'} |")
    lines += ['', '| B1 variant | NFE | Fine IoU | F1 |', '|---|---:|---:|---:|']
    for name in names[:2]:
        for nfe, value in runs[name].get('evaluation', {}).get('nfes', {}).items():
            lines.append(f"| {name} | {nfe} | {value['macro']['iou']:.6f} | {value['macro']['f1']:.6f} |")
    lines += ['', 'B1 has no demonstrated geometry-input benefit on these 900 frames. Both B1 models show decreasing native IoU with increasing NFE. This does not distinguish mean estimation from integration/model errors.', '',
              '| Encoder (single-GPU 20K cohort) | Parent IoU | F1 | Fine-GT ceiling |', '|---|---:|---:|---:|']
    for name, run in encoders.items():
        for value in run.get('evaluation', {}).get('nfes', {}).values():
            m = value['macro']
            lines.append(f"| {name} | {m['iou']:.6f} | {m['f1']:.6f} | {m['fine_recall_ceiling']:.6f} |")
    lines += ['', '| Encoder minus R0 | IoU delta (pp) | Group bootstrap 95% CI (pp) | Worst frame delta (pp) |',
              '|---|---:|---|---:|']
    for name,values in encoder_paired.items():
        x=values['iou'];ci=x['ci95']
        lines.append(f"| {name} | {100*x['mean']:+.4f} | [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] | {100*x['worst_frames'][0]['delta']:+.4f} |")
    lines += ['', 'These are single-training-seed results. E1 recovered by the final milestone and exceeds R0 macro IoU in this cohort; it is not rejected based on the earlier 12-frame or different-cohort result. Trained same-backend latency, fixed-K ranking and full Stage2 compatibility are still unverified.']
    lines += ['', '## Protocol limits', '',
              '- C1 static FM and BCE: frozen mature Stage1 EMA, predicted support, Stage1 NFE4, strict score >0.5, batch2, scratch 20K. C1 final results are pending.',
              '- No strictly matched original dynamic FM 20K control was found. B1 child-conditioned models cannot fill that role.',
              '- Old mature dynamic FM uses Oracle support with 25% empty neighbors and constant score1. Its source weights may initialize a separately identified matched fine-tuning cohort; its old metrics are not a predicted-support control.',
              '- Existing native reports do not contain CD, 0.5/0.2/0.1m physical IoU, or JSD. Those are NOT TESTED here.',
              '- Saved evaluation stage times are diagnostic and were partly collected during authorized GPU sharing; they are not dedicated performance benchmarks.',
              '- No new 100K training or M2/M3/M4 formal training is authorized by this audit.']
    (args.output / 'EVIDENCE_AUDIT.md').write_text('\n'.join(lines) + '\n')
    with (args.output / 'results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['run', 'nfe', 'frames', 'iou', 'f1'])
        for name, run in {**runs, **encoders}.items():
            for nfe, value in run.get('evaluation', {}).get('nfes', {}).items():
                writer.writerow([name, nfe, value['frames'], value['macro']['iou'], value['macro']['f1']])
    print(json.dumps({'output': str(args.output), 'encoder_iou': {n: r.get('evaluation', {}).get('nfes', {}).get('2', {}).get('macro', {}).get('iou') for n, r in encoders.items()}}, indent=2))


if __name__ == '__main__':
    main()

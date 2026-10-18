"""Matched Stage1-only control for the frozen 41-frame Stage2 evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import numpy as np
import torch
from eval_seq08_combinations import physical_grid_ids, support_counts
from export_stage2_snapshot import admit_gpu1, predicted_policy
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg, g = load_stage2_config(args.config)
    protocol = json.loads((args.evaluation / 'protocol.json').read_text())
    assert protocol['stage1'] == cfg['stage1']
    assert protocol['geometry_sha256'] == g.sha256
    assert [r['frame'] for r in protocol['frames']] == list(range(0, 4001, 100))
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'admission.json', admit_gpu1())
    atomic_json(args.output / 'protocol.json', {
        'paired_evaluation_protocol_sha256': file_sha(args.evaluation / 'protocol.json'),
        'stage1': cfg['stage1'], 'geometry_sha256': g.sha256,
        'noise_seed': 'identical scene_seed(sequence,frame,candidates.seed)',
        'metric': protocol['iou05'], 'frames': protocol['frames'],
        'controls': ['parent centers', 'all valid child centers inside selected parents'],
        'no_test_tuning': True, 'code_sha256': file_sha(__file__),
    })
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    adapter = FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval()
    rows = []
    start = time.perf_counter()
    for i, record in enumerate(protocol['frames']):
        frame = record['frame']
        meta = json.loads((args.reference / 'targets' / f'08_{frame:06d}.json').read_text())
        assert meta['input_sha256'] == file_sha(record['input'])
        assert meta['geometry_sha256'] == g.sha256
        with np.load(args.reference / 'targets' / f'08_{frame:06d}.npz', allow_pickle=False) as cache:
            target = cache['half_meter_ids'].copy()
        xyz = torch.from_numpy(input_xyz(record['input'])).to(device)
        pyramid = adapter.encode_features(xyz)
        seed = scene_seed('08', frame, cfg['candidates']['seed'])
        for nfe in (1, 2, 4):
            with torch.autocast('cuda', dtype=torch.float16):
                score, state = adapter.model.sample(xyz, nfe=nfe, condition=pyramid,
                    generator=torch.Generator(device=device).manual_seed(seed))
            if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()):
                raise ValueError('Nonfinite Stage1 control')
            ids, _, _ = select_occupancy_candidates(score[0, 0].float().cpu().numpy(), g,
                **predicted_policy(cfg)['selection'])
            coords = g.unflatten(ids)
            parent = g.centers(coords).astype(np.float32)
            children = coords[:, None, :] * np.asarray(g.child_split) + g.child_offsets()[None]
            fine = g.centers(children[g.valid_children(ids)], fine=True).astype(np.float32)
            for label, points in [('parent_centers', parent), ('all_children_filled', fine)]:
                counts = support_counts(physical_grid_ids(points, g), target)
                rows.append({'frame': frame, 'stage1_nfe': nfe, 'control': label,
                    'parents': len(ids), **counts,
                    'precision': counts['tp'] / max(counts['tp'] + counts['fp'], 1),
                    'recall': counts['tp'] / max(counts['tp'] + counts['fn'], 1)})
        atomic_json(args.output / 'per_frame.json', rows)
        print(f'control {i+1}/41 frame={frame:06d} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    summary = []
    for nfe in (1, 2, 4):
        for label in ('parent_centers', 'all_children_filled'):
            selected = [r for r in rows if r['stage1_nfe'] == nfe and r['control'] == label]
            assert len(selected) == 41
            summary.append({'stage1_nfe': nfe, 'control': label, 'frames': 41,
                **{key: float(np.mean([r[key] for r in selected])) for key in ('iou', 'precision', 'recall', 'parents')}})
    paired = []
    for row in json.loads((args.evaluation / 'results.json').read_text())['results']:
        base = next(r for r in summary if r['stage1_nfe'] == row['stage1_nfe'] and r['control'] == 'parent_centers')
        paired.append({**row, 'stage1_only_iou05': base['iou'],
            'delta_iou_percentage_points': 100 * (row['iou05_macro'] - base['iou'])})
    atomic_json(args.output / 'results.json', {'status': 'COMPLETE', 'summary': summary,
        'paired': paired, 'elapsed_seconds': time.perf_counter()-start,
        'encoder_forwards': adapter.encoder_forward_count})
    print(json.dumps({'controls': summary, 'paired': paired}, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Measure support ceiling of score-neighborhood/observed unions without GT selection."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from eval_stage1_occ import load_model
from train_stage1_occ import load_config
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_candidates import expand_parent_ids, parent_valid_field
from fpsgen.hier_occ_flow.storage import SparseShardDataset, atomic_json, file_sha
from fpsgen.hier_occ_flow.voxelization import unpack_children


def ceiling(candidate_ids, record, geometry):
    gt_ids = np.asarray(record['parent_ids'], dtype=np.int64)
    gt_children = unpack_children(record['child_occ_mask'], geometry)
    if not len(gt_ids) or not gt_children.any():
        return 1.0
    idx = np.searchsorted(np.asarray(candidate_ids, dtype=np.int64), gt_ids)
    present = idx < len(candidate_ids)
    possible = np.flatnonzero(present)
    present[possible] = candidate_ids[idx[possible]] == gt_ids[possible]
    return float(gt_children[present].sum() / gt_children.sum())


def deterministic_topk(score, valid_ids, k):
    k = min(int(k), len(valid_ids))
    values = score[valid_ids]
    cutoff = np.partition(values, len(values) - k)[len(values) - k]
    above = valid_ids[values > cutoff]
    tied = valid_ids[values == cutoff]
    return np.sort(np.concatenate((above, tied[:k - len(above)])))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-k', type=int, default=84963)
    p.add_argument('--alpha', type=float, default=1.5)
    p.add_argument('--max-frames', type=int)
    a = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in ('2', '3'):
        raise ValueError('Select exactly physical GPU2 or GPU3')
    admission = gpu_preflight((int(visible),))
    if admission['status'] != 'AVAILABLE':
        atomic_json(a.output, {'status': 'BLOCKED', 'gpu_admission': admission})
        return
    cfg = load_config(a.config)
    device = torch.device('cuda')
    model, g, saved = load_model(cfg, a.checkpoint, device, use_ema=True)
    ds = SparseShardDataset(cfg['development_dataset'], g, verify=False)
    count = min(len(ds), a.max_frames or len(ds))
    indices = np.linspace(0, len(ds) - 1, count, dtype=int)
    valid = parent_valid_field(g).ravel()
    records = []
    for n, index in enumerate(indices):
        record = ds[int(index)]
        path = Path(cfg['data_root']) / record['sequence'] / 'input_' / f"{record['frame']:06d}.npy"
        points = input_xyz(path)
        xyz = torch.from_numpy(points).to(device)
        with torch.autocast('cuda', dtype=torch.float16,
                            enabled=cfg['training']['precision'] == 'fp16'):
            condition = model.encode_condition(xyz)
        gen = torch.Generator(device=device).manual_seed(cfg['validation']['seed'] + int(index))
        with torch.autocast('cuda', dtype=torch.float16,
                            enabled=cfg['training']['precision'] == 'fp16'):
            score, _ = model.sample(xyz, generator=gen, nfe=cfg['validation']['nfes'][0], condition=condition)
        score = score.detach().float().cpu().numpy().reshape(-1)
        choices = np.flatnonzero(valid)
        top = deterministic_topk(score, choices, a.base_k)
        top_alpha = deterministic_topk(score, choices, int(np.ceil(a.base_k * a.alpha)))
        neighbors = expand_parent_ids(top, g, radius=1)
        fine, point_valid = g.voxelize_points(points)
        observed = np.unique(g.flatten(g.split_fine(fine[point_valid])[0])) if point_valid.any() else np.empty(0, np.int64)
        observed = observed[valid[observed]]
        union = np.union1d(np.union1d(top_alpha, neighbors), observed)
        # GT enters only after all candidate sets above are frozen.
        sets = {'topk': top, 'top_alpha': top_alpha, 'neighbors': neighbors,
                'observed': observed, 'union': union}
        result = {name: {'candidate_count': int(len(ids)),
                         'fine_recall_ceiling': ceiling(ids, record, g)}
                  for name, ids in sets.items()}
        records.append({'sequence': record['sequence'], 'frame': int(record['frame']),
                        'input_sha256': file_sha(path), 'input_points': int(len(points)),
                        'supports': result})
        if (n + 1) % 25 == 0 or n + 1 == count:
            print(f'support_union_scenes={n+1}/{count}', flush=True)
    summaries = {}
    for name in ('topk', 'top_alpha', 'neighbors', 'observed', 'union'):
        summaries[name] = {
            'candidate_count_mean': float(np.mean([r['supports'][name]['candidate_count'] for r in records])),
            'fine_recall_ceiling': {
                'mean': float(np.mean([r['supports'][name]['fine_recall_ceiling'] for r in records])),
                'median': float(np.median([r['supports'][name]['fine_recall_ceiling'] for r in records])),
                'p10': float(np.quantile([r['supports'][name]['fine_recall_ceiling'] for r in records], .1)),
                'p90': float(np.quantile([r['supports'][name]['fine_recall_ceiling'] for r in records], .9)),
            },
        }
    atomic_json(a.output, {
        'status': 'COMPLETE', 'frames': len(records), 'base_k': a.base_k,
        'alpha': a.alpha, 'nfe': int(cfg['validation']['nfes'][0]),
        'checkpoint_sha256': file_sha(a.checkpoint), 'config_sha256': file_sha(a.config),
        'development_manifest_sha256': file_sha(Path(cfg['development_dataset']) / 'manifest.json'),
        'geometry_sha256': g.sha256, 'gpu_admission': admission,
        'gt_used_only_for_evaluation_after_candidate_generation': True,
        'summaries': summaries, 'rows': records,
    })


if __name__ == '__main__':
    main()

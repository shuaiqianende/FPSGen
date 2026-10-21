"""Offline full-development PR and fixed-candidate-budget evaluation."""
from pathlib import Path
import hashlib

import numpy as np
import torch

from .dataset import input_xyz
from .stage1_curve_metrics import (
    fixed_budget_metrics, summarize_fixed_budget, threshold_curve_metrics,
)
from .stage1_evaluation import summarize
from .storage import SparseShardDataset, file_sha


@torch.no_grad()
def evaluate_stage1_curves(model, cfg, geometry, device, *, candidate_budgets,
                           max_frames=None, thresholds=None):
    was_training = model.training
    model.eval()
    try:
        dataset = SparseShardDataset(cfg['development_dataset'], geometry, verify=False)
        dataset_manifest_sha256 = file_sha(dataset.root / 'manifest.json')
        source_manifest_sha256 = dataset.meta.get('source_manifest_sha256')
        count = min(len(dataset), max_frames or len(dataset))
        indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
        val = cfg['validation']
        thresholds = np.asarray(
            thresholds if thresholds is not None else np.linspace(0.0, 1.0, 101),
            dtype=np.float32,
        ).astype(np.float64)
        valid = model.valid.detach().cpu().numpy().reshape(-1)
        per_threshold = {float(t): [] for t in thresholds}
        per_budget = {int(k): [] for k in candidate_budgets}
        records = []
        for pos, index in enumerate(indices):
            record = dataset[int(index)]
            path = Path(cfg['data_root']) / record['sequence'] / 'input_' / f"{record['frame']:06d}.npy"
            xyz = torch.from_numpy(input_xyz(path)).to(device)
            gt_digest = hashlib.sha256()
            gt_digest.update(np.ascontiguousarray(record['parent_ids'], dtype='<u4').tobytes())
            gt_digest.update(np.ascontiguousarray(record['child_occ_mask'], dtype=geometry.mask_dtype).tobytes())
            with torch.autocast(
                'cuda', dtype=torch.float16,
                enabled=cfg['training']['precision'] == 'fp16',
            ):
                condition = model.encode_condition(xyz)
            generator = torch.Generator(device=device).manual_seed(val['seed'] + int(index))
            with torch.autocast(
                'cuda', dtype=torch.float16,
                enabled=cfg['training']['precision'] == 'fp16',
            ):
                score, state = model.sample(
                    xyz, generator=generator, nfe=val['nfes'][0], condition=condition,
                )
            torch.cuda.synchronize(device)
            if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()):
                raise RuntimeError(f'Nonfinite Stage1 output on {record["sequence"]}/{record["frame"]:06d}')
            score = score.detach().float().cpu().numpy().reshape(-1)
            curve = threshold_curve_metrics(score, record, valid, thresholds)
            fixed = fixed_budget_metrics(score, record, valid, candidate_budgets)
            for row in curve:
                per_threshold[row['threshold']].append(row)
            for row in fixed:
                per_budget[row['requested_k']].append(row)
            at_frozen = min(curve, key=lambda r: abs(r['threshold'] - val['threshold']))
            records.append({
                'sequence': record['sequence'], 'frame': int(record['frame']),
                'input_path': str(path), 'input_sha256': file_sha(path),
                'gt_sha256': gt_digest.hexdigest(),
                'threshold_metrics': at_frozen,
                'threshold_curve': curve,
                'fixed_budget': fixed,
            })
            if pos == 0 or (pos + 1) % 10 == 0 or pos + 1 == count:
                print(f'curve_eval_scenes={pos + 1}/{count}', flush=True)

        threshold_summary = {}
        for threshold, rows in per_threshold.items():
            threshold_summary[f'{threshold:.6f}'] = summarize(rows)
        fixed_summary = {
            str(k): summarize_fixed_budget(rows) for k, rows in per_budget.items()
        }
        return {
            'frames': len(records), 'objective': model.objective,
            'development_manifest_sha256': dataset_manifest_sha256,
            'source_manifest_sha256': source_manifest_sha256,
            'nfe': int(val['nfes'][0]), 'frozen_threshold': float(val['threshold']),
            'threshold_semantics': 'strict_score_greater_than_threshold',
            'thresholds': [float(t) for t in thresholds],
            'threshold_summaries': threshold_summary,
            'fixed_candidate_budgets': [int(k) for k in candidate_budgets],
            'fixed_budget_source': 'R0 10K threshold-0.5 candidate-count quartiles, frozen before cross-variant comparison',
            'fixed_budget_summaries': fixed_summary,
            'rows': records,
        }
    finally:
        model.train(was_training)

"""Run full-dev Stage1 PR-curve and fixed-K diagnostics for a pilot checkpoint."""
import argparse
import os
from pathlib import Path

import torch

from eval_stage1_occ import load_model
from train_dense_encoder_pilot import load_config
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_curve_evaluation import evaluate_stage1_curves
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-budgets', type=int, nargs='+', required=True)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--threshold-count', type=int, default=101)
    args = parser.parse_args()
    if args.threshold_count < 2:
        parser.error('--threshold-count must be at least 2')

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in ('2', '3'):
        raise ValueError('Select exactly physical GPU2 or GPU3 with CUDA_VISIBLE_DEVICES')
    admission = gpu_preflight((int(visible),))
    if admission['status'] != 'AVAILABLE':
        atomic_json(args.output, {'status': 'BLOCKED', 'resources': admission})
        return

    cfg = load_config(args.config)
    device = torch.device('cuda')
    model, geometry, checkpoint = load_model(cfg, args.checkpoint, device, use_ema=True)
    thresholds = [i / (args.threshold_count - 1) for i in range(args.threshold_count)]
    result = evaluate_stage1_curves(
        model, cfg, geometry, device,
        candidate_budgets=args.candidate_budgets,
        max_frames=args.max_frames, thresholds=thresholds,
    )
    result.update({
        'status': 'COMPLETE', 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': file_sha(args.checkpoint),
        'config_sha256': file_sha(args.config),
        'evaluation_script_sha256': file_sha(Path(__file__)),
        'successful_updates': int(checkpoint['successful_updates']),
        'ema': True, 'gpu_admission': admission,
    })
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()

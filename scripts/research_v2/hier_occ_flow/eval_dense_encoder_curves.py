"""Run full-dev Stage1 PR-curve and fixed-K diagnostics for a pilot checkpoint."""
import argparse
import os
from pathlib import Path

import torch

from eval_stage1_occ import load_model
from train_stage1_occ import load_config
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_curve_evaluation import evaluate_stage1_curves
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha
from fpsgen.hier_occ_flow.full_fp16_inference import explicit_fp16_copy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-budgets', type=int, nargs='+', required=True)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--threshold-count', type=int, default=101)
    parser.add_argument('--allow-shared-gpu', action='store_true',
                        help='Explicitly allow co-running when enough VRAM remains; records existing GPU processes.')
    parser.add_argument('--minimum-free-gib', type=float, default=8.0)
    parser.add_argument('--encoder-explicit-fp16', action='store_true',
                        help='Convert the loaded inference encoder copy to explicit FP16 before evaluation.')
    parser.add_argument('--tensorrt-engine', type=Path,
                        help='Optional static FP16 TensorRT engine replacing one complete backbone.')
    parser.add_argument('--tensorrt-module', choices=('local_backbone', 'voxel_backbone'),
                        default='local_backbone')
    args = parser.parse_args()
    if args.threshold_count < 2:
        parser.error('--threshold-count must be at least 2')

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in ('2', '3'):
        raise ValueError('Select exactly physical GPU2 or GPU3 with CUDA_VISIBLE_DEVICES')
    admission = gpu_preflight((int(visible),), minimum_free_gib=args.minimum_free_gib,
                              allow_shared_compute=args.allow_shared_gpu)
    if admission['status'] != 'AVAILABLE':
        atomic_json(args.output, {'status': 'BLOCKED', 'resources': admission})
        return

    cfg = load_config(args.config)
    device = torch.device('cuda')
    model, geometry, checkpoint = load_model(cfg, args.checkpoint, device, use_ema=True)
    explicit_fp16_audit = None
    precision_hooks = []
    if args.encoder_explicit_fp16 or args.tensorrt_engine is not None:
        explicit_fp16_audit, precision_hooks = explicit_fp16_copy(model.encoder)
        model.encoder._v6_explicit_fp16 = True
    trt_adapter = None
    if args.tensorrt_engine is not None:
        from fpsgen.hier_occ_flow.fidelity_trt_adapter import FidelityTensorRTBackbone
        trt_adapter = FidelityTensorRTBackbone(args.tensorrt_engine, device=device)
        setattr(model.encoder, args.tensorrt_module, trt_adapter)
    thresholds = [i / (args.threshold_count - 1) for i in range(args.threshold_count)]
    result = evaluate_stage1_curves(
        model, cfg, geometry, device,
        candidate_budgets=args.candidate_budgets,
        max_frames=args.max_frames, thresholds=thresholds,
        encoder_autocast=not (args.encoder_explicit_fp16 or args.tensorrt_engine is not None),
    )
    result.update({
        'status': 'COMPLETE', 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': file_sha(args.checkpoint),
        'config_sha256': file_sha(args.config),
        'evaluation_script_sha256': file_sha(Path(__file__)),
        'successful_updates': int(checkpoint['successful_updates']),
        'ema': True, 'gpu_admission': admission,
        'encoder_explicit_fp16': bool(args.encoder_explicit_fp16 or args.tensorrt_engine is not None),
        'explicit_fp16_audit': explicit_fp16_audit,
        'tensorrt_engine': str(args.tensorrt_engine.resolve()) if args.tensorrt_engine else None,
        'tensorrt_engine_sha256': trt_adapter.engine_sha256 if trt_adapter is not None else None,
        'tensorrt_module': args.tensorrt_module if trt_adapter is not None else None,
    })
    for hook in precision_hooks:
        hook.remove()
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()

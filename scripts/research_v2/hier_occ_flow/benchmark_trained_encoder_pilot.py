"""Benchmark the trained EMA Dense Encoder on a frozen 5+20 dev-frame set."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from eval_stage1_occ import load_model
from train_stage1_occ import load_config
from fpsgen.hier_occ_flow.full_fp16_inference import explicit_fp16_copy, parameter_dtype_audit
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_curve_evaluation import evaluate_stage1_curves
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha
from fpsgen.hier_occ_flow.storage import SparseShardDataset


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stats(values):
    values = [float(x) for x in values]
    return {
        'count': len(values), 'mean_ms': statistics.mean(values),
        'std_ms': statistics.pstdev(values), 'median_ms': statistics.median(values),
        'p90_ms': percentile(values, 90), 'p95_ms': percentile(values, 95),
        'min_ms': min(values), 'max_ms': max(values),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True, choices=(2, 3))
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--precision', choices=('explicit_fp16', 'selective_fp16', 'amp_original'),
                        default='explicit_fp16')
    parser.add_argument('--curve-output', type=Path,
                        help='Optional full-dev PR/fixed-K output; run separately from timing.')
    parser.add_argument('--candidate-budgets', type=int, nargs='+',
                        default=[78694, 84963, 94087])
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible != str(args.physical_gpu):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly --physical-gpu')
    admission = gpu_preflight((args.physical_gpu,))
    if admission['status'] != 'AVAILABLE':
        atomic_json(args.output, {'status': 'BLOCKED', 'gpu_admission': admission})
        return

    cfg = load_config(args.config)
    device = torch.device('cuda:0')
    model, geometry, checkpoint = load_model(cfg, args.checkpoint, device, use_ema=True)
    precision_audit = None
    hooks = []
    if args.precision in ('explicit_fp16', 'selective_fp16'):
        precision_audit, hooks = explicit_fp16_copy(
            model.encoder, selective_norm=args.precision == 'selective_fp16')
        model.encoder._v6_explicit_fp16 = True

    dataset = SparseShardDataset(cfg['development_dataset'], geometry, verify=False)
    selected = np.linspace(0, len(dataset) - 1, 25, dtype=int)
    records = [dataset[int(i)] for i in selected]
    frame_data = []
    for record in records:
        path = Path(cfg['data_root']) / record['sequence'] / 'input_' / f"{record['frame']:06d}.npy"
        xyz = input_xyz(path)
        frame_data.append({
            'sequence': record['sequence'], 'frame': int(record['frame']),
            'path': str(path), 'input_sha256': hashlib.sha256(xyz.tobytes()).hexdigest(),
            'xyz_cpu': xyz,
        })
    input_manifest = [{k: v for k, v in row.items() if k != 'xyz_cpu'} for row in frame_data]
    input_manifest_sha256 = hashlib.sha256(
        json.dumps(input_manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()

    warmup_rows = []
    timed_rows = []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    for run in range(args.rounds):
        for idx, item in enumerate(frame_data[:5]):
            xyz = torch.from_numpy(np.ascontiguousarray(item['xyz_cpu'])).to(device)
            with torch.inference_mode():
                if args.precision == 'amp_original':
                    with torch.autocast('cuda', dtype=torch.float16):
                        condition = model.encode_condition(xyz)
                else:
                    condition = model.encode_condition(xyz)
            torch.cuda.synchronize(device)
            warmup_rows.append({'round': run + 1, 'sequence': item['sequence'],
                                'frame': item['frame'], 'condition_shapes': [list(x.shape) for x in condition]})
            del xyz, condition

        for item in frame_data[5:]:
            xyz_cpu = torch.from_numpy(np.ascontiguousarray(item['xyz_cpu']))
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            xyz = xyz_cpu.to(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode():
                if args.precision == 'amp_original':
                    with torch.autocast('cuda', dtype=torch.float16):
                        condition = model.encode_condition(xyz)
                else:
                    condition = model.encode_condition(xyz)
            end.record()
            torch.cuda.synchronize(device)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            network_ms = float(start.elapsed_time(end))
            finite = all(bool(torch.isfinite(x).all()) for x in condition)
            timed_rows.append({
                'round': run + 1, 'sequence': item['sequence'], 'frame': item['frame'],
                'input_sha256': item['input_sha256'], 'generation_encoder_wall_ms': wall_ms,
                'network_cuda_event_ms': network_ms,
                'condition_dtypes': [str(x.dtype) for x in condition],
                'condition_shapes': [list(x.shape) for x in condition], 'finite': finite,
            })
            if not finite:
                raise RuntimeError(f'Nonfinite Encoder output at {item["sequence"]}/{item["frame"]:06d}')
            del xyz, condition, xyz_cpu

    cudnn_runtime_version = None
    cudnn_runtime_query_error = None
    try:
        cudnn_runtime_version = torch.backends.cudnn.version()
    except RuntimeError as exc:
        # The inference loop has already completed. Preserve benchmark data,
        # but make a PyTorch/runtime cuDNN mismatch explicit in the artifact.
        cudnn_runtime_query_error = f'{type(exc).__name__}: {exc}'

    result = {
        'status': 'COMPLETE', 'variant': cfg['model']['encoder_variant'],
        'precision': args.precision, 'physical_gpu': args.physical_gpu,
        'device_name': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'cudnn_runtime_version': cudnn_runtime_version,
        'cudnn_runtime_query_error': cudnn_runtime_query_error,
        'checkpoint_sha256': file_sha(args.checkpoint),
        'config_sha256': file_sha(args.config),
        'benchmark_script_sha256': file_sha(Path(__file__)),
        'successful_updates': int(checkpoint['successful_updates']), 'ema': True,
        'geometry_sha256': geometry.sha256,
        'candidate_budgets_for_separate_evaluation': args.candidate_budgets,
        'input_manifest_sha256': input_manifest_sha256,
        'warmup_frames': input_manifest[:5],
        'benchmark_frame_manifest': input_manifest[5:],
        'warmup_repetitions': warmup_rows, 'per_frame': timed_rows,
        'summary': {
            'network_cuda_event': stats([r['network_cuda_event_ms'] for r in timed_rows]),
            'encoder_wall_including_h2d': stats([r['generation_encoder_wall_ms'] for r in timed_rows]),
        },
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
        'parameter_dtype_audit': parameter_dtype_audit(model.encoder),
        'explicit_precision_audit': precision_audit, 'gpu_admission': admission,
    }
    for hook in hooks:
        hook.remove()
    atomic_json(args.output, result)

    if args.curve_output:
        curve = evaluate_stage1_curves(
            model, cfg, geometry, device, candidate_budgets=args.candidate_budgets,
        )
        curve.update({
            'status': 'COMPLETE', 'checkpoint_sha256': file_sha(args.checkpoint),
            'config_sha256': file_sha(args.config),
            'evaluation_script_sha256': file_sha(Path(__file__)),
            'successful_updates': int(checkpoint['successful_updates']), 'ema': True,
            'geometry_sha256': geometry.sha256, 'gpu_admission': admission,
        })
        atomic_json(args.curve_output, curve)


if __name__ == '__main__':
    main()

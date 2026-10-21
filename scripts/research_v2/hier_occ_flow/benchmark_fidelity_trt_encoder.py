#!/usr/bin/env python
"""Benchmark a TensorRT FP16 Stage1 backbone inside the real encoder pipeline."""
from __future__ import annotations

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
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.fidelity_trt_adapter import FidelityTensorRTBackbone
from fpsgen.hier_occ_flow.full_fp16_inference import explicit_fp16_copy, parameter_dtype_audit
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha, SparseShardDataset


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_ms": float(a.mean()), "std_ms": float(a.std()),
            "median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "max_ms": float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--module', choices=('local_backbone', 'voxel_backbone'), required=True)
    parser.add_argument('--physical-gpu', type=int, choices=(2, 3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=25)
    args = parser.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly --physical-gpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    admission = gpu_preflight((args.physical_gpu,))
    if admission.get('status') != 'AVAILABLE':
        atomic_json(args.output, {'status': 'BLOCKED', 'gpu_admission': admission})
        return

    cfg = load_config(args.config)
    model, geometry, checkpoint = load_model(cfg, args.checkpoint, torch.device('cuda:0'), use_ema=True)
    precision_audit, hooks = explicit_fp16_copy(model.encoder)
    model.encoder._v6_explicit_fp16 = True
    adapter = FidelityTensorRTBackbone(args.engine, device=torch.device('cuda:0'))
    setattr(model.encoder, args.module, adapter)

    dataset = SparseShardDataset(cfg['development_dataset'], geometry, verify=False)
    indices = np.linspace(0, len(dataset) - 1, min(args.max_frames, len(dataset)), dtype=int)
    items = []
    for index in indices:
        row = dataset[int(index)]
        path = Path(cfg['data_root']) / str(row['sequence']) / 'input_' / f"{int(row['frame']):06d}.npy"
        xyz = input_xyz(path)
        items.append({'sequence': str(row['sequence']), 'frame': int(row['frame']),
                      'path': str(path), 'input_sha256': hashlib.sha256(xyz.tobytes()).hexdigest(),
                      'xyz': xyz})
    manifest = [{key: value for key, value in row.items() if key != 'xyz'} for row in items]
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                  separators=(',', ':')).encode()).hexdigest()

    timing, warmups = [], []
    device = torch.device('cuda:0')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    for index, item in enumerate(items):
        xyz_cpu = torch.from_numpy(np.ascontiguousarray(item['xyz']))
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        xyz = xyz_cpu.to(device, dtype=torch.float32)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            condition = model.encode_condition(xyz)
        end.record()
        torch.cuda.synchronize(device)
        row = {'sequence': item['sequence'], 'frame': item['frame'],
               'input_sha256': item['input_sha256'],
               'encoder_wall_including_h2d_ms': (time.perf_counter() - wall_start) * 1000,
               'encoder_cuda_event_ms': float(start.elapsed_time(end)),
               'pyramid_shapes': [list(value.shape) for value in condition],
               'pyramid_dtypes': [str(value.dtype) for value in condition],
               'finite': all(bool(torch.isfinite(value).all()) for value in condition)}
        if not row['finite']:
            raise RuntimeError(f"Non-finite TensorRT encoder output at {item['sequence']}/{item['frame']:06d}")
        if index < 5:
            row['warmup'] = True
            warmups.append(row)
        else:
            row['warmup'] = False
            timing.append(row)
        del xyz, xyz_cpu, condition

    for hook in hooks:
        hook.remove()
    result = {
        'status': 'COMPLETE', 'variant': cfg['model']['encoder_variant'],
        'backend': 'TensorRT FP16 backbone embedded in PyTorch Stage1 encoder',
        'module': args.module, 'physical_gpu': args.physical_gpu,
        'gpu_admission': admission, 'gpu_name': torch.cuda.get_device_name(device),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'config_sha256': file_sha(args.config), 'checkpoint_sha256': file_sha(args.checkpoint),
        'successful_updates': int(checkpoint['successful_updates']), 'ema': True,
        'geometry_sha256': geometry.sha256, 'engine_sha256': adapter.engine_sha256,
        'engine_path': str(args.engine.resolve()), 'input_manifest_sha256': manifest_sha,
        'frame_manifest': manifest, 'warmup_rows': warmups, 'per_frame': timing,
        'summary': {'encoder_cuda_event': summarize([row['encoder_cuda_event_ms'] for row in timing]),
                    'encoder_wall_including_h2d': summarize([row['encoder_wall_including_h2d_ms'] for row in timing])},
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
        'parameter_dtype_audit': parameter_dtype_audit(model.encoder),
        'explicit_fp16_audit': precision_audit,
        'notes': ['First five registered development examples are warm-up only; remaining examples are timed.',
                  'This measures the full Stage1 encoder path and includes TensorRT handoff/enqueue in CUDA timing.',
                  'Run the matching PyTorch explicit-FP16 benchmark with the same config, checkpoint and manifest.'],
    }
    atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'module': args.module,
                      'timed_frames': len(timing), 'summary': result['summary'],
                      'peak_allocated_gib': result['peak_allocated_gib']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Export one trained dense-fidelity Stage1 backbone using a real scan.

The script captures the selected backbone's actual FP16 feature input from the
registered Stage1 pipeline, exports only that backbone, and saves eager
reference outputs. Large ONNX and tensor artifacts belong in the local
experiment directory and must not be committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval_stage1_occ import load_model
from train_stage1_occ import load_config
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.full_fp16_inference import explicit_fp16_copy
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import file_sha


class BackboneExportWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, features):
        output = self.backbone(features)
        return tuple(output) if isinstance(output, (tuple, list)) else (output,)


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/data-12/M2024-HWZ/KITTI_Odometry'))
    parser.add_argument('--sequence', default='00')
    parser.add_argument('--frame', type=int, default=100)
    parser.add_argument('--module', choices=('local_backbone', 'voxel_backbone'), required=True)
    parser.add_argument('--physical-gpu', type=int, choices=(2, 3), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--precision', choices=('explicit_fp16', 'amp_original'), default='explicit_fp16')
    args = parser.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly --physical-gpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    admission = gpu_preflight((args.physical_gpu,))
    if admission.get('status') != 'AVAILABLE':
        raise RuntimeError(f'GPU admission failed: {admission}')
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite export directory: {args.output_dir}')

    cfg = load_config(args.config)
    model, geometry, checkpoint = load_model(cfg, args.checkpoint, torch.device('cuda:0'), use_ema=True)
    precision_audit, hooks = (None, [])
    if args.precision == 'explicit_fp16':
        precision_audit, hooks = explicit_fp16_copy(model.encoder)
        model.encoder._v6_explicit_fp16 = True
    module = getattr(model.encoder, args.module)
    captured = {}

    def capture_input(_module, inputs):
        if 'features' in captured:
            raise RuntimeError('Backbone was called more than once during input capture')
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError('Backbone input is not a Tensor')
        captured['features'] = inputs[0].detach().clone().contiguous()

    handle = module.register_forward_pre_hook(capture_input)
    source = args.data_root / args.sequence / 'input_' / f'{args.frame:06d}.npy'
    xyz_np = input_xyz(source).astype(np.float32, copy=False)
    if not np.isfinite(xyz_np).all():
        raise ValueError(f'Non-finite LiDAR input: {source}')
    xyz = torch.from_numpy(np.ascontiguousarray(xyz_np)).to('cuda:0', dtype=torch.float32)
    with torch.inference_mode():
        if args.precision == 'amp_original':
            with torch.autocast('cuda', dtype=torch.float16):
                pyramid = model.encode_condition(xyz)
        else:
            pyramid = model.encode_condition(xyz)
    torch.cuda.synchronize()
    handle.remove()
    for hook in hooks:
        hook.remove()
    if 'features' not in captured:
        raise RuntimeError('Backbone input was not captured')

    sample = captured['features']
    if not sample.is_cuda or not sample.is_contiguous():
        raise RuntimeError('Captured feature input must be contiguous and CUDA resident')
    if args.precision == 'explicit_fp16' and sample.dtype != torch.float16:
        raise RuntimeError(f'Explicit FP16 capture yielded {sample.dtype}')
    wrapper = BackboneExportWrapper(module).eval()
    with torch.inference_mode():
        reference = wrapper(sample)
    torch.cuda.synchronize()
    if not all(bool(torch.isfinite(value).all()) for value in reference):
        raise RuntimeError('Backbone reference contains non-finite values')

    args.output_dir.mkdir(parents=True)
    input_path = args.output_dir / 'real_input.npy'
    np.save(input_path, sample.cpu().numpy())
    reference_paths = []
    for index, value in enumerate(reference):
        path = args.output_dir / f'pytorch_output_{index}.npy'
        np.save(path, value.cpu().numpy())
        reference_paths.append(path)
    onnx_path = args.output_dir / f'{args.module}_{args.precision}.onnx'
    output_names = [f'feature_{index}' for index in range(len(reference))]
    torch.onnx.export(
        wrapper, sample, str(onnx_path), export_params=True, opset_version=args.opset,
        do_constant_folding=True, input_names=['features'], output_names=output_names,
        dynamic_axes=None,
    )
    report = {
        'status': 'EXPORTED_UNQUANTIZED', 'physical_gpu': args.physical_gpu,
        'gpu_admission': admission, 'module': args.module, 'precision': args.precision,
        'sequence': args.sequence, 'frame': args.frame, 'input_path': str(source),
        'input_file_sha256': file_sha(source), 'input_xyz_sha256': array_sha(xyz_np),
        'input_point_count': len(xyz_np), 'config': str(args.config.resolve()),
        'config_sha256': file_sha(args.config), 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': file_sha(args.checkpoint),
        'successful_updates': int(checkpoint['successful_updates']), 'ema': True,
        'geometry_sha256': geometry.sha256, 'input_shape': list(sample.shape),
        'input_dtype': str(sample.dtype), 'input_stride': list(sample.stride()),
        'output_shapes': [list(value.shape) for value in reference],
        'output_dtypes': [str(value.dtype) for value in reference],
        'output_names': output_names, 'onnx_opset': args.opset,
        'onnx_path': str(onnx_path), 'onnx_sha256': file_sha(onnx_path),
        'saved_input': str(input_path), 'saved_input_sha256': file_sha(input_path),
        'pytorch_reference_paths': [str(path) for path in reference_paths],
        'pytorch_reference_sha256': [file_sha(path) for path in reference_paths],
        'encoder_pyramid_shapes': [list(value.shape) for value in pyramid],
        'precision_audit': precision_audit,
        'notes': ['Export is unquantized. Successful ONNX export does not establish TensorRT support or INT8 kernel execution.'],
    }
    (args.output_dir / 'export_protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'module', 'precision', 'onnx_path',
                     'onnx_sha256', 'input_shape', 'output_shapes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()

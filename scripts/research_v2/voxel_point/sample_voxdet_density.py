"""XYZ-only Stage1 checkpoint inference and exactly180k Gaussian anchors."""

import json
from pathlib import Path
import time
import ast
import inspect
import sys
import textwrap
import types

import numpy as np
import torch

from common import parser, read_config, configure_runtime, load_checkpoint, file_sha, save_json
from fpsgen.voxel_point.dataset import load_xyz
from fpsgen.voxel_point.voxdet_density import VoxDetDensity
from fpsgen.voxel_point.count_allocator import allocate_counts
from fpsgen.voxel_point.anchor_sampler import sample_gaussian_anchors


def enable_cpu_tokenizer(tokenizer):
    """Only adapt official get_device()=-1 to torch.device('cpu'); retain ops."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(type(tokenizer).forward)))
    class DeviceSelection(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'get_device':
                return ast.copy_location(ast.Attribute(value=node.func.value, attr='device', ctx=ast.Load()), node)
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(DeviceSelection().visit(tree))
    namespace = vars(sys.modules[type(tokenizer).__module__]).copy()
    exec(compile(tree, '<CPU-only VoxDet device compatibility>', 'exec'), namespace)
    tokenizer.forward = types.MethodType(namespace['forward'], tokenizer)


def main():
    p = parser(__doc__)
    p.add_argument('--input-manifest', required=True, type=Path)
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--memory-limit-gib', type=float, default=0,
                   help='Cap this inference allocator when sharing a training GPU')
    args = p.parse_args()
    cfg, spec, sha = read_config(args.config)
    frames = json.loads(args.input_manifest.read_text())
    if not frames or any(set(row) != {'id', 'input'} for row in frames):
        raise ValueError('Inference manifest must contain only id/input, never GT')
    if len({row['id'] for row in frames}) != len(frames):
        raise ValueError('Duplicate frame IDs')
    if args.max_frames:
        frames = frames[:args.max_frames]
    device = torch.device('cpu') if args.device == 'cpu' else configure_runtime()[0]
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(cfg['seed'])
    if args.memory_limit_gib and device.type == 'cuda':
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = args.memory_limit_gib * 1024**3 / total
        if not 0 < fraction <= 1:
            raise ValueError('Invalid allocator memory cap')
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    model = VoxDetDensity(spec, **cfg['model']).to(device).eval()
    if device.type == 'cpu':
        enable_cpu_tokenizer(model.lidar_tokenizer)
    checkpoint = load_checkpoint(model, cfg['stage1_checkpoint'], spec, stage='stage1')
    if checkpoint['mechanics_smoke_only'] and not args.smoke:
        raise ValueError('Formal preview requires a formally trained Stage1 checkpoint')
    checkpoint_sha = file_sha(cfg['stage1_checkpoint'])
    del checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index, row in enumerate(frames):
            frame_id = row['id']
            if Path(frame_id).name != frame_id or frame_id in ('.', '..'):
                raise ValueError('Unsafe frame ID')
            xyz = load_xyz(row['input']).to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type=='cuda' and cfg['runtime']['amp']):
                density = model(xyz)['pred_density'][0, 0]
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_ms = (time.perf_counter() - started) * 1000
            # CPU source allocation leaves the training GPU's memory available.
            density = density.cpu()
            del xyz
            seed = cfg['seed'] + index
            generator = torch.Generator().manual_seed(seed)
            counts = allocate_counts(density, spec, generator=generator)
            anchors, ids = sample_gaussian_anchors(spec, counts, generator=generator)
            assert anchors.shape == (180000, 3) and int(counts.sum()) == 180000
            assert bool(spec.valid_points(anchors).all())
            folder = args.output / frame_id
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'voxel_density.npz', density=density.numpy())
            np.savez_compressed(folder/'allocated_counts.npz', counts=counts.numpy())
            np.save(folder/'source_anchors.npy', anchors.numpy())
            np.save(folder/'voxel_ids.npy', ids.numpy())
            record = {'id':frame_id, 'input':row['input'], 'seed':seed,
                      'checkpoint_sha256':checkpoint_sha, 'config_sha256':sha,
                      'geometry':spec.to_dict(), 'allocated_points':180000,
                      'gt_used':False, 'teacher_used':False, 'sinkhorn_used':False,
                      'pointflow_used':False, 'stage1_network_evaluations':1,
                      'device':str(device), 'precision':'fp32' if device.type=='cpu' else 'fp16_amp',
                      'density_sum':float(density.sum()), 'stage1_inference_ms':inference_ms}
            save_json(folder/'manifest.json', record)
            summaries.append(record)
            print(frame_id, f'{inference_ms:.1f}ms', f'mass={float(density.sum()):.0f}', flush=True)
    save_json(args.output/'inference_summary.json', {'frames':summaries,
              'physical_visible':__import__('os').environ['CUDA_VISIBLE_DEVICES'],
              'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0,
              'allocator_limit_gib':args.memory_limit_gib,
              'checkpoint':cfg['stage1_checkpoint'], 'gt_used':False})


if __name__ == '__main__':
    main()

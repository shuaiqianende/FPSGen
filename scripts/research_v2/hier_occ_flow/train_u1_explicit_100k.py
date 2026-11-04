"""Independent U1 scratch run: full training split, FP16 model, FP32 masters."""
import argparse
import copy
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import FP32Masters, apply_explicit_policy, update_fp32_ema
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.morphology import MORPH_DATA_VERSION
from fpsgen.hier_occ_flow.stage1_batch_transition import batch_indices
from fpsgen.hier_occ_flow.stage1_network import OccupancyUNet
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy, ids_to_dense_occupancy, occupancy_flow_pair, masked_occ_loss
from fpsgen.hier_occ_flow.storage import SparseShardDataset, atomic_json, file_sha
from fpsgen.hier_occ_flow.trainer import atomic_checkpoint, restore_rng, rng_state, seed_all

CONFIG = ROOT / 'configs/research_v2/hier_occ_flow/u1_fulltrain_explicit_fp16_100k.yaml'
BASE = ROOT / 'experiments/desescene_u1_fulltrain_explicit_fp16_100k_v1'


def source_sha():
    names = ('stage1_occupancy.py', 'stage1_network.py', 'patch_unet_condition.py',
             'explicit_fp16_training_policy.py', 'geometry.py', 'stage1_batch_transition.py')
    h = hashlib.sha256(Path(__file__).read_bytes())
    for name in names:
        h.update(name.encode())
        h.update((ROOT / 'fpsgen/hier_occ_flow' / name).read_bytes())
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('gate', 'smoke', 'formal'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('Only physical GPU2 is registered')
    cfg = yaml.safe_load(CONFIG.read_text())
    t, m = cfg['training'], cfg['model']
    if (t['batch_per_gpu'], t['successful_updates'], t['successful_frame_exposures'], t['seed']) != (2, 50000, 100000, 20261032):
        raise ValueError('Unexpected training protocol')
    out = BASE / args.mode
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'checkpoints/resume_latest.pt'
    progress = out / 'status.json'
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False
        geometry = SparseHierarchyGeometry(**cfg['geometry'])
        dataset = SparseShardDataset(cfg['train_dataset'], geometry, data_version=MORPH_DATA_VERSION)
        sequences = {x['sequence'].decode() for x in dataset.index}
        if len(dataset) != 19130 or sequences != {f'{i:02}' for i in (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)}:
            raise ValueError('Full training split identity mismatch')
        if dataset.meta.get('parent_ids_unchanged') is not True:
            raise ValueError('Stage1 requires parent-preserving target dataset')
        identity = {'experiment': cfg['experiment'], 'config_sha256': file_sha(CONFIG),
                    'manifest_sha256': file_sha(Path(cfg['train_dataset']) / 'manifest.json'),
                    'source_sha256': source_sha(), 'initialization': 'scratch',
                    'target': 'original surface-hit Parent IDs, morphology masks unused',
                    'physical_gpu': 2, 'batch': 2, 'seed': t['seed'],
                    'precision': t['precision'], 'compile_target': t['compile_target']}
        proofs = []
        if args.mode == 'formal':
            for name in ('gate', 'smoke'):
                proof = json.loads((BASE / name / 'status.json').read_text())
                if proof.get('status') != 'PASS' or proof.get('identity') != identity:
                    raise ValueError(f'{name} has not passed for the same identity')
                if name == 'smoke' and proof.get('updates') != 32:
                    raise ValueError('The independent 32-update smoke is incomplete')
                proofs.append(proof)
        if checkpoint.exists() and not args.resume:
            raise FileExistsError('Use --resume for an existing run')
        seed_all(t['seed'])
        # Flow's independent RNG matches the previous scratch U1+A cohort.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(t['seed'] + 104729)
            template = OccupancyUNet('fm', widths=tuple(m['widths']), activation_checkpointing=False)
            flow_init = copy.deepcopy(template.state_dict())
        del template
        model = Stage1DirectOccupancy(geometry, objective='fm', widths=m['widths'],
            activation_checkpointing=m['activation_checkpointing'], encoder_variant=m['encoder_variant']).cuda()
        model.flow.load_state_dict(flow_init, strict=True)
        flow_initialization_sha = state_sha(flow_init)
        if flow_initialization_sha != m['expected_flow_initialization_sha256']:
            raise ValueError('Shared FM initialization differs from the U1+A scratch cohort')
        del flow_init
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        initialization = {'model_fp32_sha256': state_sha(initial),
                          'flow_fp32_sha256': flow_initialization_sha,
                          'pretrained_loaded': False}
        if any(proof.get('initialization') != initialization for proof in proofs):
            raise ValueError('Scratch initialization differs from engineering gate/smoke')
        masters = FP32Masters(model, initial)
        precision = apply_explicit_policy(model)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        ema = {name: value.detach().clone() for name, value in initial.items()}
        groups = masters.optimizer_groups([
            {'params': list(model.encoder.parameters()), 'lr': t['lr_encoder'], 'name': 'encoder'},
            {'params': list(model.flow.parameters()), 'lr': t['lr_flow'], 'name': 'flow'}])
        optimizer = torch.optim.AdamW(groups, weight_decay=t['weight_decay'])
        def multiplier(step):
            return (step + 1) / 1000 if step < 1000 else .5 * (1 + math.cos(math.pi * min(1., (step - 1000) / 49000)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        scaler = torch.cuda.amp.GradScaler(init_scale=t['amp_initial_scale'])
        generator = torch.Generator(device='cuda').manual_seed(t['seed'] + 1000)
        model.encoder.down_blocks[0].forward = torch.compile(model.encoder.down_blocks[0].forward,
                                                              backend='inductor', dynamic=False)
        model.train()
        updates = attempts = exposures = cursor = skips = 0
        if args.resume:
            saved = torch.load(checkpoint, map_location='cpu')
            if saved['identity'] != identity or saved['mode'] != args.mode:
                raise ValueError('Resume identity mismatch')
            if saved['initialization'] != initialization:
                raise ValueError('Resume scratch initialization differs')
            model.load_state_dict(saved['model'], strict=True)
            masters.load_state_dict(saved['masters'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            ema = {k: v.cuda() for k, v in saved['ema'].items()}
            ema_model.load_state_dict(ema, strict=True)
            restore_rng(saved['rng'], generator, torch.device('cuda'))
            updates, attempts, exposures, cursor, skips = (saved[k] for k in ('updates', 'attempts', 'exposures', 'cursor', 'skips'))
            for name, param in model.named_parameters():
                if not torch.equal(param, masters.params[name].to(param.dtype)):
                    raise ValueError(f'FP16/master mismatch: {name}')
        def save():
            atomic_checkpoint(checkpoint, {'identity': identity, 'initialization': initialization,
                'mode': args.mode, 'model': model.state_dict(),
                'masters': masters.state_dict(), 'ema': ema, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(generator, torch.device('cuda')), 'updates': updates, 'attempts': attempts,
                'exposures': exposures, 'cursor': cursor, 'skips': skips})
        torch.cuda.reset_peak_memory_stats()
        budget = 1 if args.mode == 'gate' else 32 if args.mode == 'smoke' else 50000
        limit = budget if args.max_updates is None else args.max_updates
        if limit < 1 or limit > budget or (args.mode == 'formal' and limit != budget):
            raise ValueError('Only the smoke can be deliberately shortened')
        gate_indices = np.argsort(dataset.index['length'])[-2:][::-1].tolist()
        initial_time = time.time()
        failures = 0
        atomic_json(progress, {'status': 'RUNNING', 'identity': identity, 'initialization': initialization,
                               'updates': updates, 'exposures': exposures})
        with (out / 'train.jsonl').open('a', buffering=1) as log:
            while updates < limit and not stopped[0]:
                tick = time.perf_counter()
                indices = gate_indices if args.mode == 'gate' else batch_indices(cursor, 2, 0, 1, len(dataset), t['seed'])
                records = [dataset[int(i)] for i in indices]
                scenes = [torch.from_numpy(input_xyz(Path(cfg['data_root']) / r['sequence'] / 'input_' / f"{r['frame']:06d}.npy")).cuda() for r in records]
                target = torch.cat([ids_to_dense_occupancy(r['parent_ids'], geometry, device='cuda') for r in records], 0)
                pair = occupancy_flow_pair(target, model.valid, generator=generator)
                model.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                    prediction = model(scenes, state=pair['state'], time=pair['time'])
                    loss = masked_occ_loss(prediction, pair['velocity'], model.valid, objective='fm')
                scaler.scale(loss).backward()
                masters.copy_gradients()
                scaler.unscale_(optimizer)
                norms = {}
                for prefix in ('encoder.', 'flow.'):
                    parts = [p.grad.float().square().sum() for n, p in masters.params.items() if n.startswith(prefix) and p.grad is not None]
                    norms[prefix] = float(torch.stack(parts).sum().sqrt()) if parts else 0.
                grad = torch.nn.utils.clip_grad_norm_(masters.params.values(), t['grad_clip'])
                finite = bool(torch.isfinite(loss) and torch.isfinite(grad) and float(grad) > 0 and all(np.isfinite(v) and v > 0 for v in norms.values()))
                attempts += 1
                cursor += 2
                if finite:
                    scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < scale:
                        raise RuntimeError('Unexpected hidden AMP skip')
                    masters.copy_to_network()
                    scheduler.step()
                    update_fp32_ema(ema, ema_model, model, masters, t['ema_decay'])
                    updates += 1
                    exposures += 2
                    failures = 0
                else:
                    skips += 1
                    failures += 1
                    scaler.update(new_scale=scaler.get_scale() / 2)
                    if failures >= 3:
                        raise FloatingPointError('Three consecutive nonfinite updates')
                torch.cuda.synchronize()
                row = {'updates': updates, 'exposures': exposures, 'attempts': attempts, 'skips': skips,
                       'finite': finite, 'loss': float(loss) if torch.isfinite(loss) else None,
                       'gradient_norm': float(grad) if torch.isfinite(grad) else None,
                       'module_gradients': {k: v if np.isfinite(v) else None for k, v in norms.items()},
                       'records': [{'sequence': r['sequence'], 'frame': r['frame']} for r in records],
                       'seconds': time.perf_counter() - tick, 'peak_allocated': torch.cuda.max_memory_allocated(),
                       'lr': [group['lr'] for group in optimizer.param_groups]}
                log.write(json.dumps(row, allow_nan=False) + '\n')
                print(json.dumps(row, allow_nan=False), flush=True)
                if args.mode == 'formal' and (updates % t['checkpoint_every'] == 0 or stopped[0]):
                    save()
                    atomic_json(progress, {'status': 'RUNNING', 'identity': identity, 'initialization': initialization, **row})
        if args.mode in ('formal', 'smoke'):
            save()
        status = 'PAUSED' if stopped[0] else 'PARTIAL' if limit < budget else 'COMPLETE' if args.mode == 'formal' else 'PASS'
        atomic_json(progress, {'status': status,
            'identity': identity, 'initialization': initialization, 'updates': updates, 'exposures': exposures, 'skips': skips,
            'peak_allocated': torch.cuda.max_memory_allocated(), 'precision': precision,
            'elapsed_seconds': time.time() - initial_time})
    except Exception as e:
        atomic_json(progress, {'status': 'BLOCKED', 'reason': repr(e), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

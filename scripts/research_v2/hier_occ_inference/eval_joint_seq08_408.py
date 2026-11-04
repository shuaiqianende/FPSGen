"""Evaluate completed T1/Stage2 models against original seq08 map points."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import numpy as np
import torch
import yaml
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.joint_fm_inference import input_only_conditions, predicted_support
from fpsgen.hier_occ_flow.raw_gt_exact_cache import ExactRawGTCache
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_no_lidar import remove_lidar_and_score
from fpsgen.hier_occ_flow.stage2_remove_score import remove_score
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha
from fpsgen.hier_occ_flow.t1_stage1_adapter import FrozenT1Stage1Adapter
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords, pack_children
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from unified_pointcloud_metrics import evaluate_prediction
from voxel_guided_resampling import voxel_guided_uniform_resample

W = Path('/data-12/M2024-HWZ/FPSGen/worktrees')
RUNS = {
    'with_encoder': ('desescene-stage2-b4-paired-v1', 'stage2_b4_paired_v1', 'with_encoder', 'stage2_b4_with_encoder.yaml'),
    'no_encoder': ('desescene-stage2-b4-paired-v1', 'stage2_b4_paired_v1', 'no_encoder', 'stage2_b4_no_encoder.yaml'),
    'position': ('desescene-stage2-child-b4-followup-v1', 'stage2_child_b4_followup_v1', 'position', 'stage2_child_b4_position.yaml'),
    'late_encoder': ('desescene-stage2-child-injection-v1', 'stage2_child_b4_followup_v1', 'attention', 'stage2_child_b4_attention.yaml'),
}


@torch.no_grad()
def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('This protocol runs on physical GPU2')
    if not 1 <= args.max_frames <= 408:
        raise ValueError('Expected 1..408 frames')
    tree, exp, arm, cfgname = RUNS[args.arm]
    base = W / tree
    cfgpath = base / 'configs/research_v2/hier_occ_flow' / cfgname
    cfg = yaml.safe_load(cfgpath.read_text())
    ckpath = base / 'experiments' / exp / arm / 'formal/checkpoints/resume_latest.pt'
    ck = torch.load(ckpath, map_location='cpu')
    if (ck['arm'], ck['mode'], ck['step'], ck['frames'], ck['batch_per_gpu']) != (arm, 'formal', 25000, 100000, 4):
        raise ValueError('Completed scratch 100K Stage2 checkpoint required')
    if ck['config_sha256'] != file_sha(cfgpath):
        raise ValueError('Stage2 configuration identity mismatch')
    g = SparseHierarchyGeometry(**cfg['geometry'])
    # The no-encoder Stage2 training configuration has stage1.type=none. At
    # deployment all variants still share the same real T1 Stage1 support.
    teacher_path = (W / 'desescene-stage2-b4-paired-v1/configs/research_v2/hier_occ_flow'
                    / 'stage2_b4_with_encoder.yaml')
    teacher = yaml.safe_load(teacher_path.read_text())
    if SparseHierarchyGeometry(**teacher['geometry']) != g:
        raise ValueError('Stage1 and Stage2 geometry do not match')
    torch.set_num_threads(4)
    adapter = FrozenT1Stage1Adapter(**teacher['stage1']['adapter']).cuda().eval()
    apply_explicit_policy(adapter.model)
    opts = dict(cfg['model'])
    opts['condition_mode'] = 'pyramid_observed'
    model = Stage2OccupancyModel(g, **opts).cuda()
    with torch.no_grad():
        model.score[2].weight.zero_()
        model.score[2].bias.zero_()
    dummy = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    if args.arm == 'no_encoder':
        remove_lidar_and_score(model, dummy)
    else:
        remove_score(model, dummy, model.state_dict())
    del dummy
    apply_explicit_policy(model)
    model.load_state_dict(ck['ema'], strict=True)
    model.eval().requires_grad_(False)
    training_source_sha = ck['source_sha256']
    del ck
    frames = list(range(0, 4071, 10))[:args.max_frames]
    cache = ExactRawGTCache(cfg['data_root'], g, ROOT / 'experiments/desescene_eval408_nextstage2_v1/shared_raw_gt_cache')
    protocol = {
        'arm': args.arm, 'frames': frames, 'stage2_checkpoint_sha256': file_sha(ckpath),
        'config_sha256': file_sha(cfgpath), 'stage1': adapter.provenance,
        'geometry_sha256': g.sha256, 'source_sha256': file_sha(Path(__file__)),
        'stage1_nfe': 2, 'stage2_nfe': 4, 'stage1_threshold': .5,
        'stage2_threshold': 0., 'stage1_seed': f"scene_seed(08,frame,{teacher['candidates']['seed']})",
        'stage2_seed': '20261031+frame', 'resample_seed': '20261019+frame',
        'points': 'exactly 1080000, 0.15625m PatchScene-style occupied-cell resampling',
        'gt': 'original RawMapSource points in ROI, no Poisson or subsampling',
        'gt_cache_identity_sha256': cache.sha256,
        'precision': 'explicit FP16 network, FP32 Euler/geometry',
        'score_injection': False, 'encoder_injection': args.arm != 'no_encoder',
        'stage2_source_sha256': training_source_sha,
        'runtime_sources': {str(p.relative_to(ROOT)): file_sha(p) for p in (
            ROOT / 'fpsgen/hier_occ_flow/stage2_model.py',
            ROOT / 'fpsgen/hier_occ_flow/explicit_fp16_training_policy.py',
            ROOT / 'fpsgen/hier_occ_flow/t1_stage1_adapter.py',
            Path(__file__).resolve().parent / 'voxel_guided_resampling.py',
            Path(__file__).resolve().parent / 'unified_pointcloud_metrics.py')},
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol_file, journal = out / 'protocol.json', out / 'per_frame.jsonl'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('Cannot mix different evaluation protocols')
    atomic_json(protocol_file, protocol)
    rows = [json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    done = {r['frame'] for r in rows}
    if len(done) != len(rows) or not done.issubset(frames):
        raise ValueError('Corrupt or unexpected frame journal')
    metric = GPUKDTreeMetrics(('cd',), geometry=None)
    for frame in frames:
        if frame in done:
            continue
        start = time.perf_counter()
        path = Path(cfg['data_root']) / '08/input_' / f'{frame:06d}.npy'
        xyz = torch.from_numpy(input_xyz(path)).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            pyramid = adapter.encode_features(xyz)
            seed = scene_seed('08', frame, teacher['candidates']['seed'])
            score, _ = adapter.model.sample(xyz, nfe=2, condition=pyramid,
                generator=torch.Generator(device='cuda').manual_seed(seed))
            ids, scores = predicted_support(score[0, 0].float(), adapter.model.valid, g)
            cond = input_only_conditions(ids, scores, xyz, g,
                pyramid=None if args.arm == 'no_encoder' else pyramid)
            cond.pop('parent_score')
            valid = cond['valid_child_mask']
            noise = torch.randn((g.num_parents, g.k), device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(20261031 + frame))
            x = noise[ids] * valid
            del noise
            for i in range(4):
                velocity = model(ids, state=x, time=x.new_tensor(i / 4), **cond)
                if not bool(torch.isfinite(velocity).all()):
                    raise ValueError('Nonfinite Stage2 velocity')
                x = x + velocity.float() / 4
            occ = (x > 0) & valid
        torch.cuda.synchronize()
        network_seconds = time.perf_counter() - start
        masks = pack_children(occ.cpu().numpy(), g)
        fine = occupied_fine_coords(ids.cpu().numpy(), masks, g)
        centers = g.centers(fine, fine=True).astype(np.float32)
        points, resampling = voxel_guided_uniform_resample(centers,
            bounds={'minimum': g.origin, 'maximum': g.maximum, 'sphere_radius': g.sphere_radius},
            voxel_size=.15625, target_points=1_080_000, seed=20261019 + frame)
        if points.shape != (1_080_000, 3):
            raise ValueError('Wrong output point count')
        target = cache.load(frame)
        row = {'frame': frame, 'candidate_parents': len(ids), 'occupied_fine': len(fine),
            'network_seconds': network_seconds, 'resampling_seconds': resampling['resampling_seconds'],
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            **evaluate_prediction(points, target, g, metric, device='cuda')}
        row['elapsed_seconds'] = time.perf_counter() - start
        if row['predicted_points'] != 1_080_000:
            raise ValueError('Metric output count mismatch')
        with journal.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        rows.append(row)
        done.add(frame)
        atomic_json(out / 'progress.json', {'status': 'RUNNING', 'completed': len(rows), 'total': len(frames), 'last_frame': frame})
        print(json.dumps(row, allow_nan=False), flush=True)
    if len(rows) != len(frames):
        raise ValueError('Incomplete evaluation')
    keys = ('cd_m', 'iou05', 'iou02', 'iou01', 'jsd_bev', 'jsd_3d', 'predicted_points', 'gt_points', 'network_seconds')
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    with (out / 'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out / 'results.json', {'status': 'COMPLETE', 'protocol': protocol, 'summary': summary, 'rows': rows})
    atomic_json(out / 'progress.json', {'status': 'COMPLETE', 'completed': len(rows), 'total': len(frames)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=RUNS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-frames', type=int, default=408)
    run(parser.parse_args())

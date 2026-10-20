"""Evaluate final Stage1 + Stage2 checkpoints on seq08/skip10 (2x2 and 4x4).

CD uses the project's exact GPU KD-tree backend. JSD reproduces FPSGen's
legacy per-frame 0.5 m histogram protocol with a sparse equivalent so it does
not allocate dense 200^3 histograms for every frame.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

from eval_seq08_combinations import physical_grid_ids, support_counts
from export_stage2_snapshot import EncodedInput, predicted_policy, validate_snapshot
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_condition import input_conditions, scene_seed
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords, pack_children
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from native_voxel_point_sampling import sample_native_occupied_voxels
from unified_pointcloud_metrics import evaluate_prediction


def admit_gpu(physical_gpu, *, startup=True):
    if physical_gpu not in (2, 3):
        raise ValueError('This evaluation is restricted to physical GPU2 or GPU3')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical_gpu):
        raise RuntimeError(f'Set CUDA_VISIBLE_DEVICES={physical_gpu} for this evaluation')
    out = subprocess.check_output([
        'nvidia-smi', f'--id={physical_gpu}', '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    free, util = map(float, out.strip().split(','))
    ps = subprocess.check_output([
        'nvidia-smi', f'--id={physical_gpu}', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    procs = [line.strip() for line in ps.splitlines()
             if line.strip() and int(line.split(',', 1)[0].strip()) != os.getpid()]
    # The 18 GiB headroom is a startup-only requirement.  During evaluation
    # this process itself retains model/allocator memory; subsequent pack
    # checks only reject external compute processes.
    if (startup and (free < 18000 or util > 10)) or procs:
        raise RuntimeError(f'GPU{physical_gpu} admission failed: startup={startup} free={free}MiB util={util}% processes={procs}')
    return {'physical_gpu': physical_gpu, 'free_mib_at_admission': free,
            'utilization_percent_at_admission': util, 'compute_processes': procs}


def _sparse_histogram(xyz, *, bev):
    """Sparse counts equivalent to histogramdd([-50,50]^3, bins=200)."""
    p = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(p[:, :2], axis=1)
    p = p[(r >= 0.0) & (r < 50.0)]
    if not len(p):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    ijk = np.floor((p + 50.0) / 0.5).astype(np.int64)
    valid = ((ijk >= 0) & (ijk < 200)).all(axis=1)
    ijk = ijk[valid]
    if not len(ijk):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    xyz_id = (ijk[:, 0] * 200 + ijk[:, 1]) * 200 + ijk[:, 2]
    ids, counts = np.unique(xyz_id, return_counts=True)
    if not bev:
        return ids, counts.astype(np.float64)
    # Legacy BEV first clips the 3D histogram to binary, then sums over Z.
    xy_id, z_counts = np.unique(ids // 200, return_counts=True)
    return xy_id, z_counts.astype(np.float64)


def _jsd_sparse(a, b):
    ka, ca = a; kb, cb = b
    if not len(ka) or not len(kb):
        return 0.0  # matches compute_hist_metrics_multirange's empty-range policy
    keys = np.union1d(ka, kb)
    pa = np.zeros(len(keys), dtype=np.float64)
    pb = np.zeros(len(keys), dtype=np.float64)
    pa[np.searchsorted(keys, ka)] = ca / ca.sum()
    pb[np.searchsorted(keys, kb)] = cb / cb.sum()
    return float(jensenshannon(pa, pb, base=None))


def jsd_metrics(gt, pred):
    return {
        'jsd_3d': _jsd_sparse(_sparse_histogram(gt, bev=False), _sparse_histogram(pred, bev=False)),
        'jsd_bev': _jsd_sparse(_sparse_histogram(gt, bev=True), _sparse_histogram(pred, bev=True)),
    }


def verify_jsd_parity(gt, pred):
    from fpsgen.utils.histogram_metrics import compute_hist_metrics_multirange
    for bev, key in ((False, 'jsd_3d'), (True, 'jsd_bev')):
        ref = compute_hist_metrics_multirange(gt, pred, bev=bev, dist_bins=[(0, 50.0)])['0_50.0m']
        got = jsd_metrics(gt, pred)[key]
        if not np.isclose(got, ref, rtol=1e-10, atol=1e-12):
            raise RuntimeError(f'Sparse {key} differs from FPSGen reference: {got} vs {ref}')
    return True


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, choices=(2, 3), default=3)
    p.add_argument('--max-frames', type=int, default=408)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stage1-nfe', type=int, default=2,
                   help='Stage1 Euler NFE (default: 2)')
    p.add_argument('--stage2-nfe', type=int, default=4,
                   help='Stage2 Euler NFE (default: 4)')
    p.add_argument('--point-mode', choices=('voxel_center', 'native_fine_jitter'),
                   default='voxel_center',
                   help='Native occupancy export mode; jitter remains inside model voxels')
    p.add_argument('--target-points', type=int, default=1_080_000)
    p.add_argument('--point-seed-base', type=int, default=20261019)
    p.add_argument('--save-point-samples', action='store_true',
                   help='Save per-frame center and selected-mode XYZ arrays (large output)')
    a = p.parse_args()
    if not 1 <= a.max_frames <= 408:
        raise ValueError('Expected 1..408 frames from the frozen seq08/skip10 protocol')
    if a.stage1_nfe < 1 or a.stage2_nfe < 1:
        raise ValueError('Stage1 and Stage2 NFE must be positive integers')

    admission = admit_gpu(a.physical_gpu)
    cfg, g = load_stage2_config(a.config)
    saved = torch.load(a.checkpoint, map_location='cpu')
    validate_snapshot(saved, cfg, g)
    if saved['successful_updates'] != 25000 or saved['sampler']['successful_frame_exposures'] != 100000:
        raise RuntimeError('Latest Stage2 checkpoint does not match the completed 100k-frame budget')

    data_root = Path(cfg['data_root'])
    input_files = sorted((data_root / '08' / 'input_').glob('*.npy'), key=lambda q: int(q.stem))
    if len(input_files) != 4071 or [int(q.stem) for q in input_files] != list(range(4071)):
        raise RuntimeError('Expected the audited 4071 seq08 input frames')
    frames = input_files[::10]
    if len(frames) != 408 or int(frames[-1].stem) != 4070:
        raise RuntimeError('seq08 skip10 manifest mismatch')
    frames = frames[:a.max_frames]

    stage2_sha = file_sha(a.checkpoint)
    stage1_sha = cfg['stage1']['adapter']['checkpoint_sha256']
    protocol = {
        'status': 'RUNNING', 'sequence': '08', 'stride': 10,
        'frames': [int(q.stem) for q in frames], 'frame_count': len(frames),
        'stage1_nfe_stage2_nfe_pairs': [[a.stage1_nfe, a.stage2_nfe]],
        'joint_inference': True, 'candidate_mode': 'predicted_stage1_candidates',
        'stage1_checkpoint_sha256': stage1_sha, 'stage1_weights': 'ema',
        'stage2_checkpoint_sha256': stage2_sha, 'stage2_updates': saved['successful_updates'],
        'stage2_frame_exposures': saved['sampler']['successful_frame_exposures'],
        'stage2_weights': 'online', 'geometry_sha256': g.sha256,
        'stage1_threshold': 0.5, 'stage2_threshold': 0.0,
        'source_seed': 'scene_seed(08, frame, candidate seed); identical across Stage1 NFE',
        'stage2_source': 'full parent-indexed Gaussian table, seed=validation.seed+frame; same values by parent ID across settings',
        'gt': 'original pre-Poisson raw-map local candidates, existing viewpoint support, frozen 50m sphere and z range',
        'cd': 'exact GPU KD-tree; half sum of directional mean Euclidean nearest-neighbor distances (m); all raw GT points',
        'iou_metrics': 'physical-grid support IoU at 0.5m, 0.2m, and 0.1m; origin[-50,-50,-4], clipped upper Z=4.4; sphere-intersecting ROI cells; missed GT counted FN',
        'jsd': 'FPSGen histogram_metrics.compute_hist_metrics_multirange semantics; radial XY [0,50), 0.5m bins on [-50,50]^3; scipy Jensen-Shannon distance (square-root JSD); per-frame macro mean',
        'jsd_3d': '3D point-count histogram',
        'jsd_bev': 'binary 3D voxel histogram summed over Z (occupied-height-bin count per XY cell)',
        'jsd_sparse_implementation_checked_against_existing_dense_helper': True,
        'physical_gpu': a.physical_gpu,
        'metric_device': f'physical GPU{a.physical_gpu} for model and exact CD; CPU sparse histogram JSD; CPU support IoU cross-check',
        'stage2_batch': 4, 'stage2_batching': 'ragged variable-length parent tokens, no padding or truncation; FP16 approximate batch equivalence',
        'metric_filter': 'NONE', 'source_downsampling': 'NONE',
        'config_sha256': file_sha(a.config), 'evaluator_sha256': file_sha(Path(__file__)),
        'stage2_training_target': 'morph-close26 8-way FM, oracle GT parents only; evaluation uses actual predicted parents',
        'point_mode': a.point_mode,
        'target_points': int(a.target_points) if a.point_mode == 'native_fine_jitter' else None,
        'point_seed_base': int(a.point_seed_base) if a.point_mode == 'native_fine_jitter' else None,
        'point_distribution': 'uniform_within_native_fine_voxel_intersection_with_roi' if a.point_mode == 'native_fine_jitter' else 'native_voxel_centers',
        'native_voxel_size_xyz': list(g.fine_size),
        'point_roi_policy': 'frozen_existing_roi:xyz_bounds_and_open_3d_sphere',
        'point_sampler_sha256': file_sha(Path(__file__).with_name('native_voxel_point_sampling.py')),
    }
    if a.resume:
        if not a.output.is_dir():
            raise FileNotFoundError(f'Resume output directory does not exist: {a.output}')
    else:
        if a.output.exists() and any(a.output.iterdir()):
            raise FileExistsError('Fresh evaluation output must be empty')
        a.output.mkdir(parents=True, exist_ok=True)
    protocol_path = a.output / 'protocol.json'
    journal = a.output / 'per_frame.jsonl'
    if a.resume:
        saved_protocol = json.loads(protocol_path.read_text())
        comparable_protocol = dict(saved_protocol)
        comparable_protocol['status'] = 'RUNNING'
        if comparable_protocol != protocol:
            # The first full run stopped after its first 4-scene pack because
            # the per-pack idle check counted this evaluator's own persistent
            # model allocations against the 18 GiB startup headroom.  The only
            # permitted resume migration is that admission-check fix; all data,
            # checkpoint, geometry, and metric protocol fields must still match.
            old_code = '2491db544f7fe0ef40bd2dfdbab6612a8d61325c29ce111052695864a53e9379'
            old_base = {k: v for k, v in saved_protocol.items() if k != 'evaluator_sha256'}
            new_base = {k: v for k, v in protocol.items() if k != 'evaluator_sha256'}
            if saved_protocol.get('evaluator_sha256') != old_code or old_base != new_base:
                raise RuntimeError('Resume protocol identity changed')
            protocol['previous_evaluator_sha256'] = old_code
        rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
    else:
        atomic_json(protocol_path, protocol)
        atomic_json(a.output / 'admission.json', admission)
        rows = []
    done = {(r['frame'], r['stage1_nfe']) for r in rows}
    if len(done) != len(rows):
        raise RuntimeError('Duplicate completed frame/config rows')

    torch.set_num_threads(4)
    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES maps the selected physical GPU here
    adapter = FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval()
    model = Stage2OccupancyModel(g, **cfg['model']).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    del saved
    metric = GPUKDTreeMetrics(('cd',), geometry=g)
    from skip10_metrics import initialize as initialize_gt_worker, load_target
    records = [{'sequence': '08', 'frame': int(q.stem), 'input': str(q.resolve()),
                'input_sha256': file_sha(q)} for q in frames]
    start = time.perf_counter()
    reference = ROOT.parent / 'hs-occflow-384-stage1occ' / 'experiments/hier_occ_flow/seq08_stride100_step024000_20261017'
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'),
            initializer=initialize_gt_worker,
            initargs=(str(data_root), g.to_dict(), str(reference))) as executor:
        pending = {}
        next_target = 0
        def prefetch_targets():
            nonlocal next_target
            while len(pending) < 8 and next_target < len(records):
                record = records[next_target]
                pending[record['frame']] = executor.submit(load_target, record)
                next_target += 1
        prefetch_targets()
        for offset in range(0, len(frames), 4):
            paths = frames[offset:offset+4]
            batch_records = records[offset:offset+4]
            batch_frames = [int(q.stem) for q in paths]
            if all((f, a.stage1_nfe) in done for f in batch_frames):
                for f in batch_frames:
                    pending.pop(f, None)
                prefetch_targets()
                continue
            admit_gpu(a.physical_gpu, startup=False)
            xyzs = [torch.from_numpy(input_xyz(path)).to(device) for path in paths]
            features = [adapter.encode_features(xyz) for xyz in xyzs]
            targets = [pending.pop(f).result() for f in batch_frames]
            prefetch_targets()
            for s1nfe, s2nfe in ((a.stage1_nfe, a.stage2_nfe),):
                active = []
                for i, (path, record, frame, xyz, pyramid, target_pair) in enumerate(
                        zip(paths, batch_records, batch_frames, xyzs, features, targets)):
                    if (frame, s1nfe) in done:
                        continue
                    seed = scene_seed('08', frame, cfg['candidates']['seed'])
                    with torch.autocast('cuda', dtype=torch.float16):
                        score, stage1_state = adapter.model.sample(
                            xyz, nfe=s1nfe, condition=pyramid,
                            generator=torch.Generator(device=device).manual_seed(seed))
                    if not bool(torch.isfinite(score).all() and torch.isfinite(stage1_state).all()):
                        raise RuntimeError(f'Nonfinite Stage1 output at frame {frame}, NFE{s1nfe}')
                    policy = predicted_policy(cfg); policy['stage1_nfe'] = s1nfe
                    encoded = EncodedInput(pyramid, score[0, 0].float(), adapter.provenance)
                    ids, cond, candidate_info = input_conditions(xyz, g, encoded, policy, seed=seed)
                    if len(ids) > cfg['training']['max_parent_tokens']:
                        raise RuntimeError(f'Predicted token count {len(ids)} exceeds registered maximum')
                    noise_seed = cfg['validation']['seed'] + frame
                    parent_noise = torch.randn((g.num_parents, g.k), device=device,
                        generator=torch.Generator(device=device).manual_seed(noise_seed))
                    source = parent_noise[ids] * cond['valid_child_mask']
                    active.append((ids, cond, source, candidate_info, frame, target_pair[0], target_pair[1], i))
                    del score, stage1_state, parent_noise

                if not active:
                    continue
                scene_pack = [(x[0], x[1], x[2], x[3]) for x in active]
                ids, cond, state, lengths = pack_scenes(scene_pack)
                if len(ids) > cfg['training']['max_batch_parent_tokens']:
                    raise RuntimeError(f'Packed token count {len(ids)} exceeds registered batch budget')
                for step in range(s2nfe):
                    times = state.new_full((len(active),), step / s2nfe)
                    with torch.autocast('cuda', dtype=torch.float16):
                        velocity = model(ids, state=state, time=times, **cond)
                    if not bool(torch.isfinite(velocity).all()):
                        raise RuntimeError(f'Nonfinite packed Stage2 velocity at NFE{s1nfe}')
                    state = state + velocity.float() / s2nfe
                endpoints = state.split(lengths)
                for scene, endpoint in zip(active, endpoints):
                    scene_ids, scene_cond, _, candidate_info, frame, target, target_meta, i = scene
                    occupancy = (endpoint > 0) & scene_cond['valid_child_mask']
                    fine = occupied_fine_coords(scene_ids.cpu().numpy(), pack_children(occupancy.cpu().numpy(), g), g)
                    predicted = g.centers(fine, fine=True).astype(np.float32)
                    if not len(predicted) or not len(target):
                        raise RuntimeError(f'Empty prediction or raw GT at frame {frame}')
                    center_metrics = evaluate_prediction(predicted, target, g, metric, device=device)
                    if offset == 0 and i == 0 and s1nfe == a.stage1_nfe:
                        verify_jsd_parity(target, predicted)
                    native_jitter_metadata = None
                    native_jitter_metrics = {}
                    if a.point_mode == 'native_fine_jitter':
                        point_seed = int(a.point_seed_base + frame)
                        predicted_native_jitter, native_jitter_metadata = sample_native_occupied_voxels(
                            fine, g, point_mode='native_fine_jitter',
                            target_points=a.target_points, seed=point_seed,
                        )
                        if a.save_point_samples:
                            samples_dir = a.output / 'point_samples'
                            samples_dir.mkdir(parents=True, exist_ok=True)
                            np.save(samples_dir / f'predicted_center_{frame:06d}.npy', predicted)
                            np.save(samples_dir / f'predicted_native_fine_jitter_{frame:06d}.npy', predicted_native_jitter)
                        torch.cuda.synchronize()
                        up_started = time.perf_counter()
                        up_metrics = evaluate_prediction(
                            predicted_native_jitter, target, g, metric, device=device)
                        torch.cuda.synchronize()
                        up_eval_seconds = time.perf_counter() - up_started
                        if offset == 0 and i == 0 and s1nfe == a.stage1_nfe:
                            verify_jsd_parity(target, predicted_native_jitter)
                        native_jitter_metrics = {
                            'cd_m_native_jitter': up_metrics['cd_m'],
                            'iou05_native_jitter': up_metrics['iou05'],
                            'iou02_native_jitter': up_metrics['iou02'],
                            'iou01_native_jitter': up_metrics['iou01'],
                            **{f'{key}_native_jitter_{part}': up_metrics[f'{key}_{part}']
                               for key in ('iou05', 'iou02', 'iou01')
                               for part in ('tp', 'fp', 'fn')},
                            'jsd_bev_native_jitter': up_metrics['jsd_bev'],
                            'jsd_3d_native_jitter': up_metrics['jsd_3d'],
                            'evaluation_seconds_native_jitter': float(up_eval_seconds),
                            'native_jitter_occupied_voxels': native_jitter_metadata['original_occupied_voxels'],
                            'native_jitter_sampled_voxels': native_jitter_metadata['sampled_occupied_voxels'],
                            'native_jitter_point_count': native_jitter_metadata['actual_output_points'],
                            'native_jitter_seconds': native_jitter_metadata['sampling_seconds'],
                            'native_jitter_seed': point_seed,
                            'native_jitter_roi_hit_and_run_fallback_samples': native_jitter_metadata['roi_hit_and_run_fallback_samples'],
                            'native_jitter_points_per_voxel_mean': native_jitter_metadata['mean_points_per_sampled_voxel'],
                            'native_jitter_points_per_voxel_min': native_jitter_metadata['min_points_per_sampled_voxel'],
                            'native_jitter_points_per_voxel_max': native_jitter_metadata['max_points_per_sampled_voxel'],
                            'native_jitter_dropped_voxels': native_jitter_metadata['dropped_occupied_voxels'],
                            'native_jitter_dropped_invalid_roi_voxels': native_jitter_metadata['dropped_invalid_roi_voxels'],
                            'native_jitter_dropped_unrepresentable_roi_voxels': native_jitter_metadata['dropped_unrepresentable_roi_voxels'],
                            'native_jitter_dropped_by_point_budget_voxels': native_jitter_metadata['dropped_by_point_budget_voxels'],
                            'native_jitter_coverage_ratio': native_jitter_metadata['voxel_coverage_ratio'],
                            'native_fine_voxel_size_xyz': native_jitter_metadata['native_voxel_size_xyz'],
                        }
                    row = {
                        'sequence': '08', 'frame': frame, 'stage1_nfe': s1nfe,
                        'stage2_nfe': s2nfe, **center_metrics,
                        **native_jitter_metrics, 'candidate_parents': int(len(scene_ids)),
                        'input_sha256': batch_records[i]['input_sha256'],
                        'gt_sha256': target_meta['target_sha256'],
                        'point_mode': a.point_mode,
                        'predicted_fine_voxels': int(len(predicted)), 'raw_gt_points': int(len(target)),
                        'raw_candidate_points_before_roi': target_meta['raw_candidate_points_before_roi'],
                        'filtered_points': target_meta['filtered_points'],
                        'stage2_batch': len(active), 'stage2_batch_fp16_approximate': True,
                        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                        'candidate_provenance': candidate_info,
                    }
                    with journal.open('a') as f:
                        f.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row); done.add((frame, s1nfe))
                    atomic_json(a.output / 'progress.json', {
                        'status': 'RUNNING', 'completed_rows': len(rows),
                        'requested_rows': len(frames), 'elapsed_seconds': time.perf_counter() - start,
                        'last_frame': frame, 'last_stage1_nfe': s1nfe,
                    })
                    print(f'{len(rows)}/{len(frames)} frame={frame:04d} S1/S2={s1nfe}/{s2nfe} '
                          f'CD={row["cd_m"]:.6f} IoU05/02/01='
                          f'{row["iou05"]:.5f}/{row["iou02"]:.5f}/{row["iou01"]:.5f} '
                          f'JSD-BEV={row["jsd_bev"]:.5f} JSD-3D={row["jsd_3d"]:.5f} '
                          f'tokens={len(scene_ids)} batch={len(active)}', flush=True)
                    if a.point_mode == 'native_fine_jitter':
                        del predicted_native_jitter
                    del occupancy, fine, predicted
                # Keep per-frame inputs/features/GT targets for the second NFE
                # configuration in this same 4-scene pack.  They are released
                # when the outer pack loop advances.
                del ids, cond, state, endpoints, scene_pack, active

    selected = [r for r in rows if r['stage1_nfe'] == a.stage1_nfe
                and r['stage2_nfe'] == a.stage2_nfe]
    if len(selected) != len(frames):
        raise RuntimeError(
            f'Incomplete S1/S2 {a.stage1_nfe}/{a.stage2_nfe}: '
            f'{len(selected)}/{len(frames)}')
    summary = {
        'frames': len(selected), 'stage1_nfe': a.stage1_nfe,
        'stage2_nfe': a.stage2_nfe,
        'cd_m_macro': float(np.mean([r['cd_m'] for r in selected])),
        'iou05_macro': float(np.mean([r['iou05'] for r in selected])),
        'iou05_micro': sum(r['iou05_tp'] for r in selected) / max(sum(r['iou05_tp'] + r['iou05_fp'] + r['iou05_fn'] for r in selected), 1),
        'iou02_macro': float(np.mean([r['iou02'] for r in selected])),
        'iou02_micro': sum(r['iou02_tp'] for r in selected) / max(sum(r['iou02_tp'] + r['iou02_fp'] + r['iou02_fn'] for r in selected), 1),
        'iou01_macro': float(np.mean([r['iou01'] for r in selected])),
        'iou01_micro': sum(r['iou01_tp'] for r in selected) / max(sum(r['iou01_tp'] + r['iou01_fp'] + r['iou01_fn'] for r in selected), 1),
        'jsd_bev_macro': float(np.mean([r['jsd_bev'] for r in selected])),
        'jsd_3d_macro': float(np.mean([r['jsd_3d'] for r in selected])),
        'mean_candidate_parents': float(np.mean([r['candidate_parents'] for r in selected])),
        'mean_predicted_fine_voxels': float(np.mean([r['predicted_fine_voxels'] for r in selected])),
        'mean_raw_gt_points': float(np.mean([r['raw_gt_points'] for r in selected])),
    }
    if a.point_mode == 'native_fine_jitter':
        summary.update({
            'cd_m_native_jitter_macro': float(np.mean([r['cd_m_native_jitter'] for r in selected])),
            'iou05_native_jitter_macro': float(np.mean([r['iou05_native_jitter'] for r in selected])),
            'iou05_native_jitter_micro': sum(r['iou05_native_jitter_tp'] for r in selected) / max(sum(r['iou05_native_jitter_tp'] + r['iou05_native_jitter_fp'] + r['iou05_native_jitter_fn'] for r in selected), 1),
            'iou02_native_jitter_macro': float(np.mean([r['iou02_native_jitter'] for r in selected])),
            'iou02_native_jitter_micro': sum(r['iou02_native_jitter_tp'] for r in selected) / max(sum(r['iou02_native_jitter_tp'] + r['iou02_native_jitter_fp'] + r['iou02_native_jitter_fn'] for r in selected), 1),
            'iou01_native_jitter_macro': float(np.mean([r['iou01_native_jitter'] for r in selected])),
            'iou01_native_jitter_micro': sum(r['iou01_native_jitter_tp'] for r in selected) / max(sum(r['iou01_native_jitter_tp'] + r['iou01_native_jitter_fp'] + r['iou01_native_jitter_fn'] for r in selected), 1),
            'jsd_bev_native_jitter_macro': float(np.mean([r['jsd_bev_native_jitter'] for r in selected])),
            'jsd_3d_native_jitter_macro': float(np.mean([r['jsd_3d_native_jitter'] for r in selected])),
            'mean_native_jitter_point_count': float(np.mean([r['native_jitter_point_count'] for r in selected])),
            'min_native_jitter_point_count': int(min(r['native_jitter_point_count'] for r in selected)),
            'max_native_jitter_point_count': int(max(r['native_jitter_point_count'] for r in selected)),
            'mean_native_jitter_occupied_voxels': float(np.mean([r['native_jitter_occupied_voxels'] for r in selected])),
            'mean_native_jitter_seconds': float(np.mean([r['native_jitter_seconds'] for r in selected])),
            'mean_native_jitter_evaluation_seconds': float(np.mean([r['evaluation_seconds_native_jitter'] for r in selected])),
            'mean_points_per_native_voxel': float(np.mean([r['native_jitter_points_per_voxel_mean'] for r in selected])),
            'mean_native_voxel_coverage_ratio': float(np.mean([r['native_jitter_coverage_ratio'] for r in selected])),
            'total_native_jitter_dropped_voxels': int(sum(r['native_jitter_dropped_voxels'] for r in selected)),
        })
    protocol['status'] = 'COMPLETE'
    atomic_json(protocol_path, protocol)
    summaries = [summary]
    result = {'status': 'COMPLETE', 'protocol': protocol, 'summary': summaries,
              'elapsed_seconds': time.perf_counter() - start, 'rows': len(rows)}
    atomic_json(a.output / 'results.json', result)
    atomic_json(a.output / 'progress.json', {k: v for k, v in result.items() if k != 'protocol'})
    with (a.output / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    with (a.output / 'per_frame.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'candidate_provenance'])
        writer.writeheader(); writer.writerows([{k:v for k,v in row.items() if k != 'candidate_provenance'} for row in rows])
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

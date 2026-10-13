"""Evaluate density/baselines on development; optionally create predicted count cache."""

import json
import time

import numpy as np
import torch

from common import parser, read_config, frames_for, configure_runtime, save_json, load_checkpoint, file_sha
from fpsgen.voxel_point.dataset import VoxelPointDataset
from fpsgen.voxel_point.voxdet_density import VoxDetDensity
from fpsgen.voxel_point.metrics import density_metrics, point_voxel_occupancy_metrics
from fpsgen.voxel_point.count_allocator import allocate_counts
from fpsgen.voxel_point.anchor_sampler import sample_gaussian_anchors


def main():
    p = parser(__doc__)
    p.add_argument('--split',choices=('train','development','test'),default='development')
    p.add_argument('--cache-counts',action='store_true')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--frame-stride',type=int,default=1,help='Select actual numeric frame IDs divisible by this stride')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--point-voxel-size',type=float,default=0,help='Optionally sample180k anchors and measure point occupancy IoU at this cubic size')
    p.add_argument('--save-density',action='store_true')
    args = p.parse_args()
    if args.frame_stride < 1 or args.cpu_threads < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.point_voxel_size < 0:
        raise ValueError('Invalid stride, threads or shard partition')
    cfg, spec, sha = read_config(args.config)
    frames = frames_for(cfg,args.split)
    frames = [frame for frame in frames if frame['frame'] % args.frame_stride == 0]
    if args.max_frames:
        frames = frames[:args.max_frames]
    selected_frames = frames
    frames = frames[args.shard_index::args.num_shards]
    if not frames:
        raise ValueError('Empty evaluation shard')
    device = torch.device('cpu') if args.device == 'cpu' else configure_runtime()[0]
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(cfg['seed'])
    model = VoxDetDensity(spec,**cfg['model']).to(device).eval()
    if device.type == 'cpu':
        from sample_voxdet_density import enable_cpu_tokenizer
        enable_cpu_tokenizer(model.lidar_tokenizer)
    checkpoint = load_checkpoint(model,cfg['stage1_checkpoint'],spec,stage='stage1')
    if checkpoint['mechanics_smoke_only'] and not args.smoke:
        raise ValueError('Smoke checkpoint cannot produce formal density results/caches')
    checkpoint_updates = checkpoint['successful_updates']
    del checkpoint
    dataset = VoxelPointDataset(frames,spec,gt_roi_policy=cfg['data'].get('gt_roi_policy','strict'))
    threshold = cfg['evaluation']['occupancy_threshold']
    rows = []
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha = file_sha(cfg['stage1_checkpoint'])
    metadata = {'split':args.split,'threshold':threshold,'threshold_frozen':cfg['evaluation']['threshold_frozen'],
        'config_sha256':sha,'checkpoint_sha256':checkpoint_sha,'checkpoint_updates':checkpoint_updates,
        'mechanics_smoke_only':args.smoke,'device':str(device),'precision':'fp32' if device.type=='cpu' else 'fp16_amp',
        'frame_stride':args.frame_stride,'num_shards':args.num_shards,'shard_index':args.shard_index,
        'selected_frame_count':len(selected_frames),'shard_frame_count':len(frames),
        'point_voxel_size_m':args.point_voxel_size,'point_voxel_iou_policy':'floor(XYZ/size), binary occupied if any point, no morphological closing',
        'anchor_points':180000 if args.point_voxel_size else None}
    valid = spec.valid_circle_mask(device=device)
    with torch.inference_mode():
        for i in range(len(dataset)):
            started = time.perf_counter()
            sample = dataset[i]
            xyz = sample['input_xyz'].to(device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda' and cfg['runtime']['amp']):
                density = model(xyz)['pred_density'][0,0]
            gt = sample['gt_counts'].to(device)
            partial_ids,_ = spec.voxelize_points(xyz,strict=False)
            partial = spec.flat_to_dense(torch.bincount(partial_ids,minlength=spec.num_voxels))
            target_occupied = (gt > 0) & valid
            confusion = {}
            for name,prediction in (('predicted',density),('partial_density',partial)):
                occupied = (prediction >= threshold) & valid
                confusion[name] = {'tp':int((occupied & target_occupied).sum()),
                    'fp':int((occupied & ~target_occupied).sum()),'fn':int((~occupied & target_occupied).sum())}
            rows.append({'frame':sample['frame'], 'gt_validation':sample['gt_validation'],
                'occupancy_confusion':confusion,
                'predicted':density_metrics(density,gt,spec,threshold=threshold),
                'gt_density':density_metrics(gt,gt,spec,threshold=threshold),
                'partial_density':density_metrics(partial,gt,spec,threshold=threshold)})
            if args.point_voxel_size:
                frame = sample['frame']
                seed = cfg['seed']+int(frame['sequence'])*1000000+frame['frame']
                # Only predicted density generates the source; GT is metric-only.
                generator = torch.Generator().manual_seed(seed)
                source_counts = allocate_counts(density.cpu(),spec,generator=generator)
                anchors,anchor_ids = sample_gaussian_anchors(spec,source_counts,generator=generator)
                assert anchors.shape == (180000,3) and spec.valid_points(anchors).all()
                target_xyz = sample['gt_xyz']
                target_fine = spec.dense_to_flat(gt.cpu()) > 0
                source_fine = source_counts > 0
                tp = int((source_fine & target_fine).sum())
                fp,fn = int((source_fine & ~target_fine).sum()),int((~source_fine & target_fine).sum())
                fine_metrics = {'voxel_size_xyz_m':list(spec.size),'tp':tp,'fp':fp,'fn':fn,
                                'iou':tp/max(tp+fp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)}
                rows[-1]['point_voxel'] = {
                    'predicted_anchors':point_voxel_occupancy_metrics(anchors,target_xyz,voxel_size=args.point_voxel_size),
                    'partial_input':point_voxel_occupancy_metrics(sample['input_xyz'][spec.valid_points(sample['input_xyz'])],target_xyz,voxel_size=args.point_voxel_size),
                    'fine_grid_anchors':fine_metrics,
                    'seed':seed,'source_points':180000,'gt_used_for_source':False,'pointflow_used':False}
            if args.save_density:
                frame = sample['frame']
                np.savez_compressed(args.output/f"{frame['sequence']}_{frame['frame']:06d}_density.npz",density=density.cpu().numpy())
                if args.point_voxel_size:
                    prefix = f"{frame['sequence']}_{frame['frame']:06d}"
                    np.save(args.output/f"{prefix}_source_anchors.npy",anchors.numpy())
                    np.savez_compressed(args.output/f"{prefix}_allocated_counts.npz",counts=source_counts.numpy())
            if args.cache_counts:
                if args.split != 'train':
                    raise ValueError('Only training frames enter predicted source cache')
                frame = sample['frame']
                # Numeric frame identity keeps seeds stable across sharding/stride.
                generator = torch.Generator(device=device).manual_seed(cfg['seed']+int(frame['sequence'])*1000000+frame['frame'])
                counts = allocate_counts(density,spec,generator=generator)
                cache_metadata = {'source':'predicted_density','stage1_checkpoint_sha256':checkpoint_sha,
                    'geometry':spec.to_dict(),'point_count':180000,'seed':cfg['seed']+int(frame['sequence'])*1000000+frame['frame'],
                    'config_sha256':sha,'frame':frame,'mechanics_smoke_only':args.smoke}
                np.savez_compressed(args.output/f"{frame['sequence']}_{frame['frame']:06d}.npz",
                    counts=counts.cpu().numpy(),metadata=json.dumps(cache_metadata))
            save_json(args.output/'density_evaluation.partial.json',{**metadata,'status':'RUNNING','rows':rows})
            point_status = ''
            if args.point_voxel_size:
                point_status = f" point_iou_{args.point_voxel_size}={rows[-1]['point_voxel']['predicted_anchors']['iou']:.4f}"
            print(f"shard={args.shard_index} {i+1}/{len(dataset)} frame={sample['frame']['sequence']}/{sample['frame']['frame']:06d} "
                  f"iou={rows[-1]['predicted']['occupancy_iou']:.4f} f1={rows[-1]['predicted']['occupancy_f1']:.4f} "
                  f"mass_tv={rows[-1]['predicted']['mass_tv']:.4f}{point_status} sec={time.perf_counter()-started:.1f}",flush=True)
    means = {name:{key:float(np.mean([row[name][key] for row in rows])) for key in rows[0][name]}
             for name in ('predicted','gt_density','partial_density')}
    save_json(args.output/'density_evaluation.json',{**metadata,'status':'COMPLETE','means':means,'rows':rows})


if __name__ == '__main__':
    main()

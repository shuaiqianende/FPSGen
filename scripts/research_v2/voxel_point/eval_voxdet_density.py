"""Evaluate density/baselines on development; optionally create predicted count cache."""

import json

import numpy as np
import torch

from common import parser, read_config, frames_for, configure_runtime, save_json, load_checkpoint, file_sha
from fpsgen.voxel_point.dataset import VoxelPointDataset
from fpsgen.voxel_point.voxdet_density import VoxDetDensity
from fpsgen.voxel_point.metrics import density_metrics
from fpsgen.voxel_point.count_allocator import allocate_counts


def main():
    p = parser(__doc__)
    p.add_argument('--split',choices=('train','development','test'),default='development')
    p.add_argument('--cache-counts',action='store_true')
    args = p.parse_args()
    cfg, spec, sha = read_config(args.config)
    device,_ = configure_runtime()
    torch.set_num_threads(4)
    model = VoxDetDensity(spec,**cfg['model']).to(device).eval()
    checkpoint = load_checkpoint(model,cfg['stage1_checkpoint'],spec,stage='stage1')
    if checkpoint['mechanics_smoke_only'] and not args.smoke:
        raise ValueError('Smoke checkpoint cannot produce formal density results/caches')
    frames = frames_for(cfg,args.split,args.max_frames)
    dataset = VoxelPointDataset(frames,spec)
    threshold = cfg['evaluation']['occupancy_threshold']
    rows = []
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha = file_sha(cfg['stage1_checkpoint'])
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            xyz = sample['input_xyz'].to(device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['runtime']['amp']):
                density = model(xyz)['pred_density'][0,0]
            gt = sample['gt_counts'].to(device)
            partial_ids,_ = spec.voxelize_points(xyz,strict=False)
            partial = spec.flat_to_dense(torch.bincount(partial_ids,minlength=spec.num_voxels))
            rows.append({'frame':sample['frame'], 'predicted':density_metrics(density,gt,spec,threshold=threshold),
                'gt_density':density_metrics(gt,gt,spec,threshold=threshold),
                'partial_density':density_metrics(partial,gt,spec,threshold=threshold)})
            if args.cache_counts:
                if args.split != 'train':
                    raise ValueError('Only training frames enter predicted source cache')
                generator = torch.Generator(device=device).manual_seed(cfg['seed']+i)
                counts = allocate_counts(density,spec,generator=generator)
                frame = sample['frame']
                metadata = {'source':'predicted_density','stage1_checkpoint_sha256':checkpoint_sha,
                    'geometry':spec.to_dict(),'point_count':180000,'seed':cfg['seed']+i,
                    'config_sha256':sha,'frame':frame,'mechanics_smoke_only':args.smoke}
                np.savez_compressed(args.output/f"{frame['sequence']}_{frame['frame']:06d}.npz",
                    counts=counts.cpu().numpy(),metadata=json.dumps(metadata))
    save_json(args.output/'density_evaluation.json',{'split':args.split,'threshold':threshold,
        'config_sha256':sha,'checkpoint_sha256':checkpoint_sha,'mechanics_smoke_only':args.smoke,'rows':rows})


if __name__ == '__main__':
    main()

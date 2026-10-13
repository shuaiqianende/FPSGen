"""Strict training-only sphere audit and train-derived n_ref; no test access."""

import json
import numpy as np
import torch
from common import parser,read_config,frames_for,select_spaced,save_json,sha_file
from fpsgen.voxel_shortcut.dataset import ShortcutDataset,validate_gt_points
from fpsgen.voxel_point.dataset import load_xyz,density_statistics,manifest_sha


def main():
    p=parser(__doc__);p.add_argument('--sparse',action='store_true',help='Compute identical count histograms from unique voxel IDs without materializing dense volumes')
    args=p.parse_args();cfg,spec,sha=read_config(args.config)
    torch.set_num_threads(4)
    frames=select_spaced(frames_for(cfg,'train'),args.max_frames)
    validation=cfg['data'].get('gt_validation',{})
    dataset=ShortcutDataset(frames,spec,**validation);rows=[];accepted=[];rejected=[];hist=None
    valid_voxels=int(spec.valid_circle_mask().sum())
    for i,frame in enumerate(frames):
        gt=load_xyz(frame['gt']);radius=gt.double().norm(dim=-1)
        diagnostics={'frame':frame,'max_radius_m':float(radius.max()),'sphere_outside_points':int((radius>=spec.sphere_radius).sum()),
                     'outside_all_roi_points':int((~spec.valid_points(gt)).sum()),'raw_points':len(gt)}
        try:
            if args.sparse:
                gt,gt_validation=validate_gt_points(gt,spec,**validation)
                ids,valid=spec.voxelize_points(gt)
                _,occupied=torch.unique(ids,return_counts=True)
                histogram=torch.bincount(occupied).tolist();histogram[0]=spec.num_voxels-len(occupied)
                stats={'points':int(occupied.sum()),'nonempty_voxel_count':len(occupied),'occupancy_ratio':len(occupied)/valid_voxels,
                    'maximum_count':int(occupied.max()),'occupied_quantiles':dict(zip(('p50','p90','p99'),torch.quantile(occupied.float(),torch.tensor([.5,.9,.99])).tolist())),
                    'voxel_count_histogram':histogram}
            else:
                sample=dataset[i]
                gt_validation=sample['gt_validation']
                counts=spec.to_xyz(sample['gt_counts'][0])
                stats=density_statistics(counts,spec)
            h=np.asarray(stats['voxel_count_histogram'],dtype=np.int64)
            if hist is None:hist=np.zeros(len(h),dtype=np.int64)
            if len(h)>len(hist):hist=np.pad(hist,(0,len(h)-len(hist)))
            hist[:len(h)]+=h
            rows.append({**diagnostics,**stats,'gt_validation':gt_validation});accepted.append(frame)
        except ValueError as exc:
            rejected.append({**diagnostics,'reason':str(exc)})
        if (i+1)%20==0:print(f'audit{i+1}/{len(frames)} accepted{len(accepted)} rejected{len(rejected)}',flush=True)
    if not accepted:raise ValueError('No valid training scenes')
    maximum=len(hist)-1
    save_json(args.output/'dataset_audit.json',{'config_sha256':sha,'geometry':spec.to_dict(),'audited_frames':len(frames),
        'training_manifest_sha256':manifest_sha(frames),'accepted':len(accepted),'rejected':rejected,'statistics':rows,
        'aggregate_histogram':hist.tolist(),'derived_n_ref':maximum,'mapping_exceedance_policy':'no target clipping; n>n_ref maps above1',
        'audit_scope':'sampled training frames;not full-dataset maximum','full_training_manifest_audited':len(frames)==len(frames_for(cfg,'train')),
        'gt_generator_source_sha256':sha_file('scripts/generate_semantickitti_poisson_gt.py')})
    save_json(args.output/'accepted_training_frames.json',accepted)
    print(json.dumps({'accepted':len(accepted),'rejected':len(rejected),'derived_n_ref':maximum}),flush=True)


if __name__=='__main__':main()

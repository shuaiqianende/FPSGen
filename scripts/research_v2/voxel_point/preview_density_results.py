"""Separate CPU-only GT evaluation and rendering of saved Stage1 anchors."""

import json
from pathlib import Path
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from common import parser, read_config, save_json
from fpsgen.voxel_point.dataset import VoxelPointDataset
from fpsgen.voxel_point.metrics import density_metrics
from fpsgen.voxel_point.count_allocator import allocate_counts
from fpsgen.voxel_point.anchor_sampler import sample_gaussian_anchors


def point_metrics_cpu(points, target):
    p, g = points.numpy(), target.numpy()
    target_tree = cKDTree(g)
    predicted_tree = cKDTree(p)
    dp = target_tree.query(p, k=1, workers=4)[0]
    dg = predicted_tree.query(g, k=1, workers=4)[0]
    spacing = predicted_tree.query(p, k=2, workers=4)[0][:, 1]
    result = {'cd':float(.5*(dp.mean()+dg.mean())),
              'spacing_mean':float(spacing.mean()), 'spacing_cv':float(spacing.std()/max(spacing.mean(),1e-12))}
    for threshold in (.1,.2):
        precision, coverage = float((dp<threshold).mean()), float((dg<threshold).mean())
        result[f'fscore_{threshold}'] = 2*precision*coverage/max(precision+coverage,1e-12)
        result[f'coverage_{threshold}'] = coverage
    return result


def render_comparison(sample, anchors, *, output, split, checkpoint_step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    figure, axes = plt.subplots(2,3,figsize=(16,10),gridspec_kw={'height_ratios':[1,.5]})
    clouds = [sample['input_xyz'].numpy(),anchors.numpy(),sample['gt_xyz'].numpy()]
    titles = ['Partial LiDAR XYZ','VoxDet density -> Gaussian anchors','GT reference (evaluation only)']
    for column,(cloud,title) in enumerate(zip(clouds,titles)):
        for row,(horizontal,vertical) in enumerate(((0,1),(0,2))):
            axis = axes[row,column]
            dots = axis.scatter(cloud[:,horizontal],cloud[:,vertical],c=cloud[:,2],cmap='turbo',
                                vmin=-4,vmax=4.4,s=.18 if column else .5,linewidths=0,rasterized=True)
            axis.set_xlim(-50,50)
            axis.set_ylim((-50,50) if row==0 else (-4,4.4))
            axis.set_xlabel('X (m)')
            axis.set_ylabel('Y (m)' if row==0 else 'Z (m)')
            axis.grid(alpha=.15)
            if row==0:
                axis.set_aspect('equal')
                axis.add_patch(Circle((0,0),50,fill=False,color='gray',linewidth=.5))
                axis.set_title(f'{title}\n{len(cloud):,} points')
    frame=sample['frame']
    figure.suptitle(f"Stage1 checkpoint {checkpoint_step:,} | {split} {frame['sequence']}/{frame['frame']:06d}\n"
                   'Density-only source preview; PointFlow is not applied',fontsize=15)
    figure.tight_layout(rect=(0,0,.96,.92))
    color_axis=figure.add_axes([.965,.28,.012,.4])
    figure.colorbar(dots,cax=color_axis,label='Height Z (m)')
    figure.savefig(output,dpi=150,facecolor='white')
    plt.close(figure)


def main():
    p=parser(__doc__)
    p.add_argument('--inference-root',required=True,type=Path)
    p.add_argument('--evaluation-manifest',required=True,type=Path)
    args=p.parse_args()
    cfg,spec,sha=read_config(args.config)
    torch.set_num_threads(4)
    frames=json.loads(args.evaluation_manifest.read_text())
    dataset=VoxelPointDataset(frames,spec,gt_roi_policy=cfg['data'].get('gt_roi_policy','strict'))
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str((args.output/'matplotlib_cache').resolve()))
    rows=[]
    checkpoint_step=int(Path(cfg['stage1_checkpoint']).stem.rsplit('_',1)[1])
    with torch.inference_mode():
        for index,frame in enumerate(frames):
            folder=args.inference_root/frame['id']
            sample=dataset[index]
            with np.load(folder/'voxel_density.npz',allow_pickle=False) as value:
                density=torch.from_numpy(value['density'].copy())
            anchors=torch.from_numpy(np.load(folder/'source_anchors.npy',allow_pickle=False))
            assert anchors.shape==(180000,3) and spec.valid_points(anchors).all()
            counts=sample['gt_counts']
            partial_ids,_=spec.voxelize_points(sample['input_xyz'],strict=False)
            partial_counts=spec.flat_to_dense(torch.bincount(partial_ids,minlength=spec.num_voxels))
            gt_counts=allocate_counts(counts.float(),spec,generator=torch.Generator().manual_seed(cfg['seed']+index))
            gt_anchors,_=sample_gaussian_anchors(spec,gt_counts,generator=torch.Generator().manual_seed(cfg['seed']+index+100))
            row={'id':frame['id'],'split':frame['split'],'gt_validation':sample['gt_validation'],
                 'predicted_density':density_metrics(density,counts,spec,threshold=1.0),
                 'partial_density':density_metrics(partial_counts,counts,spec,threshold=1.0),
                 'predicted_anchors':point_metrics_cpu(anchors,sample['gt_xyz']),
                 'oracle_gt_density_anchors':point_metrics_cpu(gt_anchors,sample['gt_xyz'])}
            image_path=args.output/(frame['id']+'_comparison.png')
            render_comparison(sample,anchors,output=image_path,split=frame['split'],checkpoint_step=checkpoint_step)
            row['image']=str(image_path.resolve())
            rows.append(row)
            save_json(args.output/'preview_metrics.json',{'checkpoint_step':checkpoint_step,'config_sha256':sha,
                'threshold':1.0,'threshold_frozen':False,'seq08_used':False,'pointflow_used':False,'rows':rows})
            print(json.dumps(row),flush=True)
    means={}
    for split in ('train','development'):
        subset=[r for r in rows if r['split']==split]
        means[split]={group:{key:float(np.mean([r[group][key] for r in subset])) for key in subset[0][group]}
                      for group in ('predicted_density','partial_density','predicted_anchors','oracle_gt_density_anchors')}
    save_json(args.output/'summary.json',{'checkpoint_step':checkpoint_step,'config_sha256':sha,
        'threshold':1.0,'threshold_frozen':False,'seq08_used':False,'pointflow_used':False,'means':means,'rows':rows})
    print(json.dumps(means,indent=2),flush=True)


if __name__=='__main__':
    main()

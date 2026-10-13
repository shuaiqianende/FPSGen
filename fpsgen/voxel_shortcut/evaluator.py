"""Density/observed-region/anchor/trajectory metrics with explicit count units."""

import torch
from fpsgen.voxel_point.metrics import density_metrics,point_metrics,point_voxel_occupancy_metrics
from fpsgen.ops.endpoint_refinement.knn import knn
from .count_mapping import normalize_density
from .shortcut_losses import sum_pool


def field_metrics(density,target,geometry,*,threshold=1.,observed=None):
    p,g=density.float(),target.float()
    valid=geometry.field_mask(device=p.device)
    result=density_metrics(geometry.to_xyz(p[0,0]),geometry.to_xyz(g[0,0]),geometry,threshold=threshold)
    pm=(p*valid).sum();gm=(g*valid).sum()
    result['empty_gt_voxel_mass_fraction']=float((p*valid*(g==0)).sum()/pm.clamp_min(1e-8))
    for factor in (1,2,4,8):
        a=sum_pool(p*valid,factor)/pm.clamp_min(1e-8)
        b=sum_pool(g*valid,factor)/gm.clamp_min(1e-8)
        result[f'mass_tv_s{factor}']=float(.5*(a-b).abs().sum())
    if observed is not None:
        for name,mask in [('direct_hit',observed&valid),('not_directly_hit',(~observed)&valid)]:
            po=(p>=threshold)&mask;go=(g>0)&mask
            tp=(po&go).sum();union=(po|go).sum()
            result[f'{name}_iou']=float(tp/union.clamp_min(1))
            result[f'{name}_f1']=float(2*tp/(po.sum()+go.sum()).clamp_min(1))
    return result


def endpoint_density(state,mapping,valid):
    raw,diagnostics=mapping.decode(state,return_diagnostics=True)
    raw=raw*valid
    return normalize_density(raw,valid),raw,diagnostics


@torch.no_grad()
def anchor_metrics(anchors,gt,geometry,*,backend='keops'):
    result=point_metrics(anchors,gt,backend=backend,spec=geometry)
    result['point_iou_0.5']=point_voxel_occupancy_metrics(anchors,gt,voxel_size=.5)['iou']
    distances,_=knn(anchors.float(),gt.float(),1,backend)
    q=torch.quantile(distances[:,0].clamp_min(0).sqrt(),torch.tensor([.5,.9,.95,.99],device=anchors.device))
    result.update({f'source_gt_distance_p{n}_m':float(v) for n,v in zip((50,90,95,99),q)})
    return result

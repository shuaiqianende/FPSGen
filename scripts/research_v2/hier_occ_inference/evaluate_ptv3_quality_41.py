#!/usr/bin/env python3
"""Paired exact raw-map quality comparison for saved native voxel predictions."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import file_sha
from eval_seq08_combinations import physical_grid_ids, support_counts
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from eval_seq08_skip10_joint_2x4_gpu3 import jsd_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--baseline-predictions',type=Path,required=True)
    p.add_argument('--variant-predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,default=list(range(0,4001,100)))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--physical-gpu',type=int,choices=(2,3),required=True)
    a=p.parse_args()
    if __import__('os').environ.get('CUDA_VISIBLE_DEVICES')!=str(a.physical_gpu) or not torch.cuda.is_available():
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the selected GPU2/3')
    from benchmark_hs_patchscene_inference import _admit
    admission=_admit(a.physical_gpu)
    device=torch.device('cuda:0')
    metric_evaluator=GPUKDTreeMetrics(('cd',),geometry=None)
    g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    raw=RawMapSource(a.data_root,fast_viewpoint=True)
    rows=[]
    for frame in a.frames:
        inp=a.data_root/'08'/'input_'/f'{frame:06d}.npy'
        input_hash=file_sha(inp)
        target=raw.load('08',frame)
        _,valid=g.voxelize_points(target); target=np.ascontiguousarray(target[valid],dtype=np.float32)
        result={'sequence':'08','frame':frame,'input_sha256':input_hash,'gt_points':len(target)}
        for label,path in [('baseline',a.baseline_predictions),('variant',a.variant_predictions)]:
            pred_path=path/f'frame_{frame:06d}.npz'
            if not pred_path.is_file():raise FileNotFoundError(pred_path)
            with np.load(pred_path,allow_pickle=False) as z:
                if z['input_xyz_sha256'].item()!=__import__('hashlib').sha256(np.ascontiguousarray(input_xyz(inp).astype(np.float32,copy=False)).view(np.uint8)).hexdigest():
                    raise ValueError(f'Input hash mismatch for {label} frame {frame}')
                fine=z['occupied_coords_xyz'].astype(np.int32,copy=False)
            predicted=g.centers(fine,fine=True).astype(np.float32)
            cd=metric_evaluator.evaluate(torch.from_numpy(predicted).to(device),torch.from_numpy(target).to(device))
            vals={f'{label}_{k}':v for k,v in cd.items()}
            for resolution,key in ((.5,'iou05'),(.2,'iou02'),(.1,'iou01')):
                counts=support_counts(physical_grid_ids(predicted,g,resolution=resolution),
                                      physical_grid_ids(target,g,resolution=resolution,target_points=True))
                vals[f'{label}_{key}']=counts['iou']
                vals[f'{label}_{key}_tp']=counts['tp'];vals[f'{label}_{key}_fp']=counts['fp'];vals[f'{label}_{key}_fn']=counts['fn']
            vals.update({f'{label}_{k}':v for k,v in jsd_metrics(target,predicted).items()})
            vals[f'{label}_occupied_voxels']=len(fine)
            result.update(vals)
        for metric_name in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d','occupied_voxels'):
            if metric_name!='occupied_voxels':result[f'delta_{metric_name}']=result[f'variant_{metric_name}']-result[f'baseline_{metric_name}']
            else:result['delta_occupied_voxels']=result['variant_occupied_voxels']-result['baseline_occupied_voxels']
        rows.append(result)
        print(json.dumps({k:result[k] for k in ('frame','baseline_cd_m','variant_cd_m','baseline_iou01','variant_iou01')},allow_nan=False),flush=True)
    metrics=('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d','occupied_voxels')
    summary={'status':'COMPLETE','physical_gpu':a.physical_gpu,'gpu_name':torch.cuda.get_device_name(device),
             'admission':admission,'frames':len(rows),'geometry_sha256':g.sha256,
             'gt':'RawMapSource(fast_viewpoint=True), frozen HS geometry ROI; no point downsampling',
             'cd':'FPSGen GPUKDTreeMetrics exact CUDA KD-tree symmetric mean Euclidean NN, meters',
             'metrics':{m:{'baseline_macro':float(np.mean([r[f'baseline_{m}'] for r in rows])),
                           'variant_macro':float(np.mean([r[f'variant_{m}'] for r in rows])),
                           'delta_macro':float(np.mean([r[f'delta_{m}'] for r in rows]))}
                        for m in metrics},'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'status':'COMPLETE','metrics':summary['metrics']},allow_nan=False),flush=True)

if __name__=='__main__':main()

"""Frozen Stage1 candidate recall audit; development-only threshold selection.

The optional full-training pass measures token quantiles with ONE already
registered policy. It cannot select a threshold using training or test results.
"""
from common import ROOT
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,config_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.candidate_selection import candidate_metrics
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-gpu',type=int,choices=(2,3),required=True)
    parser.add_argument('--split',choices=('development','train'),default='development')
    parser.add_argument('--max-frames',type=int);parser.add_argument('--thresholds',type=float,nargs='+')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.thresholds is not None and args.split!='development':raise ValueError('Select thresholds only on development')
    cfg,g=load_stage2_config(args.config)
    admission=gpu_preflight((args.physical_gpu,))
    if admission['status']!='AVAILABLE' or not args.execute:
        atomic_json(args.output,{'status':'NOT_EXECUTED','executed_gpu_operations':False,'admission':admission});return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise ValueError('Single admitted physical GPU required')
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda')
    train,dev,manifests=dataset_identity(cfg,g,verify=True);dataset=dev if args.split=='development' else train
    if args.max_frames is not None and not 0<args.max_frames<=len(dataset):raise ValueError('Invalid audit frame count')
    indices=np.linspace(0,len(dataset)-1,args.max_frames,dtype=int).tolist() if args.max_frames else list(range(len(dataset)))
    policies=[copy.deepcopy(cfg['candidates']['selection'])]
    if args.thresholds is not None:policies=[{**policies[0],'strategy':'threshold','threshold':t} for t in args.thresholds]
    # Auditing measures an uncapped set AND explicit budget effects. It never
    # silently discards a scene that would exceed the registered token budget.
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval();rows=[]
    report={'status':'RUNNING','executed_gpu_operations':True,'geometry_sha256':g.sha256,'stage1':adapter.provenance,
            'config_sha256':config_sha(cfg),'split':args.split,'full_split':len(indices)==len(dataset),
            'frames':len(indices),'manifests':manifests,'policies':policies,'rows':rows,
            'selection_status':'diagnostic only; this script never picks a best threshold automatically'}
    for index in indices:
        record=dataset[index];path=Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
        xyz=torch.from_numpy(input_xyz(path)).to(device)
        with torch.no_grad():
            features,scores=adapter(xyz,nfe=cfg['candidates']['stage1_nfe'],seed=scene_seed(record['sequence'],record['frame'],cfg['candidates']['seed']))
        score_array=scores.cpu().numpy()
        del features,scores
        for policy_index,policy in enumerate(policies):
            uncapped={**policy,'budget':None}
            ids,_,source=select_occupancy_candidates(score_array,g,**uncapped)
            rows.append({'index':index,'sequence':record['sequence'],'frame':record['frame'],'policy_index':policy_index,
                         'tokens':len(ids),'over_budget':policy.get('budget') is not None and len(ids)>policy['budget'],
                         **candidate_metrics(ids,record['parent_ids'],record['child_occ_mask'],g),'source':source})
        del score_array
        atomic_json(args.output,report)
        print(f'Candidate audit {args.split} {index}/{len(dataset)}',flush=True)
    summaries=[]
    for i,policy in enumerate(policies):
        values=[r for r in rows if r['policy_index']==i];counts=np.array([r['tokens'] for r in values])
        summaries.append({'policy':policy,'token_quantiles':dict(zip(('p50','p90','p99','max'),map(float,np.quantile(counts,[.5,.9,.99,1.])))),
                          'over_budget_scenes':sum(r['over_budget'] for r in values),
                          'maximum_record':max(values,key=lambda r:r['tokens'])})
    report.update(status='PASS_CANDIDATE_AUDIT',summaries=summaries);atomic_json(args.output,report)


if __name__=='__main__':main()

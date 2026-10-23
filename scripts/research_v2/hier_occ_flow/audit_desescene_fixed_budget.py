"""D3: stream real frozen Stage1 scores, freeze ranking on grouped calibration only."""
from common import ROOT
import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config, dataset_identity, source_sha, config_sha
from fpsgen.hier_occ_flow.stage2_candidates import parent_valid_field
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.support_ranking_diagnostics import (
    stable_topk, six_neighbors, local_max6, is_calibration, ceiling, oracle_pool_upper_bound)
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=900)
    a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in ('2','3'):raise ValueError('One physical GPU2/3 required')
    admission=gpu_preflight((int(physical),))
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    cfg,g=load_stage2_config(a.config)
    _,ds,manifests=dataset_identity(cfg,g,verify=True)
    torch.set_num_threads(4)
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda().eval()
    valid=parent_valid_field(g).reshape(-1)
    choices=np.flatnonzero(valid)
    indices=np.linspace(0,len(ds)-1,min(a.max_frames,len(ds)),dtype=int).tolist()
    split={i:is_calibration(ds[i]['sequence'],ds[i]['frame']) for i in indices}
    calibration=[i for i in indices if split[i]]
    heldout=[i for i in indices if not split[i]]
    if not calibration or not heldout:raise ValueError('Both grouped calibration and heldout required')
    grid=list(itertools.product((0.,.25,.5,1.),(0.,.25,.5)))
    ks=(84963,120000)
    rows=[]
    frozen={}
    contract={'config_sha256':config_sha(cfg),'source_sha256':source_sha(),
              'stage1':adapter.provenance,'stage1_nfe':cfg['candidates']['stage1_nfe'],
              'geometry_sha256':g.sha256,**manifests,'gpu_admission':admission,
              'budgets':ks,'lambda_grid':grid,'calibration_indices':calibration,
              'heldout_indices':heldout,'pool':'TopK(ceil1.5K) union six-neighbors(TopK K) union observed',
              'split':'sequence + floor(frame/100), SHA256 modulo5==0 calibration; remaining heldout',
              'no_stage2_or_gt_used_to_build_pool':True}
    atomic_json(a.output/'contract.json',contract)
    with (a.output/'per_frame.jsonl').open('w',buffering=1) as log:
        for phase, selected in (('calibration',calibration),('heldout',heldout)):
            if phase=='heldout':
                for k in ks:
                    averages=[np.mean([r['budgets'][str(k)]['grid_ceiling'][j] for r in rows]) for j in range(len(grid))]
                    winner=int(np.argmax(averages))
                    frozen[str(k)]={'lambda1':grid[winner][0],'lambda2':grid[winner][1],
                                    'grid_index':winner,'calibration_mean_ceiling':float(averages[winner])}
                atomic_json(a.output/'frozen_ranking.json',frozen)
            for index in selected:
                record=ds[index]
                path=Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
                points=input_xyz(path)
                seed=scene_seed(record['sequence'],record['frame'],cfg['candidates']['seed'])
                pyramid,score=adapter(torch.from_numpy(points).cuda(),nfe=cfg['candidates']['stage1_nfe'],seed=seed,with_scores=True)
                score=score.float().cpu().numpy()
                del pyramid
                local=local_max6(score).reshape(-1);score=score.reshape(-1)
                fine,_=g.voxelize_points(points)
                observed=np.unique(g.flatten(g.split_fine(fine)[0]))
                observed=observed[valid[observed]]
                hit=np.zeros(g.num_parents,np.float32);hit[observed]=1
                row={'index':index,'sequence':record['sequence'],'frame':record['frame'],
                     'input_sha256':file_sha(path),'stage1_seed':seed,'split':phase,'budgets':{},
                     'stage1_threshold_parent_count':int(((score>.5)&valid).sum())}
                for k in ks:
                    top=stable_topk(score,choices,k)
                    pool=np.union1d(np.union1d(stable_topk(score,choices,int(np.ceil(1.5*k))),six_neighbors(top,g,valid)),observed)
                    values=[]
                    policies=grid if phase=='calibration' else [(frozen[str(k)]['lambda1'],frozen[str(k)]['lambda2'])]
                    for lam1,lam2 in policies:
                        rank=score+lam1*local+lam2*hit
                        chosen=stable_topk(rank,pool,k)
                        values.append(ceiling(chosen,record['parent_ids'],record['child_occ_mask']))
                    winner=0 if phase=='heldout' else None
                    row['budgets'][str(k)]={'pool_size':len(pool),'selected_count':k,
                        'score_topk_ceiling':ceiling(top,record['parent_ids'],record['child_occ_mask']),
                        'pool_ceiling':ceiling(pool,record['parent_ids'],record['child_occ_mask']),
                        'gt_only_oracle_pool_ceiling':oracle_pool_upper_bound(pool,record['parent_ids'],record['child_occ_mask'],k),
                        **({'grid_ceiling':values} if winner is None else {'frozen_rank_ceiling':values[0]})}
                rows.append(row);log.write(json.dumps(row)+'\n')
                if len(rows)%10==0:print('ranking scenes',len(rows),'/',len(indices),phase,flush=True)
    summaries={}
    for k in ks:
        chosen=frozen[str(k)]['grid_index']
        groups={}
        for phase in ('calibration','heldout'):
            part=[r['budgets'][str(k)] for r in rows if r['split']==phase]
            delta=[(r['grid_ceiling'][chosen] if phase=='calibration' else r['frozen_rank_ceiling'])-r['score_topk_ceiling'] for r in part]
            groups[phase]={'frames':len(part),'mean_ceiling_delta':float(np.mean(delta)),
                           'mean_baseline_ceiling':float(np.mean([r['score_topk_ceiling'] for r in part])),
                           'worst_delta':float(min(delta))}
        summaries[str(k)]={**groups,'ranker_training_eligible':groups['heldout']['mean_ceiling_delta']>=.02}
    atomic_json(a.output/'summary.json',{'status':'COMPLETE','frozen_ranking':frozen,'summaries':summaries,
                'formal_ranker_started':False,'selection_used_heldout_gt':False,'frames':len(rows)})
    with (a.output/'per_frame.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sequence','frame','split','k','score_ceiling','frozen_rank_ceiling','oracle_upper_bound'])
        for row in rows:
            for k in ks:
                value=row['budgets'][str(k)]
                selected=value['frozen_rank_ceiling'] if row['split']=='heldout' else value['grid_ceiling'][frozen[str(k)]['grid_index']]
                writer.writerow([row['sequence'],row['frame'],row['split'],k,value['score_topk_ceiling'],selected,value['gt_only_oracle_pool_ceiling']])


if __name__=='__main__':
    main()

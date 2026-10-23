"""D1/D2 on fixed real development scenes, paired sources and immutable support."""
from common import ROOT
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.desescene_diagnostics import parent_noise,rollout,error_stats,vector_stats,block_outputs
from fpsgen.hier_occ_flow.evaluation import fine_metrics,rates
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,config_sha,source_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords,pack_children


def native(ids,occupied,record,g):
    result=fine_metrics(ids.cpu().numpy(),occupied.cpu().numpy(),record['parent_ids'],
                        record['child_occ_mask'],g,structural=False)
    inside={k:result['fine'][k] for k in ('tp','fp','fn')}
    inside['fn']-=result['missed_parent_fine_false_negatives']
    return {**{k:result[k] for k in ('fine','candidate','distance_bins','missed_parent_fine_false_negatives')},
            'support_only_fine':rates(inside)}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ema',action='store_true')
    p.add_argument('--max-frames',type=int,default=64);p.add_argument('--seeds',type=int,nargs='+',default=[20261035,20261036,20261037])
    p.add_argument('--stage1-nfe',type=int,choices=(1,2,4,8))
    p.add_argument('--mode',choices=('d1','d2','both'),default='both')
    p.add_argument('--skip-continuous',action='store_true',help='Mark CD/physical IoU/JSD NOT TESTED explicitly')
    a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in ('2','3'):raise ValueError('One physical GPU2/3 required')
    admission=gpu_preflight((int(physical),))
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    cfg,g=load_stage2_config(a.config);_,ds,manifests=dataset_identity(cfg,g,verify=True)
    saved=torch.load(a.checkpoint,map_location='cpu')
    identity=saved['identity']
    if (identity['config_sha256']!=config_sha(cfg) or identity['geometry_sha256']!=g.sha256
        or identity['stage1_checkpoint_sha256']!=cfg['stage1']['checkpoint_sha256']
        or any(identity[k]!=v for k,v in manifests.items())):raise ValueError('Checkpoint identity mismatch')
    weights='ema' if a.ema else 'model'
    model=Stage2OccupancyModel(g,**cfg['model']).cuda().eval()
    model.load_state_dict(saved[weights],strict=True)
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda().eval()
    diagnostic_cfg=copy.deepcopy(cfg)
    diagnostic_cfg['candidates']['mode']='predicted'
    if a.stage1_nfe is not None:diagnostic_cfg['candidates']['stage1_nfe']=a.stage1_nfe
    indices=np.linspace(0,len(ds)-1,min(a.max_frames,len(ds)),dtype=int).tolist()
    raw=None;metric=None
    if not a.skip_continuous:
        sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
        from unified_pointcloud_metrics import evaluate_prediction
        from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
        metric=GPUKDTreeMetrics(('cd',),geometry=g)
        raw=RawMapSource(cfg['data_root'],fast_viewpoint=True)
    contract={'checkpoint_sha256':file_sha(a.checkpoint),'weights':weights,'training_identity':identity,
              'training_candidates':cfg['candidates'],'diagnostic_candidates':diagnostic_cfg['candidates'],
              'source_sha256':source_sha(),**manifests,'gpu_admission':admission,'seeds':a.seeds,
              'indices':indices,'scope':'mechanism diagnostic, not 900-frame generalization or dedicated runtime',
              'source_rng':'SHA256 Scene/seed + SplitMix64 Parent/Child counters + Box-Muller FP32 normal',
              'source_distribution':'Gaussian; keyed RNG differs from native torch stream but is identical across models/support overlaps',
              'continuous_metrics':'NOT TESTED' if a.skip_continuous else 'raw-map GT, exact GPU KD-tree, corrected physical IoU, shared FPSGen JSD',
              'nfe_schedule':[1,2,4,8] if model.objective=='fm' else [1]}
    atomic_json(a.output/'contract.json',contract)
    rows=[];sensitivities=[]
    torch.cuda.reset_peak_memory_stats()
    with (a.output/'per_case.jsonl').open('w',buffering=1) as log, (a.output/'sensitivity.jsonl').open('w',buffering=1) as slog:
        for index in indices:
            record=ds[index];ids,condition,target,provenance=assemble_stage2_scene(record,diagnostic_cfg,g,torch.device('cuda'),adapter)
            mask=condition['valid_child_mask'];scene=record['sequence']+'/'+str(record['frame'])
            path=Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
            base={'index':index,'sequence':record['sequence'],'frame':record['frame'],'input_sha256':file_sha(path),
                  'parent_ids_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                  'parent_count':len(ids),'candidate_source':provenance}
            raw_target=None
            def evaluate(occupied):
                nonlocal raw_target
                result=native(ids,occupied,record,g)
                if raw is not None:
                    if raw_target is None:
                        cloud=raw.load(record['sequence'],record['frame']);_,valid=g.voxelize_points(cloud)
                        raw_target=np.ascontiguousarray(cloud[valid],dtype=np.float32)
                    fine=occupied_fine_coords(ids.cpu().numpy(),pack_children(occupied.cpu().numpy(),g),g)
                    points=g.centers(fine,fine=True).astype(np.float32)
                    result['continuous']=(evaluate_prediction(points,raw_target,g,metric,device='cuda')
                        if len(points) else {'status':'EMPTY_PREDICTION','cd_m':None,'iou05':0.,'iou02':0.,'iou01':0.})
                    result['raw_gt_xyz_sha256']=hashlib.sha256(raw_target.tobytes()).hexdigest()
                return result
            trajectory_states=None
            if a.mode in ('d1','both'):
                for nfe in contract['nfe_schedule']:
                    endpoint_sum=None;endpoint_square_sum=None;occupied_sum=torch.zeros_like(mask,dtype=torch.float32)
                    for seed in a.seeds:
                        source=torch.from_numpy(parent_noise(ids.cpu().numpy(),g.k,scene,seed)).cuda()*mask
                        tick=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['precision']=='fp16'):
                            endpoint,steps,states=rollout(model,ids,condition,source,nfe,keep_states=nfe==4 and seed==a.seeds[0])
                        if states:trajectory_states=states
                        occupied=(endpoint>0)&mask
                        prediction_seconds=time.perf_counter()-tick
                        endpoint_sum=endpoint.float() if endpoint_sum is None else endpoint_sum+endpoint.float()
                        endpoint_square_sum=endpoint.float().square() if endpoint_square_sum is None else endpoint_square_sum+endpoint.float().square()
                        occupied_sum+=occupied.float()
                        row={**base,'nfe':nfe,'seed':seed,'kind':'single_sample','steps':steps,
                             'diagnostic_prediction_wall_unsynchronized':prediction_seconds,**evaluate(occupied)}
                        rows.append(row);log.write(json.dumps(row)+'\n')
                    mean=endpoint_sum/len(a.seeds);probability=occupied_sum/len(a.seeds)
                    for kind,occupied in (('mean_endpoint', (mean>0)&mask),('mean_occupancy_probability', (probability>.5)&mask)):
                        row={**base,'nfe':nfe,'seed':None,'kind':kind,
                             'mean_endpoint_variance':float((endpoint_square_sum/len(a.seeds)-mean.square()).clamp_min(0)[mask].mean()),
                             'occupancy_brier':float((probability[mask]-target[mask].float()).square().mean()),
                             **evaluate(occupied)}
                        rows.append(row);log.write(json.dumps(row)+'\n')
                if model.objective=='fm':
                    source=torch.from_numpy(parent_noise(ids.cpu().numpy(),g.k,scene,a.seeds[0])).cuda()*mask
                    binary=(2*target.float()-1)*mask
                    teacher=[]
                    for t in (0.,.25,.5,.75,1.):
                        state=(1-t)*source+t*binary
                        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['precision']=='fp16'):
                            velocity=model(ids,state=state,time=state.new_tensor(t),**condition)
                        teacher.append({'time':t,'mse':float((velocity.float()-(binary-source)).square()[mask].mean()),
                                        'velocity':vector_stats(velocity)})
                    log.write(json.dumps({**base,'kind':'teacher_forced_training_path','velocity_error':teacher})+'\n')
            if a.mode in ('d2','both'):
                source=torch.from_numpy(parent_noise(ids.cpu().numpy(),g.k,scene,a.seeds[0])).cuda()*mask
                if model.objective=='fm':
                    if trajectory_states is None:
                        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['precision']=='fp16'):
                            _,_,trajectory_states=rollout(model,ids,condition,source,4,keep_states=True)
                    state=trajectory_states[2];t=state.new_tensor(.5)
                else:state=t=None
                def forward(cond,x):
                    with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['precision']=='fp16'):
                        return model(ids,state=x,time=t,**cond)
                with block_outputs(model) as baseline_blocks:reference=forward(condition,state).detach()
                generator=torch.Generator(device='cuda').manual_seed(a.seeds[0]+index)
                cases=[]
                for key in ('lidar_features','parent_score','observed_child'):
                    changed=dict(condition);changed[key]=torch.zeros_like(condition[key]);cases.append((key+'_zero',changed,state,None))
                shuffled=dict(condition);perm=torch.randperm(len(ids),device='cuda',generator=generator)
                shuffled['lidar_features']=condition['lidar_features'][perm]
                cases.append(('lidar_parent_shuffle',shuffled,state,None))
                if state is not None:
                    changed=state.clone();subset=perm[:max(1,len(ids)//100)]
                    amplitude=state.float().square().mean().sqrt().clamp_min(1e-6)*.1
                    changed[subset]+=amplitude*mask[subset]
                    untouched=torch.ones(len(ids),device='cuda',dtype=torch.bool);untouched[subset]=False
                    cases.append(('state_1percent_relative_rms_0.1',condition,changed,untouched))
                for name,cond,x,untouched in cases:
                    with block_outputs(model) as changed_blocks:prediction=forward(cond,x)
                    row={**base,'perturbation':name,'velocity_change':error_stats(reference,prediction),
                         'blocks':{k:error_stats(value,changed_blocks[k]) for k,value in baseline_blocks.items()}}
                    if untouched is not None:row['unperturbed_parent_velocity_change']=error_stats(reference[untouched],prediction[untouched])
                    sensitivities.append(row);slog.write(json.dumps(row)+'\n')
                del baseline_blocks,changed_blocks
            print('mechanism scene',index,scene,'parents',len(ids),flush=True)
    summary={}
    for nfe in contract['nfe_schedule']:
        summary[str(nfe)]={}
        for kind in ('single_sample','mean_endpoint','mean_occupancy_probability'):
            part=[r for r in rows if r['nfe']==nfe and r['kind']==kind]
            if part:summary[str(nfe)][kind]={'rows':len(part),'macro_iou':float(np.mean([r['fine']['iou'] for r in part])),
                                             'macro_f1':float(np.mean([r['fine']['f1'] for r in part]))}
            if part and not a.skip_continuous:
                summary[str(nfe)][kind]['continuous_macro']={key:float(np.mean([r['continuous'][key] for r in part]))
                    for key in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d') if all(r['continuous'].get(key) is not None for r in part)}
    with (a.output/'per_frame.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sequence','frame','nfe','seed','kind','fine_iou','fine_f1','cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d'])
        for row in rows:
            writer.writerow([row[k] for k in ('sequence','frame','nfe','seed','kind')]+[row['fine']['iou'],row['fine']['f1']]+[row.get('continuous',{}).get(k,'NOT TESTED') for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')])
    with (a.output/'sensitivity.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sequence','frame','perturbation','velocity_relative_l2','outside_perturbed_parent_relative_l2'])
        for row in sensitivities:
            writer.writerow([row[k] for k in ('sequence','frame','perturbation')]+[row['velocity_change']['relative_l2'],row.get('unperturbed_parent_velocity_change',{}).get('relative_l2')])
    atomic_json(a.output/'summary.json',{'status':'COMPLETE','frames':len(indices),'nfe_results':summary,
                'sensitivity_rows':len(sensitivities),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'note':'NFE quality changes alone cannot identify a bug or prove a sampling/mean-estimation explanation'})


if __name__=='__main__':
    main()

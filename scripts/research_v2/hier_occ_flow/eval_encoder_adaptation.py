"""900-scene native/physical evaluation: fixed Teacher support AND deployable support."""
from common import ROOT
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from fpsgen.hier_occ_flow.encoder_adaptation import load_sources,load_stage2_source
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config,datasets,config_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage1_evaluation import parent_metrics,summarize
from fpsgen.hier_occ_flow.stage1_curve_metrics import fixed_budget_metrics,summarize_fixed_budget
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.condition import sample_pyramid,observed_children
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.inference import sample
from fpsgen.hier_occ_flow.evaluation import fine_metrics,aggregate_frames
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,occupied_fine_coords,pack_children
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
from unified_pointcloud_metrics import evaluate_prediction


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--reference',action='store_true',help='EF0 / original sources, no extra training')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if any(p.exists() for p in (a.output,a.output.with_suffix('.jsonl'),a.output.with_suffix('.protocol.json'))):
        raise FileExistsError('Preserve previous complete or partial evaluation; review before restarting')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in ('2','3'):raise ValueError('One GPU2/3')
    admission=gpu_preflight((int(gpu),))
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    cfg,g=load_adaptation_config(a.config);_,dev,manifest=datasets(cfg,g,verify=True)
    model,_,initial=load_sources(cfg,torch.device('cuda'));torch.set_num_threads(4)
    checkpoint_sha=None
    if not a.reference:
        if a.checkpoint is None:raise ValueError('Checkpoint or explicit frozen reference required')
        saved=torch.load(a.checkpoint,map_location='cpu')
        if (saved['initialization_mode']!='pretrained_ema_adaptation' or saved['identity']['config_sha256']!=config_sha(cfg)
            or any(saved['identity'][k]!=v for k,v in manifest.items())
            or saved['successful_frame_exposures']!=20000):raise ValueError('Require final matching20K checkpoint')
        model.load_state_dict(saved['ema'],strict=True);checkpoint_sha=file_sha(a.checkpoint)
    model.eval()
    fine_model=model.stage2
    if fine_model is None:fine_model,_=load_stage2_source(cfg,g);fine_model=fine_model.cuda().eval()
    teacher=FrozenStage1OccupancyAdapter(**cfg['source_stage1']).cuda().eval()
    policy=json.loads(Path(cfg['reference_stage2_config']).read_text());valid=model.stage1.valid.cpu().numpy()
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);metric=GPUKDTreeMetrics(('cd',),geometry=g)
    nfes=cfg['validation']['nfes'];seed=cfg['validation']['seed']
    stage1_rows={str(n):[] for n in nfes};fine_rows={f'{mode}_nfe{n}':[] for mode in ('teacher','deploy') for n in nfes}
    fixed_rows={str(k):[] for k in cfg['validation']['candidate_budgets']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output.with_suffix('.protocol.json'),{'config_sha256':config_sha(cfg),**manifest,
        'checkpoint_sha256':checkpoint_sha,'reference_only':a.reference,'evaluation_source_sha256':source_sha(),
        'training_identity':None if a.reference else saved['identity'],'input_mode':'original Partial LiDAR',
        'point_mode':'native_fine_voxel_centers','teacher_support_stage1_nfe':2,'stage2_nfes':nfes,
        'no_second_encoder_deployment_claim':True,'gpu_admission':admission})
    with a.output.with_suffix('.jsonl').open('x',buffering=1) as log,torch.no_grad():
        for index in range(len(dev)):
            record=dev[index];path=Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
            xyz=torch.from_numpy(input_xyz(path)).cuda()
            with torch.autocast('cuda',dtype=torch.float16):pyramid=model.stage1.encode_condition(xyz)
            for nfe in nfes:
                gen=torch.Generator(device='cuda').manual_seed(seed+index)
                with torch.autocast('cuda',dtype=torch.float16):score,state=model.stage1.sample(xyz,nfe=nfe,generator=gen,condition=pyramid)
                if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()):raise ValueError('Nonfinite Stage1 evaluation output')
                values=parent_metrics(score,record,g,valid,.5)
                # Parent metrics by distance, using the same physical centers.
                predicted=np.flatnonzero((score.cpu().numpy().reshape(-1)>.5)&valid.reshape(-1));truth=record['parent_ids'].astype(np.int64)
                distance={}
                for lo,hi in ((0,10),(10,20),(20,35),(35,50)):
                    def part(ids):
                        d=np.minimum(np.linalg.norm(g.centers(g.unflatten(ids)),axis=1),np.nextafter(50.,0.))
                        return ids[(d>=lo)&(d<hi)]
                    pi,ti=part(predicted),part(truth);tp=len(np.intersect1d(pi,ti));distance[f'{lo}-{hi}']={'tp':tp,'fp':len(pi)-tp,'fn':len(ti)-tp}
                row={'sequence':record['sequence'],'frame':record['frame'],'nfe':nfe,**values,'distance_bins':distance}
                stage1_rows[str(nfe)].append(row);log.write(json.dumps({'kind':'stage1',**row})+'\n')
            teacher_ids,teacher_condition,_,provenance=assemble_stage2_scene(record,policy,g,torch.device('cuda'),teacher,cache=None)
            # Deployment source uses the same candidate-specific source seed as
            # the frozen teacher, not the separate Stage1 metric sampling seed.
            gen=torch.Generator(device='cuda').manual_seed(scene_seed(record['sequence'],record['frame'],policy['candidates']['seed']))
            with torch.autocast('cuda',dtype=torch.float16):deploy_score,_=model.stage1.sample(xyz,nfe=2,generator=gen,condition=pyramid)
            if not bool(torch.isfinite(deploy_score).all()):raise ValueError('Nonfinite shared-Encoder deployment score')
            for f in fixed_budget_metrics(deploy_score.cpu().numpy().reshape(-1),record,valid.reshape(-1),cfg['validation']['candidate_budgets']):fixed_rows[str(f['requested_k'])].append({'sequence':record['sequence'],'frame':record['frame'],**f})
            ids_np,scores,_=select_occupancy_candidates(deploy_score[0,0].cpu().numpy(),g,**policy['candidates']['selection'])
            if not len(ids_np):raise ValueError('Empty deployable support; do not drop frame')
            deploy_ids=torch.as_tensor(ids_np,device='cuda');observed=observed_children(xyz,ids_np,g)
            deploy_condition={'parent_score':torch.as_tensor(scores,device='cuda'),'observed_child':torch.as_tensor(observed,device='cuda'),
                'valid_child_mask':torch.as_tensor(g.valid_children(ids_np),device='cuda')}
            raw_target=raw.load(record['sequence'],record['frame']);_,inside=g.voxelize_points(raw_target)
            raw_target=np.ascontiguousarray(raw_target[inside],dtype=np.float32)
            observed_ids,observed_masks,_=voxelize_surface(xyz.cpu().numpy(),g)
            for mode,ids,condition in (('teacher',teacher_ids,teacher_condition),('deploy',deploy_ids,deploy_condition)):
                condition={k:v for k,v in condition.items() if k!='lidar_features'}
                condition['lidar_features']=sample_pyramid(pyramid,g.centers(g.unflatten(ids)).float(),g.origin,g.maximum).half()
                ids_cpu=ids.cpu().numpy();parent_sha=hashlib.sha256(ids_cpu.tobytes()).hexdigest()
                for nfe in nfes:
                    gen=torch.Generator(device='cuda').manual_seed(seed+index)
                    with torch.autocast('cuda',dtype=torch.float16):_,occupied,_=sample(fine_model,ids,condition,nfe=nfe,generator=gen,threshold=0.)
                    occupied_cpu=occupied.cpu().numpy()
                    metrics=fine_metrics(ids_cpu,occupied_cpu,record['parent_ids'],record['child_occ_mask'],g,
                        observed_parent_ids=observed_ids,observed_masks=observed_masks,structural=True)
                    support=dict(metrics['fine']);support['fn']-=metrics['missed_parent_fine_false_negatives']
                    from fpsgen.hier_occ_flow.evaluation import rates
                    metrics['support_only_fine']=rates(support)
                    fine=occupied_fine_coords(ids_cpu,pack_children(occupied_cpu,g),g);points=g.centers(fine,fine=True).astype(np.float32)
                    if not len(points):raise ValueError('Empty prediction; report failure rather than skip')
                    continuous=evaluate_prediction(points,raw_target,g,metric,device='cuda')
                    row={'sequence':record['sequence'],'frame':record['frame'],'nfe':nfe,'support_mode':mode,
                        'parent_ids_sha256':parent_sha,'input_sha256':file_sha(path),
                        'occupied_child_count':len(fine),'continuous':continuous,**metrics}
                    fine_rows[f'{mode}_nfe{nfe}'].append(row);log.write(json.dumps({'kind':'stage2',**row})+'\n')
            print(f"{cfg['arm']} evaluation={index+1}/900",flush=True)
    result={'status':'COMPLETE','frames':900,'stage1':{k:{'summary':summarize(rows),'rows':rows} for k,rows in stage1_rows.items()},
        'fixed_k':{k:summarize_fixed_budget(rows) for k,rows in fixed_rows.items()},
        'fixed_k_rows':fixed_rows,
        'fixed_k_source':'pre-registered84963/120000, deployment NFE2 score, no post-hoc tuning',
        'stage2':{k:{'summary':aggregate_frames(rows),'continuous_macro':{m:float(np.mean([r['continuous'][m] for r in rows]))
            for m in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')},'rows':rows} for k,rows in fine_rows.items()},
        'reference_only':a.reference,'checkpoint_sha256':checkpoint_sha,'phase':cfg['phase'],'arm':cfg['arm']}
    atomic_json(a.output,result)


if __name__=='__main__':main()

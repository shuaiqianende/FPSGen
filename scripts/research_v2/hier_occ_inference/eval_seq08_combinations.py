"""Frozen seq08/skip100: Stage1(1,2,4) x Stage2(1,2,4), raw-map CD and IoU0.5m.

Inference-only explicitly selected GPU entry. No training sources or policies are changed. Exact
CPU KD-trees avoid dense pairwise costs; all target points participate in CD.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import numpy as np
from scipy.spatial import cKDTree
import torch
from export_stage2_snapshot import admit_gpu1,validate_snapshot,predicted_policy,EncodedInput
from eval_gpu import admit_gpu0,check_gpu0_background
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_condition import input_conditions,scene_seed
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,occupied_fine_coords,pack_children
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def physical_grid_ids(xyz,g,resolution=.5,*,target_points=False):
    """Same origin/clipped-Z/sphere-intersection contract as prior seq08 audit."""
    xyz=np.asarray(xyz,dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('Finite XYZ required')
    if not np.isfinite(resolution) or resolution<=0:raise ValueError('Positive resolution required')
    origin=np.asarray(g.origin);maximum=np.asarray(g.maximum)
    shape=np.ceil((maximum-origin)/resolution).astype(np.int64)
    inside=((xyz>origin)&(xyz<maximum)).all(1)
    if target_points:inside&=(xyz*xyz).sum(1)<g.sphere_radius**2
    coord=np.floor((xyz[inside]-origin)/resolution).astype(np.int64)
    lo=origin+coord*resolution;hi=np.minimum(lo+resolution,maximum)
    nearest=np.maximum(lo,0)+np.minimum(hi,0)
    coord=coord[(nearest*nearest).sum(1)<g.sphere_radius**2]
    return np.unique(coord[:,0]+shape[0]*(coord[:,1]+shape[1]*coord[:,2]))


def support_counts(predicted,target):
    predicted=np.unique(predicted);target=np.unique(target)
    tp=int(np.intersect1d(predicted,target,assume_unique=True).size)
    fp=len(predicted)-tp;fn=len(target)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/max(tp+fp+fn,1)}


def raw_chamfer(predicted,target,*,target_tree=None,workers=4):
    """0.5*(mean NN Euclidean P->G +mean NN Euclidean G->P), meters."""
    if len(predicted)==0 or len(target)==0:return {'valid_cd':False,'cd_m':None,'pred_to_gt_m':None,'gt_to_pred_m':None}
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():raise ValueError('Nonfinite CD cloud')
    if target_tree is None:target_tree=cKDTree(target,leafsize=32)
    pred_tree=cKDTree(predicted,leafsize=32)
    pg=target_tree.query(predicted,k=1,eps=0.,workers=workers)[0]
    gp=pred_tree.query(target,k=1,eps=0.,workers=workers)[0]
    a=float(pg.mean());b=float(gp.mean())
    return {'valid_cd':True,'cd_m':.5*(a+b),'pred_to_gt_m':a,'gt_to_pred_m':b}


@torch.no_grad()
def integrate(model,ids,condition,source,nfe):
    """Reuse exact source values; never sample new noise inside Euler steps."""
    if model.training or nfe not in (1,2,4):raise ValueError('Eval mode and registered1/2/4 NFE required')
    state=source.clone()
    for i in range(nfe):
        velocity=model(ids,state=state,time=state.new_tensor(i/nfe),**condition)
        if not bool(torch.isfinite(velocity).all()):raise ValueError('Nonfinite velocity')
        state=state+velocity.float()/nfe
    if not bool(torch.isfinite(state).all()):raise ValueError('Nonfinite endpoint')
    return (state>0)&condition['valid_child_mask']


def aggregate(rows):
    out=[]
    for s1 in (1,2,4):
        for s2 in (1,2,4):
            selected=[r for r in rows if r['stage1_nfe']==s1 and r['stage2_nfe']==s2]
            if not selected:continue
            valid=[r for r in selected if r['valid_cd']]
            tp=sum(r['tp'] for r in selected);fp=sum(r['fp'] for r in selected);fn=sum(r['fn'] for r in selected)
            out.append({'stage1_nfe':s1,'stage2_nfe':s2,'frames':len(selected),'invalid_cd_frames':len(selected)-len(valid),
                'cd_m':float(np.mean([r['cd_m'] for r in valid])) if len(valid)==len(selected) else None,
                'iou05_macro':float(np.mean([r['iou05'] for r in selected])),
                'iou05_micro':tp/max(tp+fp+fn,1),
                'mean_predicted_fine_voxels':float(np.mean([r['predicted_fine_voxels'] for r in selected]))})
    return out


def requested_nfes(values):
    if not values or len(set(values))!=len(values) or any(n not in (1,2,4) for n in values):
        raise ValueError('Choose distinct registered NFE values from1/2/4')
    return tuple(values)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-frames',type=int,default=41)
    p.add_argument('--physical-gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--idle-service-pid',type=int)
    p.add_argument('--stage1-nfes',nargs='+',type=int,choices=[1,2,4],default=[1,2,4])
    p.add_argument('--stage2-nfes',nargs='+',type=int,choices=[1,2,4],default=[1,2,4])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    s1_nfes=requested_nfes(a.stage1_nfes);s2_nfes=requested_nfes(a.stage2_nfes)
    if not 1<=a.max_frames<=41 or not 1<=a.workers<=4:raise ValueError('At most41 frames and4 CPU workers')
    cfg,g=load_stage2_config(a.config);saved=torch.load(a.checkpoint,map_location='cpu');validate_snapshot(saved,cfg,g)
    reference=json.loads((a.reference/'protocol.json').read_text())
    frames=reference['frames'];assert [r['frame'] for r in frames]==list(range(0,4001,100)) and all(r['sequence']=='08' for r in frames)
    assert reference['geometry_sha256']==g.sha256
    frames=frames[:a.max_frames]
    expected_rows=len(frames)*len(s1_nfes)*len(s2_nfes)
    protocol={'checkpoint_sha256':file_sha(a.checkpoint),'checkpoint_updates':saved['successful_updates'],
        'checkpoint_frames':saved['sampler']['successful_frame_exposures'],'geometry_sha256':g.sha256,
        'stage1':cfg['stage1'],'stage2_weights':'online','frames':frames,'stage1_nfes':list(s1_nfes),'stage2_nfes':list(s2_nfes),
        'stage1_seed':'scene_seed(sequence,frame,candidates.seed)','stage2_seed_base':cfg['validation']['seed'],
        'stage2_noise':'full parent-indexed Gaussian table shared across requested NFE combinations within each frame',
        'stage1_threshold':.5,'stage2_state_threshold':0.,'extra_filter':'NONE',
        'cd':'0.5*(mean Euclidean NN prediction->raw GT +mean Euclidean NN raw GT->prediction), meters',
        'cd_gt':'All valid raw pre-Poisson local map candidate points; no point subsampling',
        'iou05':'0.5m center rasterization; origin[-50,-50,-4],200x200x17,last Z cell clipped4.4; raw-map GT cached support',
        'raw_source_manifest_sha256':reference['raw_source_manifest_sha256'],
        'reference_input_manifest_sha256':file_sha(a.reference/'input_manifest.json'),
        'code_sha256':file_sha(__file__),'training_selection':'interim requested test only; no test tuning or model selection',
        'candidate_mode':'actual predicted parents; checkpoint trained on Oracle support','raw_cd_backend':'exact scipy.cKDTree, eps=0',
        'physical_gpu':a.physical_gpu,'workers':a.workers,
        'gpu_admission_code_sha256':file_sha(HERE/'eval_gpu.py'),
        'allowed_idle_service_pid':a.idle_service_pid}
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        if json.loads((a.output/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol identity differs')
    else:atomic_json(a.output/'protocol.json',protocol)
    rows=[];journal=a.output/'per_frame.jsonl'
    if a.resume and journal.exists():rows=[json.loads(s) for s in journal.read_text().splitlines()]
    done={(r['frame'],r['stage1_nfe'],r['stage2_nfe']) for r in rows}
    atomic_json(a.output/'progress.json',{'status':'VERIFYING_RAW_SOURCE','completed_combinations':len(rows),'requested_combinations':expected_rows})
    source,source_sha=raw_source_snapshot(cfg['data_root'],[('08',r['frame']) for r in reference['frames']])
    if source_sha!=reference['raw_source_manifest_sha256']:raise ValueError('Raw map/pose/scan/protocol sources changed')
    atomic_json(a.output/'raw_sources.json',source)
    atomic_json(a.output/'admission.json',admit_gpu1() if a.physical_gpu==1 else admit_gpu0(a.idle_service_pid))
    torch.set_num_threads(4);device=torch.device('cuda:0')
    if a.physical_gpu==0:torch.cuda.set_per_process_memory_fraction(.25,device)
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval()
    model=Stage2OccupancyModel(g,**cfg['model']).to(device).eval();model.load_state_dict(saved['model'],strict=True)
    del saved
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);begin=time.perf_counter()
    for index,record in enumerate(frames):
        if a.physical_gpu==0:check_gpu0_background(a.idle_service_pid)
        frame=record['frame']
        if all((frame,s1,s2) in done for s1 in s1_nfes for s2 in s2_nfes):continue
        tick=time.perf_counter();metadata=json.loads((a.reference/'targets'/f'08_{frame:06d}.json').read_text())
        if metadata['input_sha256']!=file_sha(record['input']) or metadata['geometry_sha256']!=g.sha256:
            raise ValueError('Target/input identity mismatch')
        xyz_gt=raw.load('08',frame);label_ids,label_masks,audit=voxelize_surface(xyz_gt,g)
        _,valid=g.voxelize_points(xyz_gt);xyz_gt=xyz_gt[valid]
        if audit['valid_points']!=metadata['valid_points'] or audit['filtered_points']!=metadata['filtered_points']:
            raise ValueError('Raw-map reconstruction differs from audited target')
        with np.load(a.reference/'targets'/f'08_{frame:06d}.npz',allow_pickle=False) as cache:
            gt_half=cache['half_meter_ids'].copy()
            if not np.array_equal(label_ids,cache['parent_ids']) or not np.array_equal(label_masks,cache['child_masks']):
                raise ValueError('Reconstructed raw-map fine support differs from cache')
        if not np.array_equal(physical_grid_ids(xyz_gt,g,target_points=True),gt_half):raise ValueError('Raw-map0.5m support differs')
        gt_tree=cKDTree(xyz_gt,leafsize=32);raw_seconds=time.perf_counter()-tick
        print(f'frame={frame:06d} raw_GT={len(xyz_gt)} audited fine={audit["fine_occupied"]} load_seconds={raw_seconds:.2f}',flush=True)
        xyz=torch.from_numpy(input_xyz(record['input'])).to(device)
        tick=time.perf_counter();pyramid=adapter.encode_features(xyz);torch.cuda.synchronize();encoding_seconds=time.perf_counter()-tick
        seed=scene_seed('08',frame,cfg['candidates']['seed'])
        noise=torch.randn((g.num_parents,g.k),device=device,generator=torch.Generator(device=device).manual_seed(cfg['validation']['seed']+frame))
        for s1 in s1_nfes:
            if all((frame,s1,s2) in done for s2 in s2_nfes):continue
            tick=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):
                score,state=adapter.model.sample(xyz,nfe=s1,condition=pyramid,generator=torch.Generator(device=device).manual_seed(seed))
            if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()):raise ValueError('Nonfinite Stage1')
            torch.cuda.synchronize();stage1_seconds=time.perf_counter()-tick
            policy=predicted_policy(cfg);policy['stage1_nfe']=s1
            ids,c,provenance=input_conditions(xyz,g,EncodedInput(pyramid,score[0,0].float(),adapter.provenance),policy,seed=seed)
            if len(ids)>cfg['training']['max_parent_tokens']:raise ValueError('No token truncation permitted')
            source_noise=noise[ids]*c['valid_child_mask']
            for s2 in s2_nfes:
                if (frame,s1,s2) in done:continue
                tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):occ=integrate(model,ids,c,source_noise,s2)
                torch.cuda.synchronize();stage2_seconds=time.perf_counter()-tick
                coords=occupied_fine_coords(ids.cpu().numpy(),pack_children(occ.cpu().numpy(),g),g)
                predicted=g.centers(coords,fine=True).astype(np.float32)
                tick=time.perf_counter();metric=raw_chamfer(predicted,xyz_gt,target_tree=gt_tree,workers=a.workers)
                support=support_counts(physical_grid_ids(predicted,g),gt_half)
                row={'sequence':'08','frame':frame,'stage1_nfe':s1,'stage2_nfe':s2,**metric,
                    'iou05':support['iou'],**{k:support[k] for k in ('tp','fp','fn')},
                    'predicted_fine_voxels':len(predicted),'raw_gt_points':len(xyz_gt),'candidate_parents':len(ids),
                    'encoding_seconds':encoding_seconds,'stage1_seconds':stage1_seconds,'stage2_seconds':stage2_seconds,
                    'metric_seconds':time.perf_counter()-tick,'raw_load_seconds':raw_seconds,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
                with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                rows.append(row);done.add((frame,s1,s2))
                atomic_json(a.output/'progress.json',{'status':'RUNNING','completed_combinations':len(rows),
                    'requested_combinations':expected_rows,'completed_scenes':sum(all((r['frame'],n1,n2) in done for n1 in s1_nfes for n2 in s2_nfes) for r in frames),
                    'seconds_current_invocation':time.perf_counter()-begin,'results':aggregate(rows)})
                print(f'{len(rows)}/{expected_rows} 08/{frame:06d} S1={s1} S2={s2} CD={metric["cd_m"]} IoU05={support["iou"]:.6f} metric_seconds={row["metric_seconds"]:.2f}',flush=True)
            del ids,c,source_noise,score,state
        del noise,pyramid,xyz,gt_tree,xyz_gt
    result={'status':'COMPLETE','frames':len(frames),'combinations':len(rows),'protocol':protocol,'results':aggregate(rows),
            'seconds_current_invocation':time.perf_counter()-begin,'rows':rows}
    if len(rows)!=expected_rows:raise ValueError('Incomplete grid evaluation')
    atomic_json(a.output/'results.json',result);atomic_json(a.output/'progress.json',{k:v for k,v in result.items() if k!='rows'})
    with (a.output/'per_frame.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (a.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['results'][0]));writer.writeheader();writer.writerows(result['results'])
    print(json.dumps({'status':'COMPLETE','results':result['results']}),flush=True)


if __name__=='__main__':main()

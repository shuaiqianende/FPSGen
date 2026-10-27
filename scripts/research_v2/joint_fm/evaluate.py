"""Actual Stage1->Stage2 FM inference on predicted support, with zero extra Parents."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
import argparse,csv,json,os,hashlib,time,traceback
import numpy as np
import torch
import yaml
from fpsgen.hier_occ_flow.joint_fm_inference import POLICY,validate_policy,predicted_support,input_only_conditions,ids_sha
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.encoder_adaptation import tensor_state_sha
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.meanflow_sampling import sample_meanflow,stage1_score
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,occupied_fine_coords,pack_children
from fpsgen.hier_occ_flow.stage1_evaluation import parent_metrics
from fpsgen.hier_occ_flow.stage1_curve_metrics import fixed_budget_metrics
from fpsgen.hier_occ_flow.evaluation import fine_metrics,rates
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from unified_pointcloud_metrics import evaluate_prediction

REFERENCE=Path('/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/stage1_final_seq08_skip100_rawcd_grid124_gpu1')
BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-explicit-fp16-training-v1/experiments/desescene_explicit_fp16_v1')


def checkpoint(arm):
    root=BASE/arm;cfg=json.loads((root/'formal/config.json').read_text())
    path=root/'formal/checkpoints/resume_latest.pt';value=torch.load(path,map_location='cpu');i=value['identity'];c=value['counts']
    if (i['arm']!=arm or i['initialization_mode']!='scratch' or i['config_sha256']!=config_sha(cfg)
        or c['updates']!=10000 or c['exposures']!=20000 or i['seed']!=20261032 or value['initialization']['pretrained_checkpoint_loaded']):
        raise ValueError('Exact registered ordinaryFM Scratch20K EMA source required')
    return cfg,value,dict(path=str(path),checkpoint_sha256=file_sha(path),identity=i,counts=c)


def load_models(device):
    cfg,saved,p1=checkpoint('S1-FM-Scratch20K');g=SparseHierarchyGeometry(**cfg['geometry'])
    s1=Stage1DirectOccupancy(g,objective='fm',widths=cfg['model']['widths'],activation_checkpointing=False).to(device)
    s1.load_state_dict({k[len('stage1.'):]:v for k,v in saved['ema'].items()},strict=True);del saved
    apply_explicit_policy(s1);s1.eval().requires_grad_(False)
    models={};provenance={'stage1':p1}
    for name,arm,mode in [('no_encoder','S2-FM-NoEncoder-Scratch20K','observed'),('with_encoder','S2-FM-Encoder-Scratch20K','pyramid_observed')]:
        c,value,proof=checkpoint(arm)
        if c['geometry']!=cfg['geometry'] or proof['identity']['train_manifest_sha256']!=p1['identity']['train_manifest_sha256']:
            raise ValueError('Stage1/Stage2 geometry or original training manifest differs')
        m=Stage2OccupancyModel(g,objective='fm',backbone='ptv3',width=64,condition_mode=mode,checkpointing=False,patch_sizes=(64,128)).to(device)
        m.load_state_dict({k[len('stage2.'):]:v for k,v in value['ema'].items()},strict=True);del value
        apply_explicit_policy(m);m.eval().requires_grad_(False);m.ptv3_topology_cache='multilevel';models[name]=m
        proof['legacy_training_extra_parent_fraction']=.25
        proof['legacy_training_policy_status']='NONCOMPLIANT_WITH_USER_NO_EXTRA_PARENT_REQUIREMENT; evaluation-only historical weights'
        provenance[name]=proof
        if name=='with_encoder':
            init=json.loads((BASE/arm/'formal/initialization.json').read_text());f=init['frozen_encoder']
            if f['sha256']!=p1['checkpoint_sha256'] or tensor_state_sha(s1.encoder.state_dict())!=f['initial_state_sha256']:
                raise ValueError('Joint Stage1 Encoder is not the exact frozen Encoder used by this Stage2')
            proof['same_encoder_as_training_exact']=True;proof['shared_initialization']=init['shared_initialization']
    return cfg,g,s1,models,provenance


@torch.no_grad()
def infer(xyz,s1,models,g,frame):
    """No GT argument or GT file access. Parent support is fixed before metrics."""
    with torch.autocast('cuda',dtype=torch.float16):
        with torch.autocast('cuda',enabled=False):pyramid=s1.encoder([xyz])
        observed=s1.geometry.observed_mask(xyz);valid=s1.valid
        seed=scene_seed('08',frame,POLICY['source_seed'])
        noise=torch.randn(valid.shape,device=xyz.device,generator=torch.Generator(device=xyz.device).manual_seed(seed))*valid
        fn=lambda x,t,s:s1.flow(x,observed,pyramid,t)*valid
        endpoint=sample_meanflow(fn,noise,valid,nfe=2,scenes=1);score=stage1_score(endpoint)*valid
        ids,scores=predicted_support(score,valid,g)
        no=input_only_conditions(ids,scores,xyz,g)
        yes=input_only_conditions(ids,scores,xyz,g,pyramid=pyramid)
        if any(not torch.equal(v,yes[k]) for k,v in no.items()):raise ValueError('Shared input conditions differ')
        seed2=scene_seed('08',frame,POLICY['source_seed'])
        source=torch.randn(no['valid_child_mask'].shape,device=xyz.device,generator=torch.Generator(device=xyz.device).manual_seed(seed2))*no['valid_child_mask']
        predictions={}
        for name,m in models.items():
            c=no if name=='no_encoder' else yes
            if ('lidar_features' in c)!=(name=='with_encoder'):raise ValueError('Encoder ablation condition mismatch')
            context=m.prepare_inference_context(ids,**c)
            velocity=lambda x,t,s:m.forward_with_inference_context(context,x,t)
            for nfe in POLICY['stage2_nfes']:
                state=sample_meanflow(velocity,source,c['valid_child_mask'],nfe=nfe,scenes=1)
                predictions[name,nfe]=((state>0)&c['valid_child_mask']).cpu().numpy()
            del context
    return ids.cpu().numpy(),score.cpu(),predictions,dict(parent_ids_sha256=ids_sha(ids),parent_count=len(ids),
        extra_parent_count=0,parent_expansion=0,gt_support_used=False,actual_stage1_nfe=2,
        actual_stage1_scores=True,stage1_encoder_forwards=1,source_noise_seed=seed2,
        support_policy='exact Stage1 score>0.5; no GT/nearby/observed union/budget truncation')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=yaml.safe_load(a.config.read_text());validate_policy(config['inference'])
    if config['scope']!='evaluation_only_legacy_weights' or config['physical_gpu']!=2 or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':
        raise ValueError('Registered GPU2 evaluation-only identity required')
    gpu=gpu_preflight((2,),minimum_free_gib=22.,allow_shared_compute=False)
    if gpu['status']!='AVAILABLE':raise RuntimeError(json.dumps(gpu))
    if a.output.exists():raise FileExistsError('Preserve previous evaluation')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(21.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg,g,s1,models,proof=load_models('cuda')
    frames=list(range(0,4001,100));reference=json.loads((REFERENCE/'protocol.json').read_text())
    sources,sha=raw_source_snapshot(cfg['data_root'],[('08',f) for f in frames])
    if sha!=reference['raw_source_manifest_sha256'] or g.sha256!=reference['geometry_sha256']:raise ValueError('Frozen rawGT/geometry identity changed')
    atomic_json(a.output/'raw_sources.json',sources)
    protocol=dict(scope='ACTUAL Stage1 FM NFE2 -> Stage2 FM; predicted support; evaluation-only legacy Stage2 weights',
        frames=frames,primary_stage1_nfe=2,primary_stage2_nfe=4,stage2_nfes=[1,2,4,8],weights='EMA',
        inference=POLICY,checkpoints=proof,geometry_sha256=g.sha256,raw_source_manifest_sha256=sha,gpu=gpu,
        point_mode='native fine voxel centers; no jitter',all_gt_false_negatives_included=True,
        gt_access='GT values loaded only after both models have generated occupancies',
        timing='quality pass only; no paired speed claim',
        compliance='INFERENCE has zero extra parents; Stage2 TRAINING previously had prohibited25% extra parents and is not retrospectively corrected',
        code_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in [Path(__file__).resolve(),ROOT/'fpsgen/hier_occ_flow/joint_fm_inference.py']})
    atomic_json(a.output/'protocol.json',protocol)
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);metric=GPUKDTreeMetrics(('cd',),geometry=g);rows=[];parent_rows=[];start=time.time()
    with (a.output/'per_frame.jsonl').open('x',buffering=1) as log,torch.no_grad():
        for ix,frame in enumerate(frames):
            path=Path(cfg['data_root'])/'08/input_'/f'{frame:06d}.npy'
            meta=json.loads((REFERENCE/'targets'/f'08_{frame:06d}.json').read_text())
            if file_sha(path)!=meta['input_sha256']:raise ValueError('Original input LiDAR changed')
            xyz_np=input_xyz(path);xyz=torch.from_numpy(xyz_np).cuda()
            ids,score,predictions,prov=infer(xyz,s1,models,g,frame)
            # Ground truth is first used here, after all network predictions.
            gt=raw.load('08',frame);_,inside=g.voxelize_points(gt);target=np.ascontiguousarray(gt[inside],dtype=np.float32)
            gt_ids,gt_masks,_=voxelize_surface(gt,g)
            with np.load(REFERENCE/'targets'/f'08_{frame:06d}.npz',allow_pickle=False) as frozen:
                if not np.array_equal(gt_ids,frozen['parent_ids']) or not np.array_equal(gt_masks,frozen['child_masks']):raise ValueError('Frozen GT mismatch')
            record=dict(parent_ids=gt_ids,child_occ_mask=gt_masks)
            coarse=parent_metrics(score,record,g,s1.valid.cpu(),.5)
            fixed=fixed_budget_metrics(score.numpy().reshape(-1),record,s1.valid.cpu().numpy().reshape(-1),[84963,120000])
            parent_rows.append(dict(frame=frame,stage1=coarse,fixed_k=fixed,provenance=prov))
            obs_ids,obs_masks,_=voxelize_surface(xyz_np,g)
            for (name,nfe),occ in predictions.items():
                native=fine_metrics(ids,occ,gt_ids,gt_masks,g,observed_parent_ids=obs_ids,observed_masks=obs_masks,structural=True)
                support=dict(**{k:native['fine'][k] for k in ('tp','fp','fn')});support['fn']-=native['missed_parent_fine_false_negatives']
                native['support_only_fine']=rates(support)
                fine=occupied_fine_coords(ids,pack_children(occ,g),g);points=g.centers(fine,fine=True).astype(np.float32)
                if not len(points):raise RuntimeError('Empty prediction; never skip frames')
                continuous=evaluate_prediction(points,target,g,metric,device='cuda')
                row=dict(sequence='08',frame=frame,variant=name,stage1_nfe=2,stage2_nfe=nfe,
                    parent=coarse,native=native,continuous=continuous,occupied_child_count=len(points),
                    input_sha256=meta['input_sha256'],**prov)
                log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            atomic_json(a.output/'stage1_per_frame.json',parent_rows)
            atomic_json(a.output/'progress.json',dict(status='RUNNING',frames_completed=ix+1,frames_requested=41,rows=len(rows),
                elapsed_seconds=time.time()-start,extra_parent_count=0,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            print(f'Joint FM seq08 {ix+1}/41 parents={len(ids)} extra=0',flush=True)
    summary=[]
    for name in models:
        for nfe in (1,2,4,8):
            q=[r for r in rows if r['variant']==name and r['stage2_nfe']==nfe]
            summary.append(dict(variant=name,stage1_nfe=2,stage2_nfe=nfe,frames=41,
                **{'fine_'+k:float(np.mean([r['native']['fine'][k] for r in q])) for k in ('precision','recall','f1','iou')},
                **{k:float(np.mean([r['continuous'][k] for r in q])) for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')},
                support_only_fine_iou=float(np.mean([r['native']['support_only_fine']['iou'] for r in q])),
                missed_parent_fine_fn=float(np.mean([r['native']['missed_parent_fine_false_negatives'] for r in q])),
                occupied_child_count=float(np.mean([r['occupied_child_count'] for r in q]))))
    atomic_json(a.output/'results.json',dict(status='EVALUATED_JOINT_PREDICTED41',protocol=protocol,summary=summary,
        stage1_macro={k:float(np.mean([r['stage1'][k] for r in parent_rows])) for k in ('precision','recall','f1','iou','active_parents','fine_recall_ceiling')}))
    with (a.output/'summary.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
    atomic_json(a.output/'progress.json',dict(status='EVALUATED_JOINT_PREDICTED41',frames_completed=41,frames_requested=41,rows=len(rows),elapsed_seconds=time.time()-start,extra_parent_count=0))


if __name__=='__main__':
    try:main()
    except Exception:
        print(traceback.format_exc(),file=sys.stderr,flush=True);raise

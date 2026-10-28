"""No-training seq08/skip100 41-frame support curves, regions and FM velocity diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,os,time,traceback
import numpy as np,torch
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy,ids_to_dense_occupancy
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.stage1_curve_metrics import threshold_curve_metrics,fixed_budget_metrics
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics

BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-triplane-encoder-v1/experiments/desescene_triplane_v1')
R0=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-explicit-fp16-training-v1/experiments/desescene_explicit_fp16_v1/S1-FM-Scratch20K/formal/checkpoints/resume_latest.pt')
REFERENCE=Path('/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/stage1_final_seq08_skip100_rawcd_grid124_gpu1')
INPUT=Path('/data-12/M2024-HWZ/KITTI_Odometry')

def regional(score,valid,ids,masks,zone):
    chosen=(score>.5)&valid&zone;gtzone=zone[ids];selected=chosen[ids];tp=int((selected&gtzone).sum());fp=int(chosen.sum())-tp;fn=int(gtzone.sum())-tp
    counts=np.unpackbits(masks.astype(np.uint8)[:,None],axis=1).sum(1);denom=int(counts[gtzone].sum());num=int(counts[selected&gtzone].sum())
    return dict(precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),iou=tp/max(tp+fp+fn,1),fine_recall_ceiling=num/max(denom,1),tp=tp,fp=fp,fn=fn)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('r0','v1','v2'),required=True);p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int,default=41);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('Physical GPU mismatch')
    gpu=gpu_preflight((a.gpu,),minimum_free_gib=22.,allow_shared_compute=False)
    if gpu['status']!='AVAILABLE':raise RuntimeError(json.dumps(gpu))
    if a.output.exists():raise FileExistsError('Prior audit preserved')
    a.output.mkdir(parents=True);(a.output/'scores').mkdir();torch.set_num_threads(4)
    geometry=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2),origin=(-50.,-50.,-4.),maximum=(50.,50.,4.4),sphere_radius=50.)
    if not 1<=a.max_frames<=41:raise ValueError('Only seq08 skip100 up to41 frames is allowed')
    frames=list(range(0,4001,100))[:a.max_frames]
    checkpoint=R0 if a.model=='r0' else BASE/f'run_{a.model}/formal/checkpoints/resume_latest.pt'
    saved=torch.load(checkpoint,map_location='cpu')
    if a.model=='r0':
        if saved['counts']['updates']!=10000 or saved['counts']['exposures']!=20000:raise ValueError('R0 scratch20K incomplete')
        weights={k[len('stage1.'):]:v for k,v in saved['ema'].items() if k.startswith('stage1.')};variant='original';identity=saved['identity']
    else:
        if saved['successful_updates']!=10000 or saved['sampler']['successful_frame_exposures']!=20000:raise ValueError('V1/V2 scratch20K incomplete')
        weights=saved['ema'];variant='triplane_'+a.model;identity=saved['identity']
    model=Stage1DirectOccupancy(geometry,objective='fm',widths=(8,16,32,64),activation_checkpointing=False,encoder_variant=variant).cuda().eval().requires_grad_(False)
    model.load_state_dict(weights,strict=True);del saved,weights
    valid=model.valid.cpu().numpy().reshape(-1);grid=np.arange(geometry.num_parents,dtype=np.int64);coords=geometry.centers(geometry.unflatten(grid));radius=np.linalg.norm(coords,axis=1)
    bands={name:(radius>=lo)&(radius<hi) for name,lo,hi in [('0_20m',0,20),('20_35m',20,35),('35_50m',35,50)]};del grid,coords,radius
    thresholds=np.linspace(0,1,101);raw_thresholds=(.25,.35,.5,.65,.75);rows=[]
    raw=RawMapSource(INPUT,fast_viewpoint=True);metric=GPUKDTreeMetrics(('cd',),geometry=geometry)
    sources={p:file_sha(ROOT/'fpsgen/hier_occ_flow'/p) for p in ('triplane_encoder.py','stage1_occupancy.py','stage1_network.py')}
    atomic_json(a.output/'protocol.json',dict(model=a.model,checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),checkpoint_identity=identity,reference_protocol_sha256=file_sha(REFERENCE/'protocol.json'),source_sha256=sources,frame_count=a.max_frames,source_seed='scene_seed(08,frame,20261031)',nfe=2,amp='fp32 weights with fp16 autocast for all models',threshold=.5,distance='Euclidean XYZ center distance',gt_used_only_for_metrics=True,gpu=gpu,gradient_scope='L2 norm of d(valid-mean MSE)/d velocity output; not parameter gradient',uncomputed=['height-layer region labels','thin/road-edge structure annotation']))
    try:
        with (a.output/'per_frame.jsonl').open('x',buffering=1) as log,torch.no_grad():
            for i in range(a.max_frames):
                frame=frames[i];sequence='08';path=INPUT/sequence/'input_'/f'{frame:06d}.npy'
                meta=json.loads((REFERENCE/'targets'/f'08_{frame:06d}.json').read_text())
                if file_sha(path)!=meta['input_sha256']:raise ValueError('Frozen input changed')
                with np.load(REFERENCE/'targets'/f'08_{frame:06d}.npz',allow_pickle=False) as frozen:
                    record=dict(sequence=sequence,frame=frame,parent_ids=frozen['parent_ids'].copy(),child_occ_mask=frozen['child_masks'].copy())
                xyz=torch.from_numpy(input_xyz(path)).cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    pyramid=model.encoder([xyz]);obs=model.geometry.observed_mask(xyz);generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',frame,20261031))
                    score,_=model.sample(xyz,generator=generator,nfe=2,condition=pyramid)
                    target=ids_to_dense_occupancy(record['parent_ids'],geometry,device='cuda');noise=torch.randn(target.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20262031+i))*model.valid
                    dest=(2*target-1)*model.valid;desired=dest-noise;vt=[]
                    for t in (0.,.25,.5,.75,1.):
                        state=((1-t)*noise+t*dest)*model.valid;velocity=model.flow(state,obs,pyramid,torch.tensor([t],device='cuda'))*model.valid
                        err=(velocity.float()-desired).square();positive=target.bool()&model.valid;negative=(~target.bool())&model.valid
                        count_pos=int(positive.sum());count_neg=int(negative.sum());total=count_pos+count_neg
                        p_mse=float(err[positive].mean());n_mse=float(err[negative].mean())
                        vt.append(dict(t=t,positive_mse=p_mse,negative_mse=n_mse,positive_fraction=count_pos/total,
                            positive_output_gradient_l2=2*(count_pos*p_mse)**.5/total,negative_output_gradient_l2=2*(count_neg*n_mse)**.5/total))
                score=score.float().cpu().numpy().reshape(-1);ids=record['parent_ids'].astype(np.int64);masks=record['child_occ_mask']
                # Full FP32 score is retained locally, never committed to Git.
                np.save(a.output/'scores'/f"{record['sequence']}_{record['frame']:06d}.npy",score,allow_pickle=False)
                curve=threshold_curve_metrics(score,record,valid,thresholds);sort=sorted((r['recall'],r['precision']) for r in curve);auc=float(np.trapz([x[1] for x in sort],[x[0] for x in sort]))
                requested=threshold_curve_metrics(score,record,valid,raw_thresholds);fixed=fixed_budget_metrics(score,record,valid,[84963,120000]);observed=obs.cpu().numpy().reshape(-1).astype(bool)
                zones={name:regional(score,valid,ids,masks,zone) for name,zone in bands.items()};zones['observed']=regional(score,valid,ids,masks,observed);zones['unobserved']=regional(score,valid,ids,masks,~observed)
                # Evaluation uses original unclosed map points, never model inputs.
                raw_gt=raw.load(record['sequence'],record['frame']);_,inside=geometry.voxelize_points(raw_gt);raw_gt=raw_gt[inside]
                pred_ids=np.flatnonzero((score>.5)&valid);pred_xyz=geometry.centers(geometry.unflatten(pred_ids)).astype(np.float32)
                gt_radius=np.linalg.norm(raw_gt,axis=1);pred_radius=np.linalg.norm(pred_xyz,axis=1)
                for name,lo,hi in [('0_20m',0,20),('20_35m',20,35),('35_50m',35,50)]:
                    pp=np.ascontiguousarray(pred_xyz[(pred_radius>=lo)&(pred_radius<hi)],dtype=np.float32);gg=np.ascontiguousarray(raw_gt[(gt_radius>=lo)&(gt_radius<hi)],dtype=np.float32)
                    zones[name]['cd_m']=metric.evaluate(torch.from_numpy(pp).cuda(),torch.from_numpy(gg).cuda())['cd_m'] if len(pp) and len(gg) else None
                hist,_=np.histogram(score[valid],bins=np.linspace(0,1,101));row=dict(index=i,sequence=record['sequence'],frame=record['frame'],thresholds=requested,curve=curve,pr_auc_101=auc,fixed_k=fixed,regions=zones,score_histogram=hist.tolist(),velocity_diagnostics=vt)
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');atomic_json(a.output/'progress.json',dict(status='RUNNING',frames_completed=i+1,frames_requested=a.max_frames))
                if (i+1)%10==0:print(a.model,'frames',i+1,'/',a.max_frames,flush=True)
        atomic_json(a.output/'results.json',dict(status='COMPLETE',frames=len(rows),macro_at_05={k:float(np.mean([r['thresholds'][2][k] for r in rows])) for k in ('precision','recall','f1','iou','fine_recall_ceiling','active_parents')},mean_pr_auc_101=float(np.mean([r['pr_auc_101'] for r in rows]))))
        atomic_json(a.output/'progress.json',dict(status='COMPLETE',frames_completed=len(rows),frames_requested=a.max_frames))
    except Exception as e:
        atomic_json(a.output/'failure.json',dict(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),frames_completed=len(rows)));raise

if __name__=='__main__':main()

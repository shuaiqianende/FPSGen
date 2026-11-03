"""Evaluate the B4 Stage2 EMA on the frozen 41 seq08 frames, predicted support."""
import argparse,csv,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import numpy as np,torch,yaml
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.t1_stage1_adapter import FrozenT1Stage1Adapter
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_no_lidar import remove_lidar_and_score
from fpsgen.hier_occ_flow.stage2_remove_score import remove_score
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.joint_fm_inference import predicted_support,input_only_conditions
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,pack_children,occupied_fine_coords
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from eval_seq08_combinations import physical_grid_ids,support_counts
from eval_seq08_skip10_joint_2x4_gpu3 import jsd_metrics

REFERENCE=Path('/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/stage1_final_seq08_skip100_rawcd_grid124_gpu1')
BASE=ROOT/'experiments/stage2_b4_paired_v1'
T1CFG=ROOT/'configs/research_v2/hier_occ_flow/stage2_b4_with_encoder.yaml'

def load_model(cfg,g,arm,ck):
    opts=dict(cfg['model']);opts['condition_mode']='pyramid_observed'
    model=Stage2OccupancyModel(g,**opts).cuda()
    with torch.no_grad():model.score[2].weight.zero_();model.score[2].bias.zero_()
    dummy=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'])
    if arm=='no_encoder':remove_lidar_and_score(model,dummy)
    else:remove_score(model,dummy,model.state_dict())
    apply_explicit_policy(model);model.load_state_dict(ck['ema'],strict=True)
    return model.eval().requires_grad_(False)

@torch.no_grad()
def run(args):
    gpu={'with_encoder':2,'no_encoder':3}[args.arm]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Incorrect physical GPU')
    cfgpath=ROOT/f'configs/research_v2/hier_occ_flow/stage2_b4_{args.arm}.yaml';cfg=yaml.safe_load(cfgpath.read_text())
    teacher=yaml.safe_load(T1CFG.read_text());g=SparseHierarchyGeometry(**cfg['geometry'])
    path=BASE/args.arm/'formal/checkpoints/resume_latest.pt';ck=torch.load(path,map_location='cpu')
    if ck['arm']!=args.arm or ck['mode']!='formal' or ck['config_sha256']!=file_sha(cfgpath) or ck['batch_per_gpu']!=4:raise ValueError('Checkpoint identity mismatch')
    if not args.smoke and (ck['step']!=25000 or ck['frames']!=100000):raise ValueError('Only completed 100K EMA accepted')
    ref=json.loads((REFERENCE/'protocol.json').read_text());frames=ref['frames']
    if ref['geometry_sha256']!=g.sha256 or [(x['sequence'],x['frame']) for x in frames]!=[('08',i) for i in range(0,4001,100)]:raise ValueError('Frozen seq08 identity mismatch')
    if args.smoke:frames=frames[:1]
    out=BASE/args.arm/('eval_smoke_1' if args.smoke else 'eval_seq08_41');out.mkdir(parents=True,exist_ok=True)
    protocol={'arm':args.arm,'frames':[r['frame'] for r in frames],'checkpoint_sha256':file_sha(path),'checkpoint_step':ck['step'],'checkpoint_frames':ck['frames'],'weights':'EMA','stage1':'T1 depth-moment EMA','stage1_nfe':2,'stage2_nfe':4,'stage1_threshold':.5,'stage2_endpoint_threshold':0.,'stage1_seed':'scene_seed(seq,frame,20261018)','stage2_seed':'20261031 + frame, full parent-indexed source','support':'Stage1 predicted only; no GT parents or expansion','score_injection':False,'encoder_injection':args.arm=='with_encoder','GT':'raw-map surface hits','points':'native fine voxel centers','geometry_sha256':g.sha256,'code_sha256':file_sha(__file__)}
    protocol_path=out/'protocol.json';journal=out/'per_frame.jsonl'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Evaluation identity changed; use a new directory')
    atomic_json(protocol_path,protocol)
    rows=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    done={r['frame'] for r in rows};torch.set_num_threads(4);torch.cuda.set_device(0)
    adapter=FrozenT1Stage1Adapter(**teacher['stage1']['adapter']).cuda().eval();apply_explicit_policy(adapter.model)
    model=load_model(cfg,g,args.arm,ck);del ck
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);metric=GPUKDTreeMetrics(('cd',),geometry=None)
    for record in frames:
        frame=record['frame']
        if frame in done:continue
        start=time.perf_counter();xyz=torch.from_numpy(input_xyz(record['input'])).cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            pyramid=adapter.encode_features(xyz)
            seed=scene_seed('08',frame,teacher['candidates']['seed'])
            score,state=adapter.model.sample(xyz,nfe=2,condition=pyramid,generator=torch.Generator(device='cuda').manual_seed(seed))
            ids,scores=predicted_support(score[0,0].float(),adapter.model.valid,g)
            cond=input_only_conditions(ids,scores,xyz,g,pyramid=pyramid if args.arm=='with_encoder' else None);cond.pop('parent_score')
            valid=cond['valid_child_mask'];noise=torch.randn((g.num_parents,g.k),device='cuda',generator=torch.Generator(device='cuda').manual_seed(20261031+frame));x=noise[ids]*valid;del noise
            for i in range(4):
                v=model(ids,state=x,time=x.new_tensor(i/4),**cond)
                if not bool(torch.isfinite(v).all()):raise ValueError('Nonfinite Stage2 velocity')
                x=x+v.float()/4
            occ=(x>0)&valid
        torch.cuda.synchronize();network_seconds=time.perf_counter()-start
        ids_cpu=ids.cpu().numpy();masks=pack_children(occ.cpu().numpy(),g);fine=occupied_fine_coords(ids_cpu,masks,g);pred=g.centers(fine,fine=True).astype(np.float32)
        target=raw.load('08',frame);_,valid_gt=g.voxelize_points(target);target=np.ascontiguousarray(target[valid_gt],dtype=np.float32)
        gtids,gtmask,_=voxelize_surface(target,g);gtfine=occupied_fine_coords(gtids,gtmask,g)
        meta=json.loads((REFERENCE/'targets'/f'08_{frame:06d}.json').read_text())
        if meta['input_sha256']!=file_sha(record['input']) or meta['geometry_sha256']!=g.sha256:raise ValueError('Reference input or geometry mismatch')
        with np.load(REFERENCE/'targets'/f'08_{frame:06d}.npz',allow_pickle=False) as reference:
            if not np.array_equal(gtids,reference['parent_ids']) or not np.array_equal(gtmask,reference['child_masks']):raise ValueError('Raw target support differs from frozen reference')
        if not len(pred) or not len(target):raise ValueError('Empty prediction or target')
        scores_out=metric.evaluate(torch.from_numpy(np.ascontiguousarray(pred)).cuda(),torch.from_numpy(target).cuda())
        row={'sequence':'08','frame':frame,'candidate_parents':len(ids_cpu),'occupied_fine':len(fine),'network_seconds':network_seconds,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,**scores_out}
        row['native_fine_iou']=support_counts(g.flatten(fine,fine=True),g.flatten(gtfine,fine=True))['iou']
        for res,key in ((.5,'iou05'),(.2,'iou02'),(.1,'iou01')):
            row[key]=support_counts(physical_grid_ids(pred,g,resolution=res),physical_grid_ids(target,g,resolution=res,target_points=True))['iou']
        row.update(jsd_metrics(target,pred))
        with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row);done.add(frame);atomic_json(out/'progress.json',{'status':'RUNNING','completed':len(rows),'total':len(frames),'last_frame':frame})
        print(json.dumps(row,allow_nan=False),flush=True)
        del xyz,pyramid,score,state,ids,scores,cond,x,occ,v,pred,target
    if len(rows)!=len(frames):raise ValueError('Incomplete frames')
    keys=('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d','native_fine_iou','candidate_parents','occupied_fine','network_seconds')
    summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
    with (out/'per_frame.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(out/'results.json',{'status':'COMPLETE','protocol':protocol,'summary':summary,'rows':rows})
    atomic_json(out/'progress.json',{'status':'COMPLETE','completed':len(rows),'total':len(frames),'summary':summary})

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['with_encoder','no_encoder'],required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:run(args)
    except Exception as e:
        out=BASE/args.arm/('eval_smoke_1' if args.smoke else 'eval_seq08_41');out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'progress.json',{'status':'FAILED','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()});raise
if __name__=='__main__':main()

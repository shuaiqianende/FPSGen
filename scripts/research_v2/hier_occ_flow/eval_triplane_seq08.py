"""Evaluate completed scratch20K TriPlane Stage1 on frozen seq08/skip100."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
import argparse,csv,json,os
import numpy as np,torch,yaml
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.stage1_evaluation import parent_metrics
from fpsgen.hier_occ_flow.stage1_curve_metrics import fixed_budget_metrics
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.meanflow_sampling import sample_meanflow,stage1_score
from fpsgen.hier_occ_flow.voxelization import voxelize_surface
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from unified_pointcloud_metrics import evaluate_prediction

REFERENCE=Path('/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/stage1_final_seq08_skip100_rawcd_grid124_gpu1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());cfg['geometry']=yaml.safe_load((a.config.parent/cfg['geometry']).read_text());g=validate_config(cfg)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['training']['physical_gpus'][0]):raise ValueError('Wrong GPU')
    ck=torch.load(a.checkpoint,map_location='cpu');identity=ck['identity']
    if ck['initialization_mode']!='scratch' or ck['successful_updates']!=10000 or ck['sampler']['successful_frame_exposures']!=20000 or identity['experiment']!='desescene-triplane-encoder-v1':raise ValueError('Own completed scratch20K checkpoint required')
    if a.output.exists():raise FileExistsError('Preserve old evaluation')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    model=Stage1DirectOccupancy(g,objective='fm',widths=cfg['model']['widths'],activation_checkpointing=False,encoder_variant=cfg['model']['encoder_variant']).cuda()
    model.load_state_dict(ck['ema'],strict=True);model.eval().requires_grad_(False)
    reference=json.loads((REFERENCE/'protocol.json').read_text());frames=list(range(0,4001,100))
    _,sha=raw_source_snapshot(cfg['data_root'],[('08',f) for f in frames])
    if sha!=reference['raw_source_manifest_sha256'] or reference['geometry_sha256']!=g.sha256:raise ValueError('Frozen GT or geometry differs')
    atomic_json(a.output/'protocol.json',dict(identity=identity,checkpoint_sha256=file_sha(a.checkpoint),frames=frames,nfes=[1,2,4,8],seed=20261031,gt_source_sha256=sha,point_mode='Parent voxel centers',precision='FP32 weights, CUDA FP16 autocast',stage2='not evaluated'))
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);metric=GPUKDTreeMetrics(('cd',),geometry=g);rows=[]
    with torch.no_grad(),(a.output/'per_frame.jsonl').open('x',buffering=1) as log:
        for index,f in enumerate(frames):
            path=Path(cfg['data_root'])/'08/input_'/f'{f:06d}.npy';meta=json.loads((REFERENCE/'targets'/f'08_{f:06d}.json').read_text())
            if file_sha(path)!=meta['input_sha256']:raise ValueError('Input XYZ changed')
            xyz=torch.from_numpy(input_xyz(path)).cuda();gt=raw.load('08',f);_,inside=g.voxelize_points(gt);target=np.ascontiguousarray(gt[inside],dtype=np.float32)
            ids,masks,_=voxelize_surface(gt,g);record=dict(sequence='08',frame=f,parent_ids=ids,child_occ_mask=masks)
            with np.load(REFERENCE/'targets'/f'08_{f:06d}.npz',allow_pickle=False) as old:
                if not np.array_equal(ids,old['parent_ids']) or not np.array_equal(masks,old['child_masks']):raise ValueError('Frozen GT changed')
            with torch.autocast('cuda',dtype=torch.float16):
                pyramid=model.encoder([xyz]);observed=model.geometry.observed_mask(xyz);valid=model.valid
                source=torch.randn(valid.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',f,20261031)))*valid
                for nfe in (1,2,4,8):
                    fn=lambda x,t,s:model.flow(x,observed,pyramid,t)*valid
                    endpoint=sample_meanflow(fn,source,valid,nfe=nfe,scenes=1);score=stage1_score(endpoint)*valid
                    selected=torch.nonzero(score.reshape(-1)>.5).flatten();points=g.centers(g.unflatten(selected)).float().cpu().numpy()
                    if not len(points):raise ValueError('Empty prediction')
                    parent=parent_metrics(score,record,g,valid,.5);fixed=fixed_budget_metrics(score.cpu().numpy().reshape(-1),record,valid.cpu().numpy().reshape(-1),[84963,120000])
                    continuous=evaluate_prediction(points,target,g,metric,device='cuda')
                    row=dict(frame=f,nfe=nfe,parent=parent,fixed_k=fixed,continuous=continuous);log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            atomic_json(a.output/'progress.json',dict(status='RUNNING',frames_completed=index+1,frames_requested=41))
            print(f"{cfg['model']['encoder_variant']} seq08 {index+1}/41",flush=True)
    summary=[]
    for nfe in (1,2,4,8):
        q=[r for r in rows if r['nfe']==nfe]
        summary.append(dict(nfe=nfe,frames=41,**{k:float(np.mean([r['parent'][k] for r in q])) for k in ('precision','recall','f1','iou','active_parents','fine_recall_ceiling')},**{k:float(np.mean([r['continuous'][k] for r in q])) for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')}))
    atomic_json(a.output/'results.json',dict(status='EVALUATED_SEQ08_41',summary=summary));atomic_json(a.output/'progress.json',dict(status='EVALUATED_SEQ08_41',frames_completed=41,frames_requested=41))
    with (a.output/'summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)

if __name__=='__main__':main()

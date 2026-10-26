"""Independent scratch20K quality on frozen41 only; never load a pretrained model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
import argparse,csv,json,os,time
import numpy as np
import torch
from fpsgen.hier_occ_flow.scratch20k_training import load_config,ScratchSystem,own_source_sha
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.meanflow_sampling import sample_meanflow,stage1_score
from fpsgen.hier_occ_flow.encoder_free_stage2 import assemble_oracle_encoder_free_scene
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,pack_children,occupied_fine_coords
from fpsgen.hier_occ_flow.stage1_evaluation import parent_metrics
from fpsgen.hier_occ_flow.stage1_curve_metrics import fixed_budget_metrics
from fpsgen.hier_occ_flow.evaluation import fine_metrics
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from unified_pointcloud_metrics import evaluate_prediction

REFERENCE=Path('/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/stage1_final_seq08_skip100_rawcd_grid124_gpu1')


class Recorder:
    def __init__(self,backend):self.backend=backend
    def evaluate(self,p,g):return self.backend.evaluate(p,g)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,g=load_config(a.config)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):raise ValueError('Own GPU only')
    if a.output.exists():raise FileExistsError('Never overwrite previous quality results')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    saved=torch.load(a.checkpoint,map_location='cpu');identity=saved['identity'];counts=saved['counts']
    if identity['initialization_mode']!='scratch' or identity['config_sha256']!=config_sha(cfg) or identity['source_sha256']!=own_source_sha() or counts['updates']!=10000 or counts['exposures']!=20000:
        raise ValueError('Own final scratch20K checkpoint required')
    model=ScratchSystem(cfg,g,'cuda');model.load_state_dict(saved['ema'],strict=True);model.eval().requires_grad_(False);del saved
    reference=json.loads((REFERENCE/'protocol.json').read_text())
    if [(r['sequence'],r['frame']) for r in reference['frames']]!=[('08',f) for f in range(0,4001,100)] or reference['geometry_sha256']!=g.sha256:raise ValueError('Frozen41 identity changed')
    sources,sha=raw_source_snapshot(cfg['data_root'],[('08',f) for f in cfg['validation']['frames']])
    if sha!=reference['raw_source_manifest_sha256']:raise ValueError('Original GT sources changed')
    atomic_json(a.output/'raw_sources.json',sources)
    protocol=dict(identity=identity,checkpoint_sha256=file_sha(a.checkpoint),frames=cfg['validation']['frames'],nfes=[1,2,4,8],
        initialization='Random network trained20K total; EMA evaluation',point_mode='coarse Parent centers' if model.is_s1 else 'native fine voxel centers',
        quality_scope='Stage1 standalone; no legacy Stage2' if model.is_s1 else 'Oracle GT coarse+25%empty support, nondeployment; no Encoder',
        source_sha256=own_source_sha(),raw_source_manifest_sha256=sha,cd_backend='exact CUDA KD-tree',physical_iou='corrected physical grids',
        timing_scope='quality only; no formal speed claim',pretrained_model_loaded=False)
    atomic_json(a.output/'protocol.json',protocol)
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);metric=Recorder(GPUKDTreeMetrics(('cd',),geometry=g));rows=[]
    with (a.output/'per_frame.jsonl').open('x',buffering=1) as log,torch.no_grad():
        for index,f in enumerate(cfg['validation']['frames']):
            path=Path(cfg['data_root'])/'08/input_'/f'{f:06d}.npy';xyz=torch.from_numpy(input_xyz(path)).cuda()
            meta=json.loads((REFERENCE/'targets'/f'08_{f:06d}.json').read_text())
            if file_sha(path)!=meta['input_sha256']:raise ValueError('Frozen input changed')
            gt=raw.load('08',f);_,inside=g.voxelize_points(gt);target=np.ascontiguousarray(gt[inside],dtype=np.float32)
            ids,masks,_=voxelize_surface(gt,g);record=dict(sequence='08',frame=f,parent_ids=ids,child_occ_mask=masks)
            with np.load(REFERENCE/'targets'/f'08_{f:06d}.npz',allow_pickle=False) as old:
                if not np.array_equal(ids,old['parent_ids']) or not np.array_equal(masks,old['child_masks']):raise ValueError('Frozen GT mismatch')
            obs_ids,obs_masks,_=voxelize_surface(xyz.cpu().numpy(),g)
            with torch.autocast('cuda',dtype=torch.float16):
                if model.is_s1:
                    network=model.stage1
                    with torch.autocast('cuda',enabled=False):pyramid=network.encoder([xyz])
                    observed=network.geometry.observed_mask(xyz);valid=network.valid
                    source=torch.randn(valid.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',f,20261031)))*valid
                else:
                    parents,c,labels,_=assemble_oracle_encoder_free_scene(record,xyz,g,seed=scene_seed('08',f,20261018),device='cuda',keep_observed=True,negative_fraction=.25)
                    valid=c['valid_child_mask'];source=torch.randn(valid.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',f,20261031)))*valid
                    model.stage2.ptv3_topology_cache='multilevel';context=model.stage2.prepare_inference_context(parents,**c)
                for nfe in (1,2,4,8):
                    if model.is_s1:
                        fn=lambda x,t,s:network.flow(x,observed,pyramid,t,time_end=s,validate_times=False)*valid
                        endpoint=sample_meanflow(fn,source,valid,nfe=nfe,scenes=1);score=stage1_score(endpoint)*valid
                        selected=torch.nonzero(score.reshape(-1)>.5).flatten();points=g.centers(g.unflatten(selected)).float().cpu().numpy()
                        native=parent_metrics(score,record,g,valid,.5)
                        fixed=fixed_budget_metrics(score.cpu().numpy().reshape(-1),record,valid.cpu().numpy().reshape(-1),[84963,120000])
                        counts_out=dict(occupied_parent_count=len(selected));native_name='parent';extra={'fixed_k':fixed}
                    else:
                        fn=lambda x,t,s:model.stage2.forward_with_inference_context(context,x,t)
                        endpoint=sample_meanflow(fn,source,valid,nfe=nfe,scenes=1);occupied=(endpoint>0)&valid
                        ids_np=parents.cpu().numpy();occ_np=occupied.cpu().numpy();fine=occupied_fine_coords(ids_np,pack_children(occ_np,g),g)
                        points=g.centers(fine,fine=True).astype(np.float32)
                        native=fine_metrics(ids_np,occ_np,ids,masks,g,observed_parent_ids=obs_ids,observed_masks=obs_masks,structural=True)
                        counts_out=dict(parent_count=len(parents),occupied_child_count=len(points));native_name='native';extra={}
                    if not len(points):raise RuntimeError('Empty output; do not skip frame')
                    continuous=evaluate_prediction(points,target,g,metric,device='cuda')
                    row=dict(sequence='08',frame=f,nfe=nfe,**{native_name:native},continuous=continuous,**extra,**counts_out)
                    log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            atomic_json(a.output/'progress.json',dict(status='RUNNING',frames_completed=index+1,frames_requested=41,rows=len(rows)))
            print(f"{cfg['arm']} seq08 {index+1}/41",flush=True)
    summary=[]
    for nfe in (1,2,4,8):
        subset=[r for r in rows if r['nfe']==nfe]
        quality=[r['parent']['iou'] if model.is_s1 else r['native']['fine']['iou'] for r in subset]
        summary.append(dict(nfe=nfe,frames=41,native_iou=float(np.mean(quality)),**{k:float(np.mean([r['continuous'][k] for r in subset])) for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')}))
    atomic_json(a.output/'results.json',dict(status='EVALUATED',frames=41,protocol=protocol,summary=summary))
    atomic_json(a.output/'progress.json',dict(status='EVALUATED',frames_completed=41,frames_requested=41,rows=len(rows)))
    with (a.output/'summary.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

if __name__=='__main__':main()

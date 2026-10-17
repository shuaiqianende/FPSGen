"""GPU1 inference authorized by the user; training GPU2/3 policy is unchanged.

Separate from the source-hashed training directory so adding an export tool does
not invalidate an actively running experiment's exact-resume source identity.
Predicted exports never load GT. Optional Oracle diagnostics are separate.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,config_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_condition import input_conditions,scene_seed
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.inference import sample,export_occupancy
from fpsgen.hier_occ_flow.stage2_visualization import write_xyz_ply
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def admit_gpu1():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Explicit user-authorized physical GPU1 only')
    result=subprocess.check_output(['nvidia-smi','--id=1',
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free,util=map(float,result.strip().split(','))
    processes=subprocess.check_output(['nvidia-smi','--id=1',
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    pids=[int(v.strip()) for v in processes.splitlines() if v.strip().isdigit()]
    if pids or free<16384 or util>10:raise RuntimeError('GPU1 busy; no interruption or memory competition allowed')
    return {'physical_gpu':1,'free_mib':free,'utilization_percent':util,'compute_pids':pids,
            'authorization':'user: 可以用1号卡','training_gpu_policy_modified':False}


def validate_snapshot(value,cfg,g):
    identity=value.get('identity',{})
    if (identity.get('experiment')!='hs-occflow-stage2-384-v1'
        or identity.get('geometry_sha256')!=g.sha256 or identity.get('config_sha256')!=config_sha(cfg)
        or identity.get('stage1_checkpoint_sha256')!=cfg['stage1']['checkpoint_sha256']
        or value.get('initialization')!='scratch_stage2' or value.get('parameter_sync_pass') is not True):
        raise ValueError('Training snapshot/config/geometry/Stage1 identity mismatch')


def predicted_policy(cfg):
    policy=copy.deepcopy(cfg['candidates']);policy['mode']='predicted'
    # This is a diagnostic transfer from Oracle training, not a calibrated or
    # fine-tuned predicted-parent checkpoint. Never modify training cfg identity.
    policy['selection'].update(strategy='threshold',threshold=.5,expansion=0,overflow='error')
    return policy


class EncodedInput:
    def __init__(self,features,score,provenance):self.features=features;self.score=score;self.provenance=provenance
    def __call__(self,xyz,*,nfe,seed,with_scores):return self.features,self.score if with_scores else None


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--oracle-labels',type=Path);parser.add_argument('--nfes',nargs='+',type=int,default=[1,2,4,8])
    parser.add_argument('--weights',nargs='+',choices=['online','ema'],default=['online'])
    args=parser.parse_args()
    if any(n not in (1,2,4,8) for n in args.nfes):raise ValueError('Registered1/2/4/8 NFE only')
    if args.output.exists():raise FileExistsError('Never overwrite an existing export')
    cfg,g=load_stage2_config(args.config);saved=torch.load(args.checkpoint,map_location='cpu');validate_snapshot(saved,cfg,g)
    admission=admit_gpu1();torch.set_num_threads(4);device=torch.device('cuda:0')
    args.output.mkdir(parents=True)
    atomic_json(args.output/'admission.json',admission)
    frame=int(args.input.stem);sequence=args.input.parent.parent.name
    if args.input.parent.name!='input_':raise ValueError('Canonical sequence/input_/frame.npy required')
    xyz=torch.from_numpy(input_xyz(args.input)).to(device)
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval()
    seed=scene_seed(sequence,frame,cfg['candidates']['seed'])
    features,score=adapter(xyz,nfe=cfg['candidates']['stage1_nfe'],seed=seed,with_scores=True)
    encoded=EncodedInput(features,score,adapter.provenance)
    write_xyz_ply(args.output/'partial_lidar.ply',xyz.cpu().numpy(),(160,160,160))
    model=Stage2OccupancyModel(g,**cfg['model']).to(device).eval()
    protocol={'stage2_checkpoint':str(args.checkpoint.resolve()),'stage2_checkpoint_sha256':file_sha(args.checkpoint),
        'successful_updates':saved['successful_updates'],'successful_frame_exposures':saved['sampler']['successful_frame_exposures'],
        'geometry_sha256':g.sha256,'input':str(args.input.resolve()),'input_sha256':file_sha(args.input),
        'training_candidate_mode':cfg['candidates']['mode'],'inference_score_threshold':.5,
        'predicted_support_status':'uncalibrated diagnostic; checkpoint trained with Oracle support',
        'stage1':adapter.provenance,'stage1_nfe':cfg['candidates']['stage1_nfe'],'stage1_seed':seed,
        'stage2_seed':cfg['validation']['seed'],'stage2_endpoint_threshold':cfg['validation']['threshold'],
        'points_are_quantized_surface_hit_centers':True,'fixed180k_points':False,'physical_gpu':1,'exports':[]}
    # Run input-only predicted support BEFORE opening any optional GT artifact.
    for mode in ['predicted']+(['oracle'] if args.oracle_labels else []):
        ids_gt=None
        policy=predicted_policy(cfg) if mode=='predicted' else cfg['candidates']
        if mode=='oracle':
            metadata=json.loads(args.oracle_labels.with_suffix('.json').read_text())
            if metadata['geometry_sha256']!=g.sha256 or metadata['input_sha256']!=file_sha(args.input):
                raise ValueError('Oracle GT geometry/input identity mismatch')
            with np.load(args.oracle_labels,allow_pickle=False) as labels:
                ids_gt=labels['parent_ids'].astype(np.int64);masks_gt=labels['child_masks'].copy()
            write_xyz_ply(args.output/'gt_fine_occupancy.ply',g.centers(occupied_fine_coords(ids_gt,masks_gt,g),fine=True),(30,200,60))
            protocol['oracle_label_sha256']=file_sha(args.oracle_labels)
        ids,c,source=input_conditions(xyz,g,encoded,policy,seed=seed,oracle_ids=ids_gt)
        if len(ids)>cfg['training']['max_parent_tokens']:raise RuntimeError('No unvalidated token truncation')
        coarse_path=args.output/('stage1_coarse_occupancy.ply' if mode=='predicted' else 'oracle_candidate_parents.ply')
        write_xyz_ply(coarse_path,g.centers(g.unflatten(ids.cpu().numpy())),(230,180,20))
        for weight in args.weights:
            state=saved['model'] if weight=='online' else saved.get('ema')
            if state is None:raise ValueError('Requested model state absent')
            model.load_state_dict(state,strict=True)
            for nfe in args.nfes:
                before=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):
                    _,occ,calls=sample(model,ids,c,nfe=nfe,generator=torch.Generator(device=device).manual_seed(cfg['validation']['seed']),threshold=cfg['validation']['threshold'])
                torch.cuda.synchronize();seconds=time.perf_counter()-before
                out=args.output/mode/weight/f'nfe{nfe}'
                export_occupancy(out,ids.cpu().numpy(),occ.cpu().numpy(),g)
                metadata={'mode':mode,'weight':weight,'actual_stage2_nfe':calls,'stage2_seconds':seconds,
                    'candidate_tokens':len(ids),'fine_occupied_voxels':int(occ.sum()),'gt_used_for_candidates':mode=='oracle',
                    'deployable_input_contract':mode=='predicted','stage1_source':source,
                    'stage2_checkpoint_sha256':protocol['stage2_checkpoint_sha256']}
                atomic_json(out/'provenance.json',metadata);protocol['exports'].append({**metadata,'ply':str((out/'fine_surface_hit_centers.ply').resolve())})
                atomic_json(args.output/'manifest.json',protocol)
                print(json.dumps({k:metadata[k] for k in ('mode','weight','actual_stage2_nfe','candidate_tokens','fine_occupied_voxels','stage2_seconds')}),flush=True)
        del ids,c
    if adapter.encoder_forward_count!=1:raise RuntimeError('LiDAR encoding must be shared across modes/weights/NFE')
    protocol.update(status='COMPLETE',actual_stage1_encoder_forwards=adapter.encoder_forward_count)
    atomic_json(args.output/'manifest.json',protocol)


if __name__=='__main__':main()

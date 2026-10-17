"""Input-only real384 Occupancy Stage1 ->8D PTv3 FM/BCE ->fine768 sparse export."""
from common import ROOT
import argparse
import json
import os
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,config_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_condition import input_conditions,scene_seed
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel as HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.inference import sample,export_occupancy
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.stage2_visualization import write_xyz_ply


def validate_inputs(rows):
    if not rows or any(set(r)!={'id','input'} for r in rows):raise ValueError('Input-only id/input manifest; GT forbidden')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    if any(Path(r['id']).name!=r['id'] or r['id'] in ('.','..') for r in rows):raise ValueError('Unsafe output ID')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nfe',type=int,choices=(1,2,4,8),default=4);p.add_argument('--ema',action='store_true');args=p.parse_args()
    cfg,g=load_stage2_config(args.config)
    if cfg['candidates']['mode']!='predicted':raise ValueError('Deploy requires predicted support; oracle is not end-to-end inference')
    rows=json.loads(args.input_manifest.read_text());validate_inputs(rows)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3'):raise ValueError('GPU2/3 only')
    gate=gpu_preflight((int(visible),))
    if gate['status']!='AVAILABLE':raise RuntimeError('Stage1/other job active; export not launched')
    value=torch.load(args.checkpoint,map_location='cpu');identity=value['identity']
    if identity['experiment']!='hs-occflow-stage2-384-v1' or identity['geometry_sha256']!=g.sha256 or identity['config_sha256']!=config_sha(cfg):
        raise ValueError('Deployment config/checkpoint mismatch')
    model=HierarchicalOccupancyModel(g,**cfg['model']).cuda().eval();state=value['ema'] if args.ema else value['model']
    if state is None:raise ValueError('EMA absent')
    model.load_state_dict(state,strict=True);adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda().eval()
    for index,row in enumerate(rows):
        xyz=torch.from_numpy(input_xyz(row['input'])).cuda()
        input_path=Path(row['input'])
        if input_path.parent.name!='input_' or not input_path.stem.isdigit():
            raise ValueError('Expected canonical sequence/input_/frame.npy pairing for the registered per-frame Stage1 seed')
        seed=scene_seed(input_path.parent.parent.name,int(input_path.stem),cfg['candidates']['seed'])
        ids,c,source=input_conditions(xyz,g,adapter,cfg['candidates'],seed=seed)
        if len(ids)>cfg['training']['max_parent_tokens']:raise RuntimeError('Unvalidated scene size; no random truncation')
        dtype=torch.bfloat16 if cfg['training']['precision']=='bf16' else torch.float16
        with torch.autocast('cuda',dtype=dtype,enabled=cfg['training']['precision']!='fp32'):
            _,occ,calls=sample(model,ids,c,nfe=args.nfe,generator=torch.Generator(device='cuda').manual_seed(cfg['validation']['seed']+index),threshold=cfg['validation']['threshold'])
        export_occupancy(args.output/row['id'],ids.cpu().numpy(),occ.cpu().numpy(),g)
        write_xyz_ply(args.output/row['id']/'partial_lidar.ply',xyz.cpu().numpy(),(160,160,160))
        write_xyz_ply(args.output/row['id']/'stage1_selected_parent_centers.ply',g.centers(g.unflatten(ids.cpu().numpy())),(230,180,20))
        atomic_json(args.output/row['id']/'provenance.json',{'stage1':source,'stage2_checkpoint_sha256':file_sha(args.checkpoint),
            'gt_loaded':False,'input':row['input'],'input_sha256':file_sha(row['input']),
            'actual_stage2_nfe':calls,'stage1_encoder_forwards':1,'points_are_fine_occupancy_centers':True})


if __name__=='__main__':main()

"""Input-only parent occupancy and candidate interface, no GT, density or points."""
from eval_stage1_occ import load_model
from train_stage1_occ import load_config
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def validate_inputs(rows):
    if not rows or any(set(row)!={'id','input'} or Path(row['id']).name!=row['id'] or row['id'] in ('.','..') for row in rows):
        raise ValueError('Only safe scene id and input XYZ path allowed; GT keys forbidden')
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate scene IDs')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--input-manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--nfe',type=int,choices=(1,2,4,8),default=4);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--ema',action='store_true')
    args=p.parse_args(); rows=json.loads(args.input_manifest.read_text());validate_inputs(rows)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3'): raise ValueError('PhysicalGPU2/3 only')
    gate=gpu_preflight((int(visible),))
    if gate['status']!='AVAILABLE': atomic_json(args.output/'blocked.json',gate);return
    cfg=load_config(args.config); model,g,ckpt=load_model(cfg,args.checkpoint,torch.device('cuda'),args.ema)
    sha=file_sha(args.checkpoint)
    for index,row in enumerate(rows):
        xyz=torch.from_numpy(input_xyz(row['input'])).cuda()
        generator=torch.Generator(device='cuda').manual_seed(cfg['validation']['seed']+index)
        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['precision']=='fp16'):
            score,state=model.sample(xyz,generator=generator,nfe=args.nfe)
        if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()): raise RuntimeError('Invalid occupancy')
        array=score.cpu().numpy()[0,0];ids=np.flatnonzero((array>args.threshold)&model.valid.cpu().numpy()[0,0]).astype(np.uint32)
        scores=array.ravel()[ids]; dest=args.output/row['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'manifest.json').exists(): raise FileExistsError('Output already contains an exported scene')
        with (dest/'occupancy_score.npz.tmp').open('wb') as f: np.savez_compressed(f,score=array)
        os.replace(dest/'occupancy_score.npz.tmp',dest/'occupancy_score.npz')
        for name,a in (('parent_ids',ids),('parent_scores',scores)):
            with (dest/f'{name}.npy.tmp').open('wb') as f: np.save(f,a,allow_pickle=False)
            os.replace(dest/f'{name}.npy.tmp',dest/f'{name}.npy')
        atomic_json(dest/'manifest.json',{'input':row['input'],'input_sha256':file_sha(row['input']),'checkpoint_sha256':sha,
            'geometry_sha256':g.sha256,'geometry':g.to_dict(),'successful_updates':ckpt['successful_updates'],
            'gt_accessed':False,'stage1_encoder_forwards':1,'velocity_nfe':args.nfe if model.objective=='fm' else 0,
            'threshold':args.threshold,'score_definition':'FM_clipped_endpoint_proxy' if model.objective=='fm' else 'BCE_sigmoid',
            'candidate_parents':len(ids),'fine_occupancy_not_generated':True})

if __name__=='__main__': main()

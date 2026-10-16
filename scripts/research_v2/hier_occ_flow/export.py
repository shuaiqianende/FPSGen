"""Input-only deployment: one frozen Stage1 encoding, sparse occupancy generation."""
from common import ROOT
import argparse
import json
import os
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.trainer import build_model
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.condition import FrozenVoxDetCondition,sample_pyramid,observed_children
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.candidate_selection import select_candidates
from fpsgen.hier_occ_flow.inference import sample,export_occupancy
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--input-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--nfe',type=int,choices=(1,2,4,8),default=4)
    args=p.parse_args(); cfg=json.loads(args.config.read_text())
    if cfg['candidates']['mode']!='predicted': raise ValueError('Deployment requires a frozen predicted-parent policy')
    frames=json.loads(args.input_manifest.read_text())
    if not frames or any(set(f)!={'id','input'} for f in frames): raise ValueError('Inference manifest must be input-only; GT fields forbidden')
    if len({f['id'] for f in frames})!=len(frames): raise ValueError('Duplicate frame IDs')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3'): raise ValueError('Physical GPU2 or GPU3 required')
    status=gpu_preflight((int(visible),))
    if status['status']!='AVAILABLE': raise RuntimeError(status)
    device=torch.device('cuda'); g=SparseHierarchyGeometry(**cfg['geometry'])
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    if checkpoint['identity']['geometry_sha256']!=g.sha256: raise ValueError('Geometry mismatch')
    model=build_model(cfg,g).to(device).eval(); model.load_state_dict(checkpoint['model'],strict=True)
    encoder=FrozenVoxDetCondition(cfg['stage1_checkpoint']).to(device)
    for index,row in enumerate(frames):
        name=str(row['id'])
        if Path(name).name!=name or name in ('.','..'): raise ValueError('Unsafe frame ID')
        xyz=input_xyz(row['input']); pyramid,density=encoder(torch.as_tensor(xyz,device=device),with_density=True)
        ids,scores=select_candidates(density.cpu().numpy(),g,**cfg['candidates']['selection'])
        if not len(ids): raise RuntimeError('No candidates')
        parent_ids=torch.as_tensor(ids,device=device)
        conditions={'parent_score':torch.as_tensor(scores,device=device),'observed_child':torch.as_tensor(observed_children(xyz,ids,g),device=device),
            'valid_child_mask':torch.as_tensor(g.valid_children(ids),device=device)}
        if 'pyramid' in cfg['model']['condition_mode']:
            centers=g.centers(g.unflatten(parent_ids)).float()
            conditions['lidar_features']=sample_pyramid(pyramid,centers,encoder.spec.origin,encoder.spec.maximum)
        _,occ,nfe=sample(model,parent_ids,conditions,nfe=args.nfe,generator=torch.Generator(device=device).manual_seed(cfg['validation']['seed']+index),threshold=cfg['validation']['threshold'])
        export_occupancy(args.output/name,ids,occ.cpu().numpy(),g)
        atomic_json(args.output/name/'provenance.json',{'input':row['input'],'input_sha256':file_sha(row['input']),
            'gt_loaded':False,'stage1_forwards':1,'actual_stage2_network_calls':nfe,'checkpoint_sha256':file_sha(args.checkpoint)})

if __name__=='__main__': main()

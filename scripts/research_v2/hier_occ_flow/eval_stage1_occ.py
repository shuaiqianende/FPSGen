from train_stage1_occ import load_config
import argparse
import json
import os
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.stage1_evaluation import evaluate_stage1
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def load_model(config,checkpoint,device,use_ema=False):
    g=validate_config(config); value=torch.load(checkpoint,map_location='cpu')
    expected_experiment=(config['experiment'] if config.get('scope') in ('architecture_pilot','architecture_fidelity_pilot')
                         else 'hs-occflow-stage1-occ384-v1')
    if value['identity'].get('experiment')!=expected_experiment or value['identity']['geometry_sha256']!=g.sha256 or value['identity']['objective']!=config['model']['objective']:
        raise ValueError('Checkpoint is not the matching384 occupancy experiment')
    model=Stage1DirectOccupancy(g,objective=config['model']['objective'],widths=config['model']['widths'],
        activation_checkpointing=False,
        encoder_variant=config['model'].get('encoder_variant','original'),
        omd_config=config['model'].get('omd3d'),
        dense_fidelity_config=config['model'].get('dense_fidelity'))
    state=value['ema'] if use_ema else value['model']
    if state is None: raise ValueError('No EMA saved')
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),g,value


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int);p.add_argument('--ema',action='store_true')
    args=p.parse_args();visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3'): raise ValueError('Select physical GPU2 or3')
    gate=gpu_preflight((int(visible),))
    if gate['status']!='AVAILABLE': atomic_json(args.output,{'status':'BLOCKED','resources':gate});return
    cfg=load_config(args.config); model,g,ckpt=load_model(cfg,args.checkpoint,torch.device('cuda'),args.ema)
    result=evaluate_stage1(model,cfg,g,torch.device('cuda'),max_frames=args.max_frames)
    result.update(checkpoint_sha256=file_sha(args.checkpoint),successful_updates=ckpt['successful_updates'],ema=args.ema)
    atomic_json(args.output,result)

if __name__=='__main__':main()

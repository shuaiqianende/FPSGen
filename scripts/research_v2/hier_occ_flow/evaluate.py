from common import ROOT
import argparse
import json
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.trainer import build_model
from fpsgen.hier_occ_flow.validation import evaluate_model
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True,help='Frozen config.json produced by trainer')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--max-frames',type=int); p.add_argument('--ema',action='store_true'); args=p.parse_args()
    if args.device=='cuda':
        import os
        from fpsgen.hier_occ_flow.resources import gpu_preflight
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if visible not in ('2','3'): raise ValueError('Select physical GPU2 or GPU3')
        status=gpu_preflight((int(visible),))
        if status['status']!='AVAILABLE': raise RuntimeError(status)
    torch.set_num_threads(4); cfg=json.loads(args.config.read_text()); g=SparseHierarchyGeometry(**cfg['geometry']); device=torch.device(args.device)
    value=torch.load(args.checkpoint,map_location='cpu')
    if value['identity']['geometry_sha256']!=g.sha256: raise ValueError('Checkpoint geometry mismatch')
    model=build_model(cfg,g).to(device); state=value['ema'] if args.ema else value['model']
    if state is None: raise ValueError('No EMA saved')
    model.load_state_dict(state,strict=True); encoder=None
    if 'pyramid' in cfg['model']['condition_mode'] or cfg['candidates']['mode']=='predicted':
        from fpsgen.hier_occ_flow.condition import FrozenVoxDetCondition
        encoder=FrozenVoxDetCondition(cfg['stage1_checkpoint']).to(device)
    report=evaluate_model(model,cfg,g,device,encoder,max_frames=args.max_frames)
    report.update(checkpoint_updates=value['successful_updates'],ema_used=args.ema)
    atomic_json(args.output,report); print(args.output)

if __name__=='__main__': main()

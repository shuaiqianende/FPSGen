from common import ROOT
import argparse
import os
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,config_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel as HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.stage2_evaluation import evaluate_stage2
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ema',action='store_true')
    p.add_argument('--visualization-output',type=Path);args=p.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3'):raise ValueError('Select only physical GPU2 or3')
    admission=gpu_preflight((int(visible),))
    if admission['status']!='AVAILABLE':atomic_json(args.output,{'status':'BLOCKED','gpu':admission});return
    cfg,g=load_stage2_config(args.config);_,dev,manifests=dataset_identity(cfg,g,verify=True)
    saved=torch.load(args.checkpoint,map_location='cpu');identity=saved['identity']
    if identity['experiment']!='hs-occflow-stage2-384-v1' or identity['geometry_sha256']!=g.sha256 or identity['config_sha256']!=config_sha(cfg) or any(identity[k]!=v for k,v in manifests.items()):
        raise ValueError('Stage2 checkpoint/config/data identity mismatch')
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda().eval()
    model=HierarchicalOccupancyModel(g,**cfg['model']).cuda().eval()
    state=saved['ema'] if args.ema else saved['model']
    if state is None:raise ValueError('Requested EMA absent')
    model.load_state_dict(state,strict=True)
    result=evaluate_stage2(model,cfg,g,torch.device('cuda'),adapter,dev,visualization_root=args.visualization_output)
    result.update(successful_updates=saved['successful_updates'],checkpoint_sha256=file_sha(args.checkpoint),ema=args.ema)
    atomic_json(args.output,result)


if __name__=='__main__':main()

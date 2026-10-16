from common import ROOT
import argparse
import json
from pathlib import Path
import yaml
from fpsgen.hier_occ_flow.trainer import run_training


def main(expected='fm'):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--resume',type=Path)
    p.add_argument('--max-updates',type=int); args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if cfg['model']['objective']!=expected: raise ValueError('Script/objective mismatch')
    cfg['geometry']=yaml.safe_load((args.config.parent/cfg['geometry']).read_text())
    run_training(cfg,args.output,device_name=args.device,resume=args.resume,max_updates=args.max_updates)

if __name__=='__main__': main()

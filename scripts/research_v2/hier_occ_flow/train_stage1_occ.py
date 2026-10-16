"""Scratch 384 Stage1 occupancy; never invoke the historical density runner."""
from common import ROOT
import argparse
from pathlib import Path
import yaml
from fpsgen.hier_occ_flow.stage1_training import run_stage1,validate_config


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    cfg['geometry']=yaml.safe_load((Path(path).parent/cfg['geometry']).read_text())
    validate_config(cfg)
    return cfg


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path); p.add_argument('--max-updates',type=int); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); run_stage1(load_config(args.config),args.output,resume=args.resume,max_updates=args.max_updates,smoke=args.smoke)

if __name__=='__main__': main()

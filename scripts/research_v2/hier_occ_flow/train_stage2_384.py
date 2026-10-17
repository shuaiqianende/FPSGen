from common import ROOT
import argparse
from pathlib import Path
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.stage2_trainer import run_stage2


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--max-updates',type=int);p.add_argument('--transition-from',type=Path)
    args=p.parse_args();cfg,g=load_stage2_config(args.config)
    run_stage2(cfg,g,args.output,resume=args.resume,max_updates=args.max_updates,transition=args.transition_from)


if __name__=='__main__':main()

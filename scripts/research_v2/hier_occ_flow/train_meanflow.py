"""Registered MeanFlow runner; source/interval RNG and successful exposure accounting."""
from common import ROOT
import argparse
from pathlib import Path
from fpsgen.hier_occ_flow.meanflow_training import load_config,run_training

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--max-updates',type=int,default=10000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg,g=load_config(a.config);run_training(cfg,g,a.output,resume=a.resume,max_updates=a.max_updates,smoke=a.smoke)

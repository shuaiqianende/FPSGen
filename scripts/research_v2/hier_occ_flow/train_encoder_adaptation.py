from common import ROOT
import argparse
from pathlib import Path
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config,run_adaptation

if __name__=='__main__':
    p=argparse.ArgumentParser('Independent pretrained-EMA Encoder adaptation')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--max-updates',type=int,default=10000);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();cfg,g=load_adaptation_config(a.config)
    run_adaptation(cfg,g,a.output,resume=a.resume,max_updates=a.max_updates,smoke=a.smoke)

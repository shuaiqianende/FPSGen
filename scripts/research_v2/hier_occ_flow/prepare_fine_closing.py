"""Create independent sparse morphology targets, with atomic/resumable shards."""
from common import ROOT
import argparse,json
from pathlib import Path
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.morphology import FineClosingSpec
from fpsgen.hier_occ_flow.morphology_storage import derive_dataset,processor_sha
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--connectivity',type=int,choices=[6,26],default=26);p.add_argument('--resume',action='store_true')
    a=p.parse_args();cfg,g=load_stage2_config(a.config);spec=FineClosingSpec(connectivity=a.connectivity)
    a.output.mkdir(parents=True,exist_ok=True)
    settings=json.loads(json.dumps({'geometry':g.to_dict(),'transform':spec.to_dict(),
        'target_transform_sha256':spec.sha256,'target_processor_sha256':processor_sha()}))
    path=a.output/'morphology.json'
    if path.exists():
        if not a.resume or json.loads(path.read_text())!=settings:raise ValueError('Target transform changed or resume not explicit')
    else:atomic_json(path,settings)
    for name,key in (('train','train_dataset'),('development','development_dataset')):
        out=a.output/name
        derive_dataset(cfg[key],out,g,spec,workers=a.workers,resume=a.resume and out.exists())


if __name__=='__main__':main()

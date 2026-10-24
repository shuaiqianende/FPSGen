"""CPU-only registered-data hash/access audit; no synthetic substitutes."""
from common import ROOT
import argparse,json,time
from pathlib import Path
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config,datasets
from fpsgen.hier_occ_flow.storage import atomic_json
from fpsgen.hier_occ_flow.raw_gt import RawMapSource


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve data access audit')
    cfg,g=load_adaptation_config(a.config);start=time.perf_counter();train,dev,identity=datasets(cfg,g,verify=True)
    missing=[]
    for ds in (train,dev):
        for row in ds.index:
            path=Path(cfg['data_root'])/row['sequence'].decode()/'input_'/f"{int(row['frame']):06d}.npy"
            if not path.is_file():missing.append(str(path))
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);record=dev[0];gt=raw.load(record['sequence'],record['frame'])
    result={'status':'PASS' if not missing else 'BLOCKED','data_root':cfg['data_root'],
        'train_frames':len(train),'development_frames':len(dev),**identity,'shard_hashes_verified':True,
        'missing_original_inputs':missing,'raw_gt_first_development_frame':{
            'sequence':record['sequence'],'frame':record['frame'],'shape':list(gt.shape)},'seconds':time.perf_counter()-start}
    atomic_json(a.output,result);print(json.dumps(result,indent=2))
    if missing:raise RuntimeError('Missing registered raw inputs; formal experiments prohibited')


if __name__=='__main__':main()

from common import ROOT
import argparse
from pathlib import Path
import time
import numpy as np
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=RawMapSource('/data-12/M2024-HWZ/KITTI_Odometry');rows=[]
    for seq,frame in [('00',0),('00',75),('09',200),('10',550)]:
        source._load(seq);source.fast_viewpoint=False;tick=time.perf_counter();reference=source.load(seq,frame);reference_seconds=time.perf_counter()-tick
        source.fast_viewpoint=True;tick=time.perf_counter();fast=source.load(seq,frame);fast_seconds=time.perf_counter()-tick
        exact=np.array_equal(reference,fast)
        row={'sequence':seq,'frame':frame,'rows':len(reference),'exact_array_equal':exact,'reference_seconds':reference_seconds,'lookup_seconds':fast_seconds}
        rows.append(row);atomic_json(args.output,{'status':'PASS' if all(r['exact_array_equal'] for r in rows) else 'FAIL','rows':rows,
            'scope':'raw candidate arrays including row order and float32 coordinates','lookup_code_sha256':file_sha(ROOT/'fpsgen/hier_occ_flow/viewpoint_lookup.py')})
        print(row,flush=True)
        if not exact:raise RuntimeError('Optimized viewpoint filter changed GT')

if __name__=='__main__':main()

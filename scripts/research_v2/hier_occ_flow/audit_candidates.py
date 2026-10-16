"""Offline candidate recall using an existing input-only Stage1 density export."""
from common import geometry
import argparse
from pathlib import Path
import numpy as np
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.candidate_selection import select_candidates,candidate_metrics


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--density',type=Path,required=True); p.add_argument('--label',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); g=geometry(args.geometry)
    with np.load(args.density,allow_pickle=False) as a: density=a['density']
    with np.load(args.label,allow_pickle=False) as a: ids,masks=a['parent_ids'],a['child_occ_mask']
    rows=[]
    settings=[{'threshold':t,'strategy':'threshold'} for t in (.01,.05,.1,.25,.5,1.)]
    settings += [{'top_k':k,'strategy':'topk'} for k in (50000,100000,200000)]
    settings += [{'threshold':.5,'strategy':'threshold','expansion':1,'budget':200000}]
    for options in settings:
        selected,_=select_candidates(density,g,**options)
        rows.append({'selection':options,**candidate_metrics(selected,ids,masks,g)})
    atomic_json(args.output,{'scope':'single development frame diagnostic, not frozen candidate policy','geometry_sha256':g.sha256,
        'density_sha256':file_sha(args.density),'raw_occupancy_label_sha256':file_sha(args.label),'rows':rows})
    print(rows)

if __name__=='__main__': main()

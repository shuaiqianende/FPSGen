"""Full sparse-label integrity/stats; optional exact raw-parent reference checks."""
from common import ROOT
import argparse
import json
from pathlib import Path
import time
import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.voxelization import unpack_children


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--reference-frames',type=int,default=0)
    args=p.parse_args();g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    raw=None
    if args.reference_frames:
        from fpsgen.hier_occ_flow.raw_gt import RawMapSource
        raw=RawMapSource('/data-12/M2024-HWZ/KITTI_Odometry')
    rows=[];keys={};dataset_shas={};start=time.perf_counter()
    valid_parents=3722136  # Independently measured voxel/cell-sphere intersection; checked by CPU construction audit.
    for split,expected in (('train',17870),('development',900)):
        ds=SparseShardDataset(args.root/split,g,verify=True)
        if len(ds)!=expected:raise ValueError(f'{split} incomplete:{len(ds)}/{expected}')
        keys[split]=set();dataset_shas[split]=file_sha(args.root/split/'manifest.json')
        references=set(np.linspace(0,len(ds)-1,args.reference_frames,dtype=int))
        for index in range(len(ds)):
            record=ds[index];ids=record['parent_ids'];masks=record['child_occ_mask']
            key=(record['sequence'],record['frame']);keys[split].add(key)
            if record['sequence']=='08' or not len(ids) or np.any(ids[1:]<=ids[:-1]) or np.any(masks==0):raise ValueError('Invalid sparse labels')
            occ=unpack_children(masks,g);valid=g.valid_children(ids.astype(np.int64))
            if np.any(occ&~valid):raise ValueError('Occupied child outside ROI')
            exact=None
            if index in references and raw is not None:
                xyz=raw.load(record['sequence'],record['frame']);_,keep=g.voxelize_points(xyz)
                parent=np.floor((xyz[keep]-np.array(g.origin))/np.array(g.parent_size)).astype(np.int64)
                exact=bool(np.array_equal(ids,np.unique(g.flatten(parent))))
                if not exact:raise ValueError('Direct raw parent and fine-child OR disagree')
            centers=g.centers(g.unflatten(ids.astype(np.int64)));radius=np.linalg.norm(centers,axis=1)
            rows.append({'split':split,'sequence':record['sequence'],'frame':record['frame'],'active_parents':len(ids),
                'positive_fraction':len(ids)/valid_parents,'fine_occupied':int(occ.sum()),'raw_direct_or_exact':exact,
                'range_counts':{f'{a}-{b}':int(((radius>=a)&(radius<b)).sum()) for a,b in ((0,10),(10,20),(20,35),(35,51))}})
        if len(keys[split])!=len(ds):raise ValueError('Duplicate frame')
    if keys['train']&keys['development']:raise ValueError('Split overlap')
    parents=np.array([r['active_parents'] for r in rows])
    report={'status':'PASS_FULL_DATA_AUDIT','geometry_sha256':g.sha256,'manifest_sha256':dataset_shas,
        'counts':{k:len(v) for k,v in keys.items()},'seq08_excluded':True,'rows':rows,
        'parent_quantiles':{str(q):float(np.percentile(parents,q)) for q in (0,50,90,99,100)},
        'bytes':sum(f.stat().st_size for f in args.root.rglob('*') if f.is_file()),'seconds':time.perf_counter()-start}
    atomic_json(args.root/'full_data_audit.json',report)

if __name__=='__main__':main()

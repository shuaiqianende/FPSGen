"""Full sparse-label integrity/stats; optional exact raw-parent reference checks."""
from common import ROOT
import argparse
import json
from pathlib import Path
import time
import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.voxelization import unpack_children,pack_children


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--reference-frames',type=int,default=0)
    args=p.parse_args();g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    raw=None
    if args.reference_frames:
        from fpsgen.hier_occ_flow.raw_gt import RawMapSource
        raw=RawMapSource('/data-12/M2024-HWZ/KITTI_Odometry')
    rows=[];keys={};dataset_shas={};start=time.perf_counter()
    frozen=json.loads((args.root/'split_manifest.json').read_text())
    source=json.loads((args.root/'source_manifest.json').read_text());source_sha=file_sha(args.root/'source_manifest.json')
    if source['split_sha256']!=file_sha(args.root/'split_manifest.json'):raise ValueError('Frozen split bytes changed')
    guard={(r['sequence'],r['frame']) for r in frozen['excluded_guard']}
    if len(guard)!=360:raise ValueError('Guard split changed')
    # One reusable packed ROI lookup; no repeated[M,8,3] world-grid allocation
    # for each of18,770 frames and no dense fine training label on disk.
    valid_bits=np.empty(g.num_parents,dtype=np.uint8)
    for begin in range(0,g.num_parents,65536):
        ids=np.arange(begin,min(begin+65536,g.num_parents));valid_bits[ids]=pack_children(g.valid_children(ids),g)
    valid_parents=int(np.count_nonzero(valid_bits));popcount=np.array([bin(i).count('1') for i in range(256)])
    patterns=np.zeros(256,dtype=np.int64)
    for split,expected in (('train',17870),('development',900)):
        ds=SparseShardDataset(args.root/split,g,verify=True)
        if len(ds)!=expected:raise ValueError(f'{split} incomplete:{len(ds)}/{expected}')
        if ds.meta['source_manifest_sha256']!=source_sha:raise ValueError('Raw-source manifest mismatch')
        keys[split]=set();dataset_shas[split]=file_sha(args.root/split/'manifest.json')
        references=set(np.linspace(0,len(ds)-1,args.reference_frames,dtype=int))
        for index in range(len(ds)):
            record=ds[index];ids=record['parent_ids'];masks=record['child_occ_mask']
            key=(record['sequence'],record['frame']);keys[split].add(key)
            if record['sequence']=='08' or not len(ids) or np.any(ids[1:]<=ids[:-1]) or np.any(masks==0):raise ValueError('Invalid sparse labels')
            input_path=Path('/data-12/M2024-HWZ/KITTI_Odometry')/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
            partial=np.load(input_path,mmap_mode='r',allow_pickle=False)
            if partial.ndim!=2 or partial.shape[0]!=18000 or partial.shape[1]<3 or not np.isfinite(partial[:,:3]).all():
                raise ValueError(f'Invalid partial XYZ:{input_path}')
            if np.any(masks & ~valid_bits[ids]):raise ValueError('Occupied child outside ROI')
            patterns+=np.bincount(masks,minlength=256)
            exact=None
            if index in references and raw is not None:
                xyz=raw.load(record['sequence'],record['frame']);_,keep=g.voxelize_points(xyz)
                parent=np.floor((xyz[keep]-np.array(g.origin))/np.array(g.parent_size)).astype(np.int64)
                exact=bool(np.array_equal(ids,np.unique(g.flatten(parent))))
                if not exact:raise ValueError('Direct raw parent and fine-child OR disagree')
            centers=g.centers(g.unflatten(ids.astype(np.int64)));radius=np.minimum(np.linalg.norm(centers,axis=1),np.nextafter(50.,0.))
            rows.append({'split':split,'sequence':record['sequence'],'frame':record['frame'],'active_parents':len(ids),
                'input_points':len(partial),'input_sha256':file_sha(input_path),
                'positive_fraction':len(ids)/valid_parents,'fine_occupied':int(popcount[masks].sum()),'raw_direct_or_exact':exact,
                'range_counts':{f'{a}-{b}':int(((radius>=a)&(radius<b)).sum()) for a,b in ((0,10),(10,20),(20,35),(35,50))}})
        if len(keys[split])!=len(ds):raise ValueError('Duplicate frame')
        expected_keys={(r['sequence'],r['frame']) for r in frozen[split]}
        if keys[split]!=expected_keys or keys[split]&guard:raise ValueError('Frozen split content changed or guard leakage')
    if keys['train']&keys['development']:raise ValueError('Split overlap')
    parents=np.array([r['active_parents'] for r in rows])
    report={'status':'PASS_FULL_DATA_AUDIT','geometry_sha256':g.sha256,'manifest_sha256':dataset_shas,
        'counts':{k:len(v) for k,v in keys.items()},'guard_frames':len(guard),'seq08_excluded':True,'rows':rows,
        'source_manifest_sha256':source_sha,'frozen_split_sha256':file_sha(args.root/'split_manifest.json'),
        'parent_quantiles':{str(q):float(np.percentile(parents,q)) for q in (0,50,90,99,100)},
        'valid_parent_cells':valid_parents,'child_pattern_histogram':patterns.tolist(),
        'range_policy':'parent representative centers; sphere-edge centers clipped to <50m',
        'logical_bytes_counting_hardlinks':sum(f.stat().st_size for f in args.root.rglob('*') if f.is_file()),
        'unique_inode_bytes':sum(v for v in {(f.stat().st_dev,f.stat().st_ino):f.stat().st_size for f in args.root.rglob('*') if f.is_file()}.values()),
        'seconds':time.perf_counter()-start}
    atomic_json(args.root/'full_data_audit.json',report)

if __name__=='__main__':main()

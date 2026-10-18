"""384+8 read-only split-wide mask audit and optional one-train-frame raw audit."""
from common import ROOT
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.occupancy_sparsity import mask_histogram,summarize_histogram
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.voxelization import unpack_children,voxelize_surface
from fpsgen.hier_occ_flow.raw_gt import RawMapSource


def audit_split(ds,g,max_frames):
    count=len(ds) if max_frames==0 else min(max_frames,len(ds))
    histogram=np.zeros(256,dtype=np.int64);lengths=[];maps={}
    start=time.perf_counter()
    # Read masks only for the population pass; do not copy parent IDs or build
    # dense grids, float child states, point tokens or extra label datasets.
    for i,row in enumerate(ds.index[:count]):
        shard=int(row['shard']);offset=int(row['offset']);length=int(row['length'])
        if shard not in maps:
            maps[shard]=np.memmap(ds.root/ds.meta['shards'][shard]['masks'],dtype=g.mask_dtype,mode='r')
        histogram+=mask_histogram(maps[shard][offset:offset+length]);lengths.append(length)
        if (i+1)%1000==0:print(f'{ds.root.name}: {i+1}/{count} frames',flush=True)
    result=summarize_histogram(histogram)
    result.update(frames=count,available_frames=len(ds),all_frames_audited=count==len(ds),
        manifest_sha256=file_sha(ds.root/'manifest.json'),seconds=time.perf_counter()-start,
        pattern_histogram=histogram.tolist(),parent_count_quantiles={str(q):float(np.percentile(lengths,q)) for q in (50,90,99,100)})
    selected=[]
    for q in (50,90,99,100):
        index=int(np.argmin(np.abs(np.asarray(lengths)-np.percentile(lengths,q))))
        record=ds[index];ids=record['parent_ids'].astype(np.int64)
        valid=g.valid_children(ids);occupied=unpack_children(record['child_occ_mask'],g)
        if bool((occupied&~valid).any()):raise ValueError('GT occupies an ROI-invalid child')
        selected.append({'quantile':q,'index':index,'sequence':record['sequence'],'frame':record['frame'],
            **summarize_histogram(mask_histogram(record['child_occ_mask'])),
            'valid_child_slots':int(valid.sum()),'valid_child_fill_fraction':int(occupied.sum())/int(valid.sum()),
            'roi_boundary_parents':int((~valid.all(1)).sum())})
    result['representatives']=selected
    return result


def raw_audit(cfg,g,train):
    # The first fixed TRAIN record only; sequence08 is never used to decide GT.
    record=train[0]
    if record['sequence']=='08':raise ValueError('No test frames in target-design audit')
    start=time.perf_counter();raw=RawMapSource(cfg['data_root'],fast_viewpoint=True).load(record['sequence'],record['frame'])
    ids,masks,checks=voxelize_surface(raw,g)
    if not np.array_equal(ids,record['parent_ids']) or not np.array_equal(masks,record['child_occ_mask']):
        raise ValueError('Rebuilt raw pre-Poisson GT disagrees with fixed labels')
    _,valid=g.voxelize_points(raw);raw=raw[valid]
    tree=cKDTree(raw,leafsize=32)
    # Positive k=2 includes self, measuring nearest distinct map row; duplicate
    # coordinates (if any) intentionally appear as zero distances.
    spacing=tree.query(raw,k=2,eps=0,workers=4)[0][:,1]
    from fpsgen.voxel_point.dataset import load_xyz
    from audit_geometry import stats
    poisson=load_xyz(Path(cfg['data_root'])/record['sequence']/'gt_possion'/f"{record['frame']:06d}.ply").numpy()
    _,_,sensitivity=stats(raw,poisson,g,np.random.default_rng(20261018))
    return {'scope':'one training-frame diagnostic; not population hole-rate truth','sequence':record['sequence'],'frame':record['frame'],
        'raw_gt_matches_dataset':True,'checks':checks,'nearest_distinct_row_distance_m':{str(q):float(np.percentile(spacing,q)) for q in (0,50,90,99,100)},
        'raw_vs_poisson':sensitivity['raw_vs_poisson'],'raw_thinning_stability':sensitivity['raw_thinning_stability'],
        'raw_map_original_downsampling_provenance':'Not established by the existing map file; no assumed0.1m fact',
        'seconds':time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=16,help='Per split; explicit0 audits all frames')
    p.add_argument('--raw-train-frame',action='store_true');a=p.parse_args()
    if a.max_frames<0 or a.output.exists():raise ValueError('Nonnegative budget and fresh output required')
    cfg,g=load_stage2_config(a.config);train,dev,identity=dataset_identity(cfg,g)
    report={'status':'RUNNING','geometry':g.to_dict(),'geometry_sha256':g.sha256,
        'data_identity':identity,'gt_mutated':False,'test_frames_used':False,'splits':{}}
    for name,ds in (('train',train),('development',dev)):
        report['splits'][name]=audit_split(ds,g,a.max_frames);atomic_json(a.output,report)
    if a.raw_train_frame:
        print('Rebuilding one original-map training frame for spacing/thinning diagnostic',flush=True)
        report['raw_train_frame']=raw_audit(cfg,g,train);atomic_json(a.output,report)
    report['status']='COMPLETE';atomic_json(a.output,report)
    print(json.dumps({name:{k:r[k] for k in ('frames','occupied_children_per_parent_mean','fine_fill_fraction_inside_occupied_parents_nominal8')} for name,r in report['splits'].items()}),flush=True)


if __name__=='__main__':main()

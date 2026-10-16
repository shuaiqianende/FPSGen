"""Real pre-Poisson GT audit. Never tunes on sequence08 or persists raw maps."""
from common import ROOT
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,unpack_children,occupied_fine_coords
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.storage import atomic_json,SparseShardWriter
from fpsgen.voxel_point.dataset import paired_frames,development_split,load_xyz


def overlap(a,b):
    intersection=len(np.intersect1d(a,b,assume_unique=True))
    return {'iou':intersection/max(len(a)+len(b)-intersection,1),'raw_coverage':intersection/max(len(a),1),
            'comparison_support':len(b),'raw_support':len(a)}


def ids_for(points,g):
    coord,_=g.voxelize_points(points)
    return np.unique(g.flatten(coord,fine=True))


def physical_ids(xyz,resolution=.15):
    coord=np.floor(xyz/resolution).astype(np.int64)
    return np.unique((coord[:,0]+400)+1000*((coord[:,1]+400)+1000*(coord[:,2]+100)))


def stats(xyz,poisson,g,rng):
    start=time.perf_counter(); ids,masks,audit=voxelize_surface(xyz,g)
    occ=unpack_children(masks,g); child_counts=occ.sum(1)
    fine=occupied_fine_coords(ids,masks,g); fineids=np.unique(g.flatten(fine,fine=True))
    values,counts=np.unique(masks,return_counts=True); top=np.argsort(-counts)[:20]
    valid_children=0; boundary_parents=0
    for j in range(0,len(ids),32768):
        mask=g.valid_children(ids[j:j+32768].astype(np.int64))
        valid_children+=int(mask.sum()); boundary_parents+=int((~mask.all(1)).sum())
    point_indices,valid=g.voxelize_points(xyz); error=np.linalg.norm(g.centers(point_indices,fine=True)-xyz[valid],axis=1)
    center=g.centers(fine,fine=True); radius=np.minimum(np.linalg.norm(center,axis=1),np.nextafter(50.,0.))
    distance={f'{a}-{b}':int(((radius>=a)&(radius<b)).sum()) for a,b in ((0,10),(10,20),(20,35),(35,50))}
    raw_physical=physical_ids(xyz[valid]); represented_physical=physical_ids(center)
    thinning={}
    for retain in (.9,.5):
        keep=rng.random(len(xyz))<retain
        thinning[str(retain)]=overlap(fineids,ids_for(xyz[keep],g))
    audit.update(geometry=g.to_dict(),geometry_sha256=g.sha256,parent_voxel_size=g.parent_size,fine_voxel_size=g.fine_size,
        rectangular_fine_voxels=int(np.prod(g.fine_resolution)),occupancy_sparsity_rectangular=len(fine)/np.prod(g.fine_resolution),
        packed_bytes=int(ids.nbytes+masks.nbytes),occupied_children_histogram=np.bincount(child_counts,minlength=g.k+1).tolist(),
        pattern_unique_count=len(values),top_patterns=[{'bits':int(values[i]),'parents':int(counts[i])} for i in top],
        active_parent_valid_children=valid_children,boundary_parents=boundary_parents,boundary_parent_fraction=boundary_parents/max(len(ids),1),
        occupied_distance_bins=distance,distance_policy='fine center radius clipped to <50 for sphere-boundary-intersecting cells',
        quantization_error_m={f'p{q}':float(np.percentile(error,q)) for q in (50,90,99)},
        raw_vs_poisson=overlap(fineids,ids_for(poisson,g)),raw_thinning_stability=thinning,
        common_015m_center_representation=overlap(raw_physical,represented_physical),
        seconds=time.perf_counter()-start)
    return ids,masks,audit


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/data-12/M2024-HWZ/KITTI_Odometry'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequences',default='00,05,09,10')
    p.add_argument('--frames',default='0,200,500,550')
    p.add_argument('--reference-check',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    splits=development_split(paired_frames(args.data_root))
    atomic_json(args.output/'split_manifest.json',splits)
    manifest_sha=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
    lookup={(f['sequence'],f['frame']):split for split in ('train','development') for f in splits[split]}
    candidates={f'{n}_{k}':SparseHierarchyGeometry(parent_resolution=(n,n,32),child_split=split) for n,k,split in
        ((256,18,(3,3,2)),(320,8,(2,2,2)),(352,8,(2,2,2)),(384,8,(2,2,2)))}
    source=RawMapSource(args.data_root); records=[]; writers={}; reference=None
    for seq in args.sequences.split(','):
        if seq=='08': raise ValueError('No seq08 in representation selection')
        for frame in map(int,args.frames.split(',')):
            if (seq,frame) not in lookup: continue
            begin=time.perf_counter(); xyz=source.load(seq,frame)
            if args.reference_check and reference is None:
                full=source.load(seq,frame,reference_full_scan=True)
                reference={'frame':f'{seq}_{frame:06d}','exact_array_equal':bool(np.array_equal(xyz,full)),
                    'indexed_points':len(xyz),'reference_points':len(full)}
                if not reference['exact_array_equal']: raise RuntimeError('Indexed candidate differs from official _full_candidate')
            raw_seconds=time.perf_counter()-begin
            poisson=load_xyz(args.data_root/seq/'gt_possion'/f'{frame:06d}.ply').numpy()
            split=lookup[(seq,frame)]
            for name,g in candidates.items():
                ids,masks,report=stats(xyz,poisson,g,np.random.default_rng(20261016+frame))
                key=(name,split)
                if key not in writers:
                    writers[key]=SparseShardWriter(args.output/name/split,g,manifest_sha,shard_frames=4)
                writers[key].add(seq,frame,ids,masks)
                baseline=args.output/name/'npz'; baseline.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(baseline/f'{seq}_{frame:06d}.npz',parent_ids=ids,child_occ_mask=masks)
                report.update(sequence=seq,frame=frame,split=split,raw_candidate_seconds=raw_seconds,source='pre-Poisson map candidate')
                records.append(report)
            atomic_json(args.output/'audit_progress.json',{'records':records,'reference':reference})
            print(f'{seq}/{frame:06d} raw={len(xyz):,} seconds={raw_seconds:.2f}',flush=True)
    for writer in writers.values(): writer.flush()
    summary={}
    for name,g in candidates.items():
        subset=[r for r in records if r['geometry_sha256']==g.sha256]
        if not subset: continue
        tokens=np.asarray([r['active_parents'] for r in subset])
        summary[name]={'sample_frames':len(subset),'geometry':g.to_dict(),'geometry_sha256':g.sha256,
          'sample_token_quantiles':{f'p{q}':float(np.percentile(tokens,q)) for q in (50,90,99,100)},
          'mean_packed_bytes':float(np.mean([r['packed_bytes'] for r in subset])),
          'estimated_train_label_bytes':float(np.mean([r['packed_bytes'] for r in subset])*len(splits['train'])),
          'raw_poisson_native_iou_mean':float(np.mean([r['raw_vs_poisson']['iou'] for r in subset])),
          'raw_90pct_thinning_coverage_mean':float(np.mean([r['raw_thinning_stability']['0.9']['raw_coverage'] for r in subset])),
          'common_015m_representation_iou_mean':float(np.mean([r['common_015m_center_representation']['iou'] for r in subset])),
          'mean_point_quantization_p90_m':float(np.mean([r['quantization_error_m']['p90'] for r in subset]))}
    atomic_json(args.output/'resolution_audit.json',{'status':'CPU_DATA_AUDIT_ONLY','source_manifest_sha256':manifest_sha,
        'split_counts':{s:len(splits[s]) for s in ('train','development','excluded_guard')},'reference':reference,
        'data_definition':'Surface-hit occupancy of FPSGen pre-Poisson local map candidate; empty is not proven free space',
        'sample_limitations':'Stratified initial cohort, NOT population P99/max; no model quality conclusion',
        'summary':summary,'records':records})
    from fpsgen.hier_occ_flow.provenance import bind_resolution_audit
    bind_resolution_audit(args.output,args.data_root)

if __name__=='__main__': main()

"""Per-sequence resumable CPU workers, then immutable hardlinked shard assembly.

Uses already completed full-train raw source hashing. Every worker rechecks its
map/pose/calibration and each scan before use. No Poisson point file is opened.
"""
from common import ROOT
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.hier_occ_flow.storage import SparseShardWriter,SparseShardDataset,atomic_json,file_sha,_atomic_array,INDEX_DTYPE,DATA_VERSION
from fpsgen.hier_occ_flow.voxelization import voxelize_surface


def process_sequence(job):
    seq,part,frames,root,data_root,source,sha=job;root=Path(root);g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    for split in ('train','development'):
        snapshot=source['train_source']['raw_sources'] if split=='train' else source['development_source']
        if seq not in snapshot['map_pose_calibration']:continue
        for name,entry in snapshot['map_pose_calibration'][seq].items():
            if file_sha(Path(data_root)/seq/name)!=entry['sha256']:raise ValueError('Raw map/pose/calibration changed')
    writers={s:SparseShardWriter(root/'parts'/part/s,g,sha,resume=True,shard_frames=32) for s in ('train','development')}
    raw=RawMapSource(data_root,fast_viewpoint=True);begin=time.perf_counter();num=0
    for f,split in frames:
        writer=writers[split]
        if (seq,f) in writer.done:continue
        snapshot=source['train_source']['raw_sources'] if split=='train' else source['development_source']
        if file_sha(Path(data_root)/seq/'velodyne'/f'{f:06d}.bin')!=snapshot['viewpoint_scan_sha256'][f'{seq}_{f:06d}']:
            raise ValueError('Raw viewpoint scan changed')
        tick=time.perf_counter();xyz=raw.load(seq,f);ids,masks,audit=voxelize_surface(xyz,g)
        # Independent direct-parent rasterization, not merely reusing the child OR.
        _,keep=g.voxelize_points(xyz)
        parent=np.floor((xyz[keep]-np.asarray(g.origin))/np.asarray(g.parent_size)).astype(np.int64)
        if not np.array_equal(ids,np.unique(g.flatten(parent))):raise ValueError('Direct parent != fine-child OR')
        writer.add(seq,f,ids,masks);num+=1
        audit.update(sequence=seq,frame=f,split=split,parent_or_exact=True,seconds=time.perf_counter()-tick)
        with (root/'parts'/part/'frame_audit.jsonl').open('a') as log:log.write(json.dumps(audit)+'\n')
        atomic_json(root/'parts'/part/'progress.json',{'status':'RUNNING','sequence':seq,'part':part,'new_frames':num,
            'done_frames':sum(len(w.done) for w in writers.values()),'requested_frames':len(frames),'seconds':time.perf_counter()-begin,
            'committed_frames':sum(w.meta['records'] for w in writers.values())})
    for w in writers.values():w.flush()
    atomic_json(root/'parts'/part/'progress.json',{'status':'COMPLETE','sequence':seq,'part':part,'done_frames':len(frames),
        'requested_frames':len(frames),'seconds':time.perf_counter()-begin,'committed_frames':len(frames)})
    return {'sequence':seq,'part':part,'frames':len(frames),'seconds':time.perf_counter()-begin}


def assemble(root,split,g,source_sha,expected):
    dest=root/split
    if (dest/'manifest.json').exists():
        ds=SparseShardDataset(dest,g,verify=True)
        if len(ds)!=expected or ds.meta['source_manifest_sha256']!=source_sha:
            raise ValueError('Existing assembled dataset is incomplete or from another source version')
        return
    dest.mkdir(exist_ok=True);shards=[];indices=[]
    for part in sorted((root/'parts').glob('*/'+split)):
        if not (part/'manifest.json').exists():continue
        ds=SparseShardDataset(part,g,verify=True);offset=len(shards);index=ds.index.copy();index['shard']+=offset;indices.append(index)
        for shard in ds.meta['shards']:
            row=dict(shard);number=len(shards)
            for kind in ('ids','masks'):
                target=f'shard_{number:04d}.{kind}.bin'
                if not (dest/target).exists():os.link(part/shard[kind],dest/target)
                if file_sha(dest/target)!=row[kind+'_sha256']:raise ValueError('Assembled shard checksum mismatch')
                row[kind]=target
            shards.append(row)
    index=np.concatenate(indices).astype(INDEX_DTYPE)
    if len(index)!=expected:raise ValueError(f'{split}: {len(index)}/{expected} records')
    _atomic_array(dest/'index.npy',index)
    atomic_json(dest/'manifest.json',{'data_version':DATA_VERSION,'geometry':g.to_dict(),'geometry_sha256':g.sha256,
        'source_manifest_sha256':source_sha,'mask_dtype':g.mask_dtype.str,'shards':shards,'records':len(index),
        'index':'index.npy','index_sha256':file_sha(dest/'index.npy')})
    SparseShardDataset(dest,g,verify=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--train-source',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--lookup-audit',type=Path,default=Path('experiments/hier_occ_flow/viewpoint_lookup_audit.json'))
    p.add_argument('--data-root',type=Path,default=Path('/data-12/M2024-HWZ/KITTI_Odometry'));args=p.parse_args()
    if not 1<=args.workers<=8:raise ValueError('CPU worker budget1–8')
    lookup=json.loads(args.lookup_audit.read_text())
    if lookup.get('status')!='PASS' or len(lookup.get('rows',[]))<4 or lookup['lookup_code_sha256']!=file_sha(ROOT/'fpsgen/hier_occ_flow/viewpoint_lookup.py'):
        raise ValueError('Exact real-frame lookup equivalence must pass before acceleration')
    splits=json.loads(args.manifest.read_text());root=args.output;root.mkdir(parents=True,exist_ok=True)
    source_path=root/'source_manifest.json'
    if source_path.exists():source=json.loads(source_path.read_text())
    else:
        print('Hash development sources; reuse verified full-train source snapshot',flush=True)
        dev,_=raw_source_snapshot(args.data_root,[(r['sequence'],r['frame']) for r in splits['development']])
        source={'train_source':json.loads(args.train_source.read_text()),'development_source':dev,'split_sha256':file_sha(args.manifest)}
        atomic_json(source_path,source)
    if source['split_sha256']!=file_sha(args.manifest) or source['train_source']!=json.loads(args.train_source.read_text()):raise ValueError('Source snapshot changed')
    source_sha=file_sha(source_path);jobs={};keys=[]
    with (root/'implementation_history.jsonl').open('a') as log:
        log.write(json.dumps({'epoch_time':time.time(),'source_manifest_sha256':source_sha,
            'lookup_equivalence_audit_sha256':file_sha(args.lookup_audit),
            'preparation_code_sha256':file_sha(Path(__file__)),'lookup_code_sha256':lookup['lookup_code_sha256']})+'\n')
    for split,expected in (('train',17870),('development',900)):
        if len(splits[split])!=expected:raise ValueError('Wrong frozen split')
        keys.append({(r['sequence'],r['frame']) for r in splits[split]})
        for row in splits[split]:
            if row['sequence']=='08':raise ValueError('seq08 leakage')
            jobs.setdefault(row['sequence'],[]).append((row['frame'],split))
    if keys[0]&keys[1]:raise ValueError('Train/development overlap')
    (root/'split_manifest.json').write_bytes(args.manifest.read_bytes())
    atomic_json(root/'preprocessing_progress.json',{'status':'RUNNING','workers':args.workers,'total_frames':18770,'source_manifest_sha256':source_sha})
    payload=[]
    for seq,frames in jobs.items():
        frames=sorted(frames)
        for i in range(0,len(frames),1024):
            payload.append((seq,f'{seq}_chunk{i//1024:02d}',frames[i:i+1024],str(root),str(args.data_root),source,source_sha))
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for result in pool.imap_unordered(process_sequence,payload):print(result,flush=True)
    g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    for split,expected in (('train',17870),('development',900)):assemble(root,split,g,source_sha,expected)
    atomic_json(root/'preprocessing_progress.json',{'status':'COMPLETE','train_frames':17870,'development_frames':900,'source_manifest_sha256':source_sha})

if __name__=='__main__':main()

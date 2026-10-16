"""Pre-Poisson sparse labels, explicit split, versioned resumable shard commits."""
from common import geometry
import argparse
from pathlib import Path
import hashlib
import json
import signal
import time
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.voxelization import voxelize_surface
from fpsgen.hier_occ_flow.storage import SparseShardWriter,atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--geometry',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/data-12/M2024-HWZ/KITTI_Odometry'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--split',choices=('train','development','test'),required=True)
    p.add_argument('--max-frames',type=int,default=16); p.add_argument('--resume',action='store_true')
    p.add_argument('--full',action='store_true'); p.add_argument('--geometry-decision',type=Path)
    p.add_argument('--shard-frames',type=int,default=256)
    args=p.parse_args(); g=geometry(args.geometry); split=json.loads(args.manifest.read_text())
    if args.split=='test':
        decision=json.loads(args.geometry_decision.read_text()) if args.geometry_decision else {}
        if not decision.get('geometry_locked') or not decision.get('method_checkpoint_frozen') or decision.get('geometry_sha256')!=g.sha256:
            raise ValueError('Test preprocessing requires frozen geometry and method/checkpoint selection')
    frames=split[args.split]
    if args.split!='test' and any(f['sequence']=='08' for f in frames): raise ValueError('seq08 leakage')
    if args.split=='test' and any(f['sequence']!='08' for f in frames): raise ValueError('Test manifest must use seq08 only')
    if args.full:
        decision=json.loads(args.geometry_decision.read_text()) if args.geometry_decision else {}
        if not decision.get('geometry_locked') or decision.get('geometry_sha256')!=g.sha256:
            raise ValueError('Full preprocessing requires a locked and matching resolution decision')
    else: frames=frames[:args.max_frames]
    from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
    snapshot,_=raw_source_snapshot(args.data_root,[(f['sequence'],f['frame']) for f in frames])
    source={'paired_split_sha256':file_sha(args.manifest),'raw_sources':snapshot,
        'sampling':'NONE; raw pre-Poisson map surface hits','selected_frames':[(f['sequence'],f['frame']) for f in frames]}
    sha=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    writer=SparseShardWriter(args.output/args.split,g,sha,resume=args.resume,shard_frames=args.shard_frames)
    atomic_json(args.output/args.split/'source_manifest.json',source)
    stop=[False]
    def request_stop(signum,frame): stop[0]=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    begin=time.perf_counter(); new_frames=0
    raw=RawMapSource(args.data_root)
    for f in frames:
        if (f['sequence'],f['frame']) in writer.done: continue
        xyz=raw.load(f['sequence'],f['frame']); ids,masks,audit=voxelize_surface(xyz,g)
        writer.add(f['sequence'],f['frame'],ids,masks)
        new_frames+=1
        row={'sequence':f['sequence'],'frame':f['frame'],'audit':audit}
        with (args.output/args.split/'frame_audit.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
        progress={'status':'RUNNING','frames_voxelized':len(writer.done),'committed_frames':writer.meta['records'],
            'requested_frames':len(frames),'new_frames':new_frames,'seconds':time.perf_counter()-begin,
            'source_manifest_sha256':sha,'geometry_sha256':g.sha256}
        atomic_json(args.output/args.split/'preprocessing_progress.json',progress)
        print(f"{args.split} {len(writer.done)}/{len(frames)} {f['sequence']}/{f['frame']:06d}: {audit}",flush=True)
        if stop[0]: break
    writer.flush()
    progress={'status':'COMPLETE' if len(writer.done)==len(frames) else 'PAUSED','committed_frames':writer.meta['records'],
        'requested_frames':len(frames),'seconds':time.perf_counter()-begin,'geometry_sha256':g.sha256,
        'source_manifest_sha256':sha,'raw_surface_hit':True}
    atomic_json(args.output/args.split/'preprocessing_progress.json',progress)
    if stop[0]: raise SystemExit(130)

if __name__=='__main__': main()

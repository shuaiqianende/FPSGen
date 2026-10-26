"""Materialize previously missing raw Surface-hit targets; never modify historical results."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import argparse,json,hashlib
import numpy as np
import yaml
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.hier_occ_flow.voxelization import voxelize_surface
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());g=SparseHierarchyGeometry(**cfg['geometry'])
    ref=Path(cfg['validation']['seq08_reference']);protocol=json.loads((ref/'protocol.json').read_text())
    frames=protocol['frames']
    if [(r['sequence'],r['frame']) for r in frames]!=[('08',f) for f in range(0,4001,100)]:raise ValueError('Not frozen41')
    if protocol['geometry_sha256']!=g.sha256:raise ValueError('Geometry mismatch')
    before={n:file_sha(ref/n) for n in ('protocol.json','raw_sources.json','results.json','per_frame.jsonl')}
    if (ref/'targets').exists() or (ref/'targets').is_symlink():raise FileExistsError('Never replace reference targets')
    a.output.mkdir(parents=True,exist_ok=False)
    snapshot,sha=raw_source_snapshot(cfg['data_root'],[('08',r['frame']) for r in frames])
    if sha!=protocol['raw_source_manifest_sha256']:raise ValueError('Historical raw sources do not match')
    old={int(r['frame']):int(r['raw_gt_points']) for r in map(json.loads,(ref/'per_frame.jsonl').read_text().splitlines())}
    atomic_json(a.output/'raw_sources.json',snapshot)
    raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);rows=[]
    target=a.output/'targets';target.mkdir()
    for r in frames:
        f=r['frame'];xyz=raw.load('08',f);_,valid=g.voxelize_points(xyz)
        if int(valid.sum())!=old[f]:raise ValueError(f'Historical raw GT count mismatch: {f}')
        ids,masks,_=voxelize_surface(xyz,g)
        npz=target/f'08_{f:06d}.npz';np.savez_compressed(npz,parent_ids=ids,child_masks=masks)
        meta=dict(sequence='08',frame=f,input_sha256=file_sha(r['input']),raw_gt_points=int(valid.sum()),
            target_sha256=file_sha(npz),raw_source_manifest_sha256=sha,geometry_sha256=g.sha256,
            provenance='New target cache from original hash-verified raw source; not a historical target file')
        atomic_json(target/f'08_{f:06d}.json',meta);rows.append(meta)
        atomic_json(a.output/'progress.json',dict(status='PREPARING',frames=len(rows),requested=41));print(f'target snapshot {len(rows)}/41',flush=True)
    if before!={n:file_sha(ref/n) for n in before}:raise ValueError('Historical reference files changed')
    atomic_json(a.output/'audit.json',dict(status='VERIFIED41',frames=rows,historical_reference_files_unchanged=before,
        source_manifest_sha256=sha,geometry_sha256=g.sha256,reference=str(ref)))
    # Add only the previously absent cache link. Existing reference files remain byte-identical.
    (ref/'targets').symlink_to(target.resolve(),target_is_directory=True)
    atomic_json(a.output/'progress.json',dict(status='VERIFIED41',frames=41,requested=41))

if __name__=='__main__':main()

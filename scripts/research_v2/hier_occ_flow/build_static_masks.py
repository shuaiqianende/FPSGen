"""Build resumable compact GT-support/observed masks, offline; no GPU features."""
import sys,time,hashlib,json,os,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,file_sha,atomic_json
from fpsgen.hier_occ_flow.morphology import MORPH_DATA_VERSION
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.condition import observed_children
from fpsgen.hier_occ_flow.voxelization import pack_children
DATA=Path('/data-12/M2024-HWZ/KITTI_Odometry');DATASET=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-t1-stage2-fulltrain-v1/experiments/desescene_t1_stage2_fulltrain_v1/data/train');OUT=ROOT/'experiments/stage2_static_masks_full_v1';g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2));DS=None;MANIFEST=None

def init():
 global DS,MANIFEST
 DS=SparseShardDataset(DATASET,g,data_version=MORPH_DATA_VERSION);MANIFEST=file_sha(DATASET/'manifest.json')

def worker(i):
 r=DS[i];seq=r['sequence'];frame=r['frame'];ids=r['parent_ids'];path=DATA/seq/'input_'/f'{frame:06d}.npy';dest=OUT/'frames'/seq/f'{frame:06d}.npz'
 if seq=='08' or len(ids)==0 or np.any(ids[1:]<=ids[:-1]):raise ValueError('Invalid training frame/parent order')
 digest=file_sha(path);parent_sha=hashlib.sha256(ids.tobytes()+r['child_occ_mask'].tobytes()).hexdigest()
 if dest.exists():
  with np.load(dest,allow_pickle=False) as z:
   if (str(z['manifest_sha'])!=MANIFEST or str(z['input_sha'])!=digest or str(z['parent_sha'])!=parent_sha or len(z['ids'])!=len(ids)):raise ValueError(f'Stale cache {dest}')
  return (seq,frame,dest.stat().st_size,True)
 xyz=input_xyz(path);obs=pack_children(observed_children(xyz,ids,g),g);valid=pack_children(g.valid_children(ids.astype(np.int64)),g)
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(f'.{os.getpid()}.tmp')
 with tmp.open('xb') as f:
  np.savez_compressed(f,ids=ids.astype('<u4'),observed=obs,valid=valid,target=r['child_occ_mask'],manifest_sha=MANIFEST,input_sha=digest,parent_sha=parent_sha);f.flush();os.fsync(f.fileno())
 os.replace(tmp,dest);return (seq,frame,dest.stat().st_size,False)

def main():
 OUT.mkdir(parents=True,exist_ok=True);status=OUT/'status.json';manifest=file_sha(DATASET/'manifest.json');ds=SparseShardDataset(DATASET,g,data_version=MORPH_DATA_VERSION)
 if len(ds)!=19130 or {x['sequence'].decode() for x in ds.index}!={f'{i:02}' for i in (0,1,2,3,4,5,6,7,9,10)}:raise ValueError('Full 19130 training set required')
 old=json.loads(status.read_text()) if status.exists() else None
 if old and old.get('source_manifest_sha256')!=manifest:raise ValueError('Cache source manifest mismatch')
 start=time.time();done=total=reused=0
 try:
  atomic_json(status,dict(status='BUILDING',frames=0,total=len(ds),source_manifest_sha256=manifest))
  workers=int(os.environ.get('STATIC_MASK_WORKERS','4'))
  if not 1<=workers<=8:raise ValueError('STATIC_MASK_WORKERS must be 1-8')
  with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
   for seq,frame,size,cached in pool.map(worker,range(len(ds)),chunksize=8):
    done+=1;total+=size;reused+=int(cached)
    if done%25==0:atomic_json(status,dict(status='BUILDING',frames=done,total=len(ds),bytes=total,reused=reused,source_manifest_sha256=manifest,elapsed_s=time.time()-start))
  atomic_json(status,dict(status='COMPLETE',frames=done,total=len(ds),bytes=total,reused=reused,source_manifest_sha256=manifest,elapsed_s=time.time()-start,geometry_sha256=g.sha256))
 except Exception as e:
  atomic_json(status,dict(status='BLOCKED',frames=done,total=len(ds),source_manifest_sha256=manifest,reason=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()

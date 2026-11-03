"""Compare original GT-only CPU mask preparation vs per-frame packed-mask cache."""
import sys,time,json,hashlib,statistics,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,file_sha,atomic_json
from fpsgen.hier_occ_flow.morphology import MORPH_DATA_VERSION
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.condition import observed_children
from fpsgen.hier_occ_flow.voxelization import pack_children,unpack_children,lookup_masks
from fpsgen.hier_occ_flow.stage2_candidates import oracle_parent_candidates
DATA=Path('/data-12/M2024-HWZ/KITTI_Odometry');DATASET=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-t1-stage2-fulltrain-v1/experiments/desescene_t1_stage2_fulltrain_v1/data/train');OUT=ROOT/'experiments/stage2_static_mask_prototype_v1';g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
def original(record,xyz):
 ids,_,_=oracle_parent_candidates(record['parent_ids'],g,seed=1,negative_fraction=0.,neutral_score=1.)
 obs=observed_children(xyz,ids,g);valid=g.valid_children(ids);target=unpack_children(lookup_masks(record['parent_ids'],record['child_occ_mask'],ids),g)
 return ids,obs,valid,target

def main():
 OUT.mkdir(parents=True,exist_ok=False);ds=SparseShardDataset(DATASET,g,data_version=MORPH_DATA_VERSION);rows=[]
 for i in np.linspace(0,len(ds)-1,16,dtype=int):
  record=ds[int(i)];path=DATA/record['sequence']/'input_'/f"{record['frame']:06d}.npy";xyz=input_xyz(path)
  t=time.perf_counter();ids,obs,valid,target=original(record,xyz);orig=time.perf_counter()-t
  t=time.perf_counter();obs_bits=pack_children(obs,g);valid_bits=pack_children(valid,g);target_bits=pack_children(target,g)
  f=OUT/f"{record['sequence']}_{record['frame']:06d}.npz"
  with f.open('xb') as h:np.savez_compressed(h,ids=ids.astype('<u4'),observed=obs_bits,valid=valid_bits,target=target_bits)
  build=time.perf_counter()-t
  measurements=[]
  for _ in range(3):
   t=time.perf_counter()
   with np.load(f,allow_pickle=False) as z:ids2=z['ids'].astype(np.int64);obs2=unpack_children(z['observed'],g);valid2=unpack_children(z['valid'],g);target2=unpack_children(z['target'],g)
   measurements.append(time.perf_counter()-t)
  if not all(np.array_equal(a,b) for a,b in ((ids,ids2),(obs,obs2),(valid,valid2),(target,target2))):raise ValueError('Mismatch')
  rows.append(dict(sequence=record['sequence'],frame=record['frame'],parents=len(ids),input_sha256=file_sha(path),original_seconds=orig,write_seconds=build,read_seconds=statistics.median(measurements),bytes=f.stat().st_size));print(rows[-1],flush=True)
 atomic_json(OUT/'result.json',dict(status='PASS',scope='CPU static mask processing only; excludes input read, GPU transfer, Encoder, PTv3 and full training',manifest_sha256=file_sha(DATASET/'manifest.json'),rows=rows,original_mean_seconds=statistics.mean(r['original_seconds'] for r in rows),cache_read_mean_seconds=statistics.mean(r['read_seconds'] for r in rows),average_file_bytes=statistics.mean(r['bytes'] for r in rows)))
if __name__=='__main__':main()

"""Add missing guard frames and assemble 00-07/09/10 closed targets without rewriting prior datasets."""
import sys,json,hashlib,os,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,SparseShardWriter,atomic_json,file_sha,_atomic_array
from fpsgen.hier_occ_flow.morphology import FineClosingSpec,MORPH_DATA_VERSION
from fpsgen.hier_occ_flow.morphology_storage import derive_dataset,processor_sha
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.hier_occ_flow.voxelization import voxelize_surface

DATA=Path('/data-12/M2024-HWZ/KITTI_Odometry');OLD=Path('/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/occ384_full_parallel');MORPH=Path('/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/occ384_morph_close26_training_v1');OUT=ROOT/'experiments/desescene_t1_stage2_fulltrain_v1/data';G=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2));SEQS=tuple(f'{i:02}' for i in (0,1,2,3,4,5,6,7,9,10))

def keys(ds):return {(row['sequence'].decode(),int(row['frame'])) for row in ds.index}
def main():
 OUT.mkdir(parents=True,exist_ok=True);state=OUT/'status.json'
 try:
  atomic_json(state,dict(status='AUDITING',time=time.time()))
  raw=[SparseShardDataset(OLD/s,G,verify=True) for s in ('train','development')]
  closed=[SparseShardDataset(MORPH/s,G,verify=True,data_version=MORPH_DATA_VERSION) for s in ('train','development')]
  for a,b in zip(raw,closed):
   if keys(a)!=keys(b):raise ValueError('Raw/closed frame identities differ')
   if b.meta.get('target_transform_sha256')!=FineClosingSpec(connectivity=26).sha256 or b.meta.get('target_processor_sha256')!=processor_sha():raise ValueError('Closing transform/implementation mismatch')
  observed=set()
  for seq in SEQS:
   for p in (DATA/seq/'input_').glob('*.npy'):
    if p.stem.isdigit():observed.add((seq,int(p.stem)))
  original=keys(raw[0])|keys(raw[1]);missing=sorted(observed-original)
  if len(original)!=18770 or len(missing)!=360 or len(observed)!=19130 or original-observed:raise ValueError('Unexpected source coverage: audit required')
  for seq,frame in observed:
   if not (DATA/seq/'velodyne'/f'{frame:06d}.bin').exists():raise FileNotFoundError(f'missing raw scan {seq}/{frame}')
  snapshot,sha=raw_source_snapshot(DATA,missing);atomic_json(OUT/'guard_source.json',snapshot)
  atomic_json(OUT/'coverage.json',dict(sequences=SEQS,input_count=len(observed),historical_train=len(raw[0]),historical_development=len(raw[1]),new_frames=missing,raw_manifest_sha256=[file_sha(OLD/s/'manifest.json') for s in ('train','development')],closed_manifest_sha256=[file_sha(MORPH/s/'manifest.json') for s in ('train','development')],guard_snapshot_sha256=sha,seq08_included=False))
  src=OUT/'guard_raw';writer=SparseShardWriter(src,G,sha,resume=(src/'manifest.json').exists(),shard_frames=16);source=RawMapSource(DATA,fast_viewpoint=True)
  for i,(seq,frame) in enumerate(missing):
   if (seq,frame) in writer.done:continue
   xyz=source.load(seq,frame);ids,masks,audit=voxelize_surface(xyz,G)
   _,keep=G.voxelize_points(xyz);parent=np.floor((xyz[keep]-np.asarray(G.origin))/np.asarray(G.parent_size)).astype(np.int64)
   if not np.array_equal(ids,np.unique(G.flatten(parent))):raise ValueError('Parent OR mismatch')
   writer.add(seq,frame,ids,masks)
   atomic_json(state,dict(status='GENERATING_GUARD',completed=i+1,total=len(missing),committed=writer.meta['records'],time=time.time()))
  writer.flush();guard=SparseShardDataset(src,G,verify=True)
  if keys(guard)!=set(missing):raise ValueError('Guard frame identity mismatch')
  dst=OUT/'guard_closed';derive_dataset(src,dst,G,FineClosingSpec(connectivity=26),workers=4,resume=(dst/'build_state.json').exists())
  parts=closed+[SparseShardDataset(dst,G,verify=True,data_version=MORPH_DATA_VERSION)];target=OUT/'train';target.mkdir(exist_ok=True);indices=[];shards=[]
  for ds in parts:
   offset=len(shards);index=ds.index.copy();index['shard']+=offset;indices.append(index)
   for shard in ds.meta['shards']:
    row=dict(shard);number=len(shards)
    for kind in ('ids','masks'):
     srcfile=ds.root/row[kind];name=f'shard_{number:04d}.{kind}.bin';dest=target/name
     if not dest.exists():os.link(srcfile,dest)
     if os.stat(srcfile).st_ino!=os.stat(dest).st_ino or os.stat(srcfile).st_dev!=os.stat(dest).st_dev:raise ValueError('Existing file is not expected hardlink')
     row[kind]=name
    shards.append(row)
  index=np.concatenate(indices);order=np.lexsort((index['frame'],index['sequence']));index=index[order]
  if len(index)!=19130 or {(x['sequence'].decode(),int(x['frame'])) for x in index}!=observed:raise ValueError('Full training coverage mismatch')
  _atomic_array(target/'index.npy',index)
  identity={'parts':[file_sha(ds.root/'manifest.json') for ds in parts],'guard_source_sha256':sha}
  manifest={'data_version':MORPH_DATA_VERSION,'geometry':G.to_dict(),'geometry_sha256':G.sha256,'source_manifest_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),'mask_dtype':G.mask_dtype.str,'shards':shards,'records':len(index),'index':'index.npy','index_sha256':file_sha(target/'index.npy'),'target_transform_sha256':FineClosingSpec(connectivity=26).sha256,'target_processor_sha256':processor_sha(),'source_parts':identity,'target_definition':'parent-preserving morphology augmented surface-hit; original GT used in evaluation','parent_ids_unchanged':True,'original_occupied_children_preserved':True}
  atomic_json(target/'manifest.json',manifest);verified=SparseShardDataset(target,G,verify=True,data_version=MORPH_DATA_VERSION)
  if keys(verified)!=observed:raise ValueError('Final validation mismatch')
  atomic_json(state,dict(status='COMPLETE',records=len(index),seq08_included=False,train_manifest_sha256=file_sha(target/'manifest.json'),time=time.time()))
 except Exception as e:
  atomic_json(state,dict(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()

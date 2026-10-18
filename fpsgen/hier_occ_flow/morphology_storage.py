"""Resumable derived masks; immutable parent IDs/index are hard-linked unchanged."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
from .geometry import SparseHierarchyGeometry
from .storage import SparseShardDataset,atomic_json,file_sha
from .morphology import FineClosingSpec,close_fine_masks,MORPH_DATA_VERSION

_DS=None;_G=None;_SPEC=None


def processor_sha():
    here=Path(__file__).resolve()
    return hashlib.sha256(here.read_bytes()+here.with_name('morphology.py').read_bytes()).hexdigest()


def initialize_worker(source,geometry,spec):
    global _DS,_G,_SPEC
    _G=SparseHierarchyGeometry(**geometry);_SPEC=FineClosingSpec(**spec);_DS=SparseShardDataset(source,_G)


def transform_record(index):
    record=_DS[index]
    masks,stats=close_fine_masks(record['parent_ids'],record['child_occ_mask'],_G,_SPEC)
    return index,masks,stats


def bounded_results(pool,indices,limit):
    todo=iter(indices);pending=deque()
    for _ in range(limit):
        try:index=next(todo)
        except StopIteration:break
        pending.append(pool.submit(transform_record,index))
    while pending:
        yield pending.popleft().result()
        try:index=next(todo)
        except StopIteration:continue
        pending.append(pool.submit(transform_record,index))


def link_or_copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if destination.stat().st_size!=source.stat().st_size:raise ValueError('Existing ID/index length mismatch')
        return
    temp=destination.with_suffix(destination.suffix+'.linktmp')
    if temp.exists():temp.unlink()
    try:os.link(source,temp)
    except OSError:shutil.copyfile(source,temp)
    os.replace(temp,destination)


def derive_dataset(source,output,geometry,spec,*,workers=4,resume=False,max_shards=None):
    if workers<1 or workers>32:raise ValueError('Register1-32 CPU workers')
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:raise ValueError('Separate derived dataset root required')
    ds=SparseShardDataset(source,geometry,verify=True);source_sha=file_sha(source/'manifest.json')
    identity={'data_version':MORPH_DATA_VERSION,'geometry_sha256':geometry.sha256,
        'source_parent_manifest_sha256':source_sha,'target_transform_sha256':spec.sha256,
        'target_processor_sha256':processor_sha(),'transform':spec.to_dict()}
    output.mkdir(parents=True,exist_ok=resume)
    state_path=output/'build_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if not resume or state['identity']!=identity:raise ValueError('Resume identity mismatch: source/geometry/transform/processor differs')
    else:
        if resume:raise ValueError('No matching build state to resume')
        state={'identity':identity,'status':'RUNNING','completed_shards':[], 'completed_frames':0,'original_fine':0,'new_fine':0,'workers':workers}
        atomic_json(state_path,state)
    manifest=deepcopy(ds.meta)
    manifest.update(identity,parent_ids_unchanged=True,original_occupied_children_preserved=True,
        source_raw_manifest_sha256=ds.meta['source_manifest_sha256'],
        source_manifest_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
        target_definition='morphology-augmented surface-hit occupancy; added cells are synthetic labels',index='index.npy')
    link_or_copy(source/ds.meta['index'],output/'index.npy')
    if file_sha(output/'index.npy')!=ds.meta['index_sha256']:raise ValueError('Index changed')
    done={v['shard'] for v in state['completed_shards']}
    for row in state['completed_shards']:
        shard=manifest['shards'][row['shard']]
        if file_sha(output/shard['masks'])!=row['mask_sha256'] or file_sha(output/shard['ids'])!=shard['ids_sha256']:
            raise ValueError('Completed derived shard corrupted')
        shard['masks_sha256']=row['mask_sha256']
    remaining=[s for s in range(len(manifest['shards'])) if s not in done]
    if max_shards is not None:remaining=remaining[:max_shards]
    needed=sum(manifest['shards'][s]['parents']*geometry.mask_dtype.itemsize for s in remaining)
    if source.stat().st_dev!=output.stat().st_dev:needed+=sum(manifest['shards'][s]['parents']*4 for s in remaining)
    if shutil.disk_usage(output).free<needed*1.1+2**28:raise RuntimeError('Insufficient space for derived sparse labels')
    state['estimated_additional_bytes']=needed
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,
            initargs=(str(source),geometry.to_dict(),spec.to_dict())) as pool:
        for sid in remaining:
            shard=manifest['shards'][sid];link_or_copy(source/shard['ids'],output/shard['ids'])
            if file_sha(output/shard['ids'])!=shard['ids_sha256']:raise ValueError('Original parent IDs changed')
            indices=np.flatnonzero(ds.index['shard']==sid).tolist()
            temp=output/(shard['masks']+'.tmp')
            temp.parent.mkdir(parents=True,exist_ok=True)
            target=np.memmap(temp,dtype=geometry.mask_dtype,mode='w+',shape=(shard['parents'],))
            count=original=new_count=offset=0
            for index,masks,stats in bounded_results(pool,indices,2*workers):
                row=ds.index[index];length=int(row['length'])
                if int(row['offset'])!=offset or len(masks)!=length:raise ValueError('Source shard ordering/length differs')
                if not stats['original_hits_preserved'] or not stats['parent_ids_unchanged'] or stats['invalid_children_added']:
                    raise ValueError('Target augmentation violated geometry/preservation contract')
                target[offset:offset+length]=masks;offset+=length;count+=1
                original+=stats['original_fine'];new_count+=stats['new_fine']
            if offset!=shard['parents']:raise ValueError('Incomplete output shard')
            target.flush();del target
            with temp.open('rb') as handle:os.fsync(handle.fileno())
            os.replace(temp,output/shard['masks']);digest=file_sha(output/shard['masks'])
            shard['masks_sha256']=digest
            state['completed_shards'].append({'shard':sid,'frames':count,'mask_sha256':digest})
            state['completed_frames']+=count;state['original_fine']+=original;state['new_fine']+=new_count
            state['seconds_this_invocation']=time.perf_counter()-begin
            atomic_json(state_path,state)
            print(f'{output.name}: {state["completed_frames"]}/{len(ds)} frames; shard={sid}; added={(state["new_fine"]/state["original_fine"]-1):.2%}',flush=True)
    if len(state['completed_shards'])==len(manifest['shards']):
        atomic_json(output/'manifest.json',manifest)
        checked=SparseShardDataset(output,geometry,verify=True,data_version=MORPH_DATA_VERSION)
        if len(checked)!=len(ds):raise ValueError('Final frame count differs')
        state.update(status='COMPLETE',manifest_sha256=file_sha(output/'manifest.json'))
    else:state['status']='PARTIAL_RESUMABLE'
    atomic_json(state_path,state)
    return state

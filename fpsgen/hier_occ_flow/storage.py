"""Transactional per-shard binary storage, sparse uint IDs and packed masks."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

DATA_VERSION='hs-occ-surface-hit-v1'
INDEX_DTYPE=np.dtype([('sequence','S2'),('frame','<i4'),('shard','<i4'),('offset','<i8'),('length','<i4')])


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def _atomic_array(path,array,raw=False):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        if raw: array.tofile(f)
        else: np.save(f,array,allow_pickle=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


class SparseShardWriter:
    def __init__(self,root,geometry,source_manifest_sha,*,resume=False,shard_frames=256):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.geometry=geometry; self.shard_frames=shard_frames; self.pending=[]
        self.meta={'data_version':DATA_VERSION,'geometry':geometry.to_dict(),'geometry_sha256':geometry.sha256,
            'source_manifest_sha256':source_manifest_sha,'mask_dtype':geometry.mask_dtype.str,'shards':[],'records':0}
        path=self.root/'manifest.json'; self.index=np.empty(0,dtype=INDEX_DTYPE)
        if path.exists():
            old=json.loads(path.read_text())
            if not resume: raise FileExistsError('Use explicit --resume')
            for key in ('data_version','geometry_sha256','source_manifest_sha256','mask_dtype'):
                if old.get(key)!=self.meta[key]: raise ValueError(f'Dataset version mismatch: {key}')
            self.meta=old; self.index=np.load(self.root/old['index'],allow_pickle=False)
            SparseShardDataset(self.root,geometry,verify=True)
        self.done={(x['sequence'].decode(),int(x['frame'])) for x in self.index}

    def add(self,sequence,frame,ids,masks):
        key=(str(sequence),int(frame))
        if key in self.done: return False
        ids=np.asarray(ids); masks=np.asarray(masks)
        if ids.dtype.kind not in 'iu' or masks.dtype.kind not in 'iu' or (ids<0).any() or (ids>=self.geometry.num_parents).any() or (masks<0).any() or (masks.astype(np.uint64)>=2**self.geometry.k).any():
            raise ValueError('Invalid IDs or mask bits before uint cast')
        ids=ids.astype('<u4'); masks=masks.astype(self.geometry.mask_dtype)
        if len(ids)!=len(masks) or not len(ids) or np.any(ids[1:]<=ids[:-1]) or np.any(ids>=self.geometry.num_parents) or np.any(masks==0) or np.any(masks.astype(np.uint64)>=2**self.geometry.k):
            raise ValueError('Invalid packed sparse label')
        self.pending.append((key,ids,masks)); self.done.add(key)
        if len(self.pending)>=self.shard_frames: self.flush()
        return True

    def flush(self):
        if not self.pending: return
        sid=len(self.meta['shards']); prefix=f'shard_{sid:04d}'
        ids=np.concatenate([p[1] for p in self.pending]); masks=np.concatenate([p[2] for p in self.pending])
        rows=[]; offset=0
        for (seq,frame),i,m in self.pending:
            rows.append((seq.encode(),frame,sid,offset,len(i))); offset+=len(i)
        for suffix,array in (('ids',ids),('masks',masks)):
            _atomic_array(self.root/f'{prefix}.{suffix}.bin',array,raw=True)
        index=np.concatenate((self.index,np.asarray(rows,dtype=INDEX_DTYPE)))
        index_name=f'index_{sid:04d}.npy'; _atomic_array(self.root/index_name,index)
        self.meta['shards'].append({'ids':f'{prefix}.ids.bin','masks':f'{prefix}.masks.bin','parents':len(ids),
            'ids_sha256':file_sha(self.root/f'{prefix}.ids.bin'),'masks_sha256':file_sha(self.root/f'{prefix}.masks.bin')})
        self.meta.update(index=index_name,index_sha256=file_sha(self.root/index_name),records=len(index))
        # Manifest is the single commit point; unreferenced temporary/final shard
        # files from interrupted writes are ignored and replaced on explicit resume.
        atomic_json(self.root/'manifest.json',self.meta)
        self.index=index; self.pending=[]


class SparseShardDataset:
    def __init__(self,root,geometry,*,verify=False,data_version=DATA_VERSION):
        self.root=Path(root); self.geometry=geometry
        self.meta=json.loads((self.root/'manifest.json').read_text())
        if self.meta['data_version']!=data_version or self.meta['geometry_sha256']!=geometry.sha256:
            raise ValueError('Dataset geometry/version mismatch')
        self.index=np.load(self.root/self.meta['index'],allow_pickle=False)
        if self.index.dtype!=INDEX_DTYPE or len(self.index)!=self.meta['records']: raise ValueError('Corrupt index')
        if verify and self.meta.get('index_sha256') and file_sha(self.root/self.meta['index'])!=self.meta['index_sha256']:
            raise ValueError('Index checksum failed')
        keys=set()
        for row in self.index:
            sid,offset,length=int(row['shard']),int(row['offset']),int(row['length'])
            if sid<0 or sid>=len(self.meta['shards']) or offset<0 or length<=0 or offset+length>self.meta['shards'][sid]['parents']:
                raise ValueError('Index points outside shard')
            key=(bytes(row['sequence']),int(row['frame']))
            if key in keys: raise ValueError('Duplicate indexed frame')
            keys.add(key)
        self.maps={}
        for shard in self.meta['shards']:
            for kind,dtype in (('ids',np.dtype('<u4')),('masks',geometry.mask_dtype)):
                path=self.root/shard[kind]
                if path.stat().st_size!=shard['parents']*dtype.itemsize: raise ValueError('Truncated shard')
                if verify and file_sha(path)!=shard[kind+'_sha256']: raise ValueError('Shard checksum failed')

    def __len__(self): return len(self.index)

    def __getitem__(self,index):
        row=self.index[index]; sid=int(row['shard'])
        if sid not in self.maps:
            s=self.meta['shards'][sid]
            self.maps[sid]=(np.memmap(self.root/s['ids'],dtype='<u4',mode='r'),np.memmap(self.root/s['masks'],dtype=self.geometry.mask_dtype,mode='r'))
        ids,masks=self.maps[sid]; start=int(row['offset']); stop=start+int(row['length'])
        return {'sequence':row['sequence'].decode(),'frame':int(row['frame']),
                'parent_ids':np.array(ids[start:stop]),'child_occ_mask':np.array(masks[start:stop])}

    def __getstate__(self):
        state=self.__dict__.copy(); state['maps']={}; return state

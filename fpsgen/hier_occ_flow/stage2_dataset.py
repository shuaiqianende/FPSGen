"""Sparse targets are queried after input-only predicted conditions are built."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from .storage import file_sha,atomic_json
from .dataset import input_xyz
from .stage2_condition import input_conditions,scene_seed
from .voxelization import lookup_masks,unpack_children,pack_children


class ConditionCache:
    """Optional bounded float16 sampled-feature cache, never dense feature grids.

    Default configs disable it for full-dataset pilots to avoid terabytes of
    Mx384 data. Overfit cohorts may explicitly enable a bounded cache.
    """
    def __init__(self,root,*,max_bytes):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.max_bytes=max_bytes

    @staticmethod
    def key(identity):return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def read(self,identity,device,geometry):
        path=self.root/(self.key(identity)+'.npz');meta=path.with_suffix('.json')
        if not path.exists() or not meta.exists():return None
        saved=json.loads(meta.read_text())
        if saved['identity']!=identity or file_sha(path)!=saved['sha256']:raise ValueError('Stale/corrupt condition cache')
        with np.load(path,allow_pickle=False) as data:
            ids=data['ids'].astype(np.int64)
            c={'parent_score':torch.from_numpy(data['scores']).to(device),
               'lidar_features':torch.from_numpy(data['features']).to(device),
               'observed_child':torch.from_numpy(unpack_children(data['observed'],geometry)).to(device),
               'valid_child_mask':torch.from_numpy(geometry.valid_children(ids)).to(device)}
        return torch.from_numpy(ids).to(device),c,saved['provenance']

    def write(self,identity,ids,condition,provenance,geometry):
        path=self.root/(self.key(identity)+'.npz')
        if path.exists():return
        arrays={'ids':ids.cpu().numpy().astype('<u4'),'scores':condition['parent_score'].cpu().numpy().astype('<f4'),
            'features':condition['lidar_features'].cpu().numpy().astype('<f2'),
            'observed':pack_children(condition['observed_child'].cpu().numpy(),geometry)}
        required=sum(a.nbytes for a in arrays.values())
        if sum(f.stat().st_size for f in self.root.glob('*.npz'))+required>self.max_bytes:
            raise RuntimeError('Condition cache budget exceeded; disable cache or explicitly register more disk space')
        tmp=path.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('wb') as handle:np.savez(handle,**arrays)
        os.replace(tmp,path)
        atomic_json(path.with_suffix('.json'),{'identity':identity,'sha256':file_sha(path),'provenance':provenance})


def assemble_stage2_scene(record,cfg,geometry,device,adapter,cache=None):
    path=Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
    seed=scene_seed(record['sequence'],record['frame'],cfg['candidates']['seed'])
    identity={'version':'stage2_384_parent_condition_v1','geometry_sha256':geometry.sha256,
        'stage1':adapter.provenance,'input_sha256':file_sha(path),'candidate_policy':cfg['candidates'],'seed':seed}
    if cfg['candidates']['mode']=='oracle':
        # Support identity belongs in the cache key, but never in model features.
        identity['oracle_support_sha256']=hashlib.sha256(record['parent_ids'].tobytes()).hexdigest()
    assembled=cache.read(identity,device,geometry) if cache else None
    if assembled is None:
        xyz=torch.from_numpy(input_xyz(path)).to(device)
        ids,c,source=input_conditions(xyz,geometry,adapter,cfg['candidates'],seed=seed,
            oracle_ids=record['parent_ids'] if cfg['candidates']['mode']=='oracle' else None)
        # Both cached and uncached paths use exactly the same feature precision.
        c['lidar_features']=c['lidar_features'].half()
        if cache:cache.write(identity,ids,c,source,geometry)
    else:ids,c,source=assembled
    if len(ids)>cfg['training']['max_parent_tokens']:
        raise RuntimeError(f'Full scene has{len(ids)}>{cfg["training"]["max_parent_tokens"]} validated tokens; no random truncation')
    target=unpack_children(lookup_masks(record['parent_ids'],record['child_occ_mask'],ids.cpu().numpy()),geometry)
    return ids,c,torch.as_tensor(target,device=device),source

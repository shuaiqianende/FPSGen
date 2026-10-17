"""Explicit own-experiment B1→B2 continuation; other protocol changes rejected."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .storage import file_sha


def batch_indices(cursor,batch_per_gpu,rank,world_size,length,seed):
    indices=[];orders={}
    for local in range(batch_per_gpu):
        epoch,index=divmod(cursor+rank*batch_per_gpu+local,length)
        if epoch not in orders:orders[epoch]=np.random.default_rng(seed+epoch).permutation(length)
        indices.append(int(orders[epoch][index]))
    return indices


def verify_batch_transition(value,identity,cfg,resume):
    transition=cfg.get('batch_transition')
    if not transition:raise ValueError('Changed config/output requires explicit batch-transition provenance')
    if Path(resume).resolve()!=Path(transition['checkpoint']).resolve() or file_sha(Path(resume))!=transition['checkpoint_sha256']:
        raise ValueError('Batch-transition source checkpoint path/SHA mismatch')
    old=copy.deepcopy(json.loads(Path(transition['source_config']).read_text()))
    if value.get('initialization_mode')!='scratch' or value.get('parameter_sync_pass') is not True:
        raise ValueError('Only synchronized scratch occupancy checkpoints may transition')
    old_identity=value['identity']
    for key in ('experiment','objective','geometry_sha256','train_manifest_sha256','development_manifest_sha256','world_size'):
        if identity[key]!=old_identity[key]:raise ValueError('Batch transition changed data/model/world identity')
    old_sha=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    if old_sha!=old_identity['config_sha256']:raise ValueError('Source config does not belong to checkpoint')
    if Path(transition['source_config']).resolve().parent!=Path(old_identity['output']).resolve():
        raise ValueError('Source config/output provenance mismatch')
    new=copy.deepcopy(cfg);new.pop('batch_transition',None)
    if old['training']['batch_per_gpu']!=1 or new['training']['batch_per_gpu']!=2:
        raise ValueError('Only the user-authorized per-GPU1→2 transition is allowed')
    old['training']['batch_per_gpu']=2
    old['engineering_gate']=new['engineering_gate']
    if old!=new:raise ValueError('Batch transition may change only batch and engineering-gate path')
    return value

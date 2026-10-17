"""Successful frame exposures, with explicit own-checkpoint budget revision."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .storage import file_sha


def final_update_for_frames(updates,frames,global_batch,target_frames):
    if any(type(x) is not int for x in (updates,frames,global_batch,target_frames)):
        raise ValueError('Budget counters must be integers')
    if updates<0 or frames<0 or global_batch<=0 or frames>target_frames:
        raise ValueError('Invalid or already exceeded frame budget')
    remaining=target_frames-frames
    if remaining%global_batch:raise ValueError('Exact frame budget requires a whole number of global batches')
    return updates+remaining//global_batch


def validate_frame_budget(cfg):
    train=cfg['training'];target=train.get('successful_frame_exposures')
    revision=cfg.get('budget_transition')
    if target is None:
        if revision:raise ValueError('Budget revision requires an explicit frame target')
        return
    if target!=100000 or not revision:raise ValueError('Register exactly100K successful frame exposures and explicit revision')
    expected=final_update_for_frames(revision['source_updates'],revision['source_successful_frames'],
        train['batch_per_gpu']*len(train['physical_gpus']),target)
    if expected!=train['successful_updates']:raise ValueError('Mixed-batch frame budget does not match final update')
    if revision['lr_policy']!='continuous_cosine_to_frame_budget' or not 0<revision['anchor_lr_factor']<=1:
        raise ValueError('Explicit continuous cosine continuation is required')


def learning_rate_multiplier(update,train,revision=None):
    if revision and update>=revision['source_updates']:
        progress=(update-revision['source_updates'])/max(train['successful_updates']-revision['source_updates'],1)
        return revision['anchor_lr_factor']*.5*(1+np.cos(np.pi*min(max(progress,0),1)))
    if update<train['warmup_updates']:return (update+1)/max(train['warmup_updates'],1)
    horizon=revision['previous_update_budget'] if revision else train['successful_updates']
    progress=(update-train['warmup_updates'])/max(horizon-train['warmup_updates'],1)
    return .5*(1+np.cos(np.pi*min(max(progress,0),1)))


def verify_budget_transition(value,identity,cfg,resume):
    revision=cfg['budget_transition'];validate_frame_budget(cfg)
    if Path(resume).resolve()!=Path(revision['checkpoint']).resolve() or file_sha(Path(resume))!=revision['checkpoint_sha256']:
        raise ValueError('Budget source checkpoint path/SHA mismatch')
    old=json.loads(Path(revision['source_config']).read_text());old_identity=value['identity']
    if value.get('initialization_mode')!='scratch' or value.get('parameter_sync_pass') is not True:
        raise ValueError('Only synchronized own scratch occupancy checkpoints may revise budget')
    for key in ('experiment','objective','geometry_sha256','train_manifest_sha256','development_manifest_sha256','world_size'):
        if identity[key]!=old_identity[key]:raise ValueError('Budget revision changed model/data/world identity')
    if hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()!=old_identity['config_sha256']:
        raise ValueError('Budget source config does not belong to checkpoint')
    if Path(revision['source_config']).resolve().parent!=Path(old_identity['output']).resolve():
        raise ValueError('Budget source output provenance mismatch')
    if value['successful_updates']!=revision['source_updates'] or value['sampler']['successful_frame_exposures']!=revision['source_successful_frames']:
        raise ValueError('Budget revision counters disagree with actual checkpoint')
    if old['training']['successful_updates']!=revision['previous_update_budget']:
        raise ValueError('Previous scheduler horizon mismatch')
    for group,nominal in zip(value['optimizer']['param_groups'],(cfg['training']['lr_encoder'],cfg['training']['lr_flow'])):
        if not np.isclose(group['lr']/nominal,revision['anchor_lr_factor'],rtol=1e-12,atol=1e-15):
            raise ValueError('Learning-rate anchor would jump at continuation')
    new=copy.deepcopy(cfg);new.pop('budget_transition')
    new['training'].pop('successful_frame_exposures');new['training']['successful_updates']=old['training']['successful_updates']
    new['engineering_gate']=old['engineering_gate']
    if old!=new:raise ValueError('Frame revision may change only stop budget, continuous scheduler registration and gate path')
    return value

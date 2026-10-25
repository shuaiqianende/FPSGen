"""Dedicated successful-exposure runner for EF/S2-E; no scratch-check relaxation."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
import yaml

from .encoder_adaptation import (ARMS,STAGE1_SHA,STAGE2_SHA,load_sources,
    frozen_state_sha,parameter_sha,update_adaptation_ema,stage2_guided_loss)
from .stage1_occ_adapter import FrozenStage1OccupancyAdapter
from .stage1_batch_transition import batch_indices
from .stage1_occupancy import ids_to_dense_occupancy
from .stage1_monitoring import module_grad_norm
from .stage2_runtime import source_sha
from .geometry import SparseHierarchyGeometry
from .storage import SparseShardDataset,atomic_json,file_sha
from .dataset import input_xyz
from .trainer import seed_all,rng_state,restore_rng,atomic_checkpoint
from .adaptation_gpu_slot import adaptation_admission


def config_sha(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def load_adaptation_config(path):
    cfg=yaml.safe_load(Path(path).read_text());g=SparseHierarchyGeometry(**cfg['geometry'])
    if g.parent_resolution!=(384,384,32) or g.child_split!=(2,2,2):raise ValueError('384/8-way geometry locked')
    if g.origin!=(-50.,-50.,-4.) or g.maximum!=(50.,50.,4.4) or g.sphere_radius!=50.:raise ValueError('ROI locked')
    if cfg['arm'] not in ARMS or cfg['initialization_mode']!='pretrained_ema_adaptation':raise ValueError('Explicit adaptation identity required')
    if cfg['phase']!=('P1' if cfg['arm'].startswith('EF') else 'P2'):raise ValueError('Arm/phase identity mismatch')
    t=cfg['training']
    if (t['successful_frame_exposures']!=20000 or t['successful_updates']!=10000 or t['batch_per_gpu']!=2
        or t['seed']!=20261032 or t['precision']!='fp16' or t['warmup_updates']!=500
        or t['ema_decay']!=.999 or t['grad_clip']!=1. or cfg.get('condition_cache',False)
        or t['optimizer']!='AdamW' or t['weight_decay']!=.01):
        raise ValueError('Registered20K/Batch2/single-seed pureFM protocol required')
    if t['lr_encoder']!=2e-5 or t['lr_flow']!=1e-5 or t['lr_stage2']!=2e-4:raise ValueError('Pre-registered matched learning rates locked')
    if cfg['source_stage1']['checkpoint_sha256']!=STAGE1_SHA or cfg['source_stage1']['weights']!='ema':raise ValueError('Original mature EMA required')
    if cfg['source_stage2']['checkpoint_sha256']!=STAGE2_SHA:raise ValueError('Original raw-GT Stage2 source required')
    reference=json.loads(Path(cfg['reference_stage2_config']).read_text());c=reference['candidates']
    if config_sha(reference)!=cfg['reference_stage2_config_sha256'] or reference['data_root']!=cfg['data_root']:
        raise ValueError('Frozen reference config/input-root identity changed')
    if (c['mode']!='predicted' or c['stage1_nfe']!=2 or c['selection']['strategy']!='threshold'
        or c['selection']['threshold']!=.5 or c['selection']['expansion']!=0
        or reference['stage1']['checkpoint_sha256']!=STAGE1_SHA):
        raise ValueError('Teacher support must use the exact frozen predicted NFE2/threshold0.5 protocol')
    if any(reference['model'].get(k,'none')!='none' for k in ('child_bridge','state_condition_interaction','local_state_encoding')):
        raise ValueError('Teacher/reference must not include M1/L0/L1')
    return cfg,g


def datasets(cfg,g,verify=True):
    train=SparseShardDataset(cfg['train_dataset'],g,verify=verify)
    dev=SparseShardDataset(cfg['development_dataset'],g,verify=verify)
    keys=lambda ds:{(r['sequence'].decode(),int(r['frame'])) for r in ds.index}
    if len(train)!=17870 or len(dev)!=900 or keys(train)&keys(dev) or any(s=='08' for s,f in keys(train)|keys(dev)):
        raise ValueError('Require original disjoint17870/900 raw surface-hit manifests, excluding seq08')
    return train,dev,{s+'_manifest_sha256':file_sha(Path(cfg[s+'_dataset'])/'manifest.json') for s in ('train','development')}


def verify_adaptation_resume(saved,identity,system):
    if saved.get('identity')!=identity or saved.get('initialization_mode')!='pretrained_ema_adaptation':
        raise ValueError('Not an exact resume of this registered adaptation run')
    if saved['freeze_policy']!=system.trainability_audit():raise ValueError('Freeze policy changed across resume')
    if saved['successful_frame_exposures']!=saved['successful_updates']*2:raise ValueError('Saved successful exposure count corrupt')
    if (saved['attempts']!=saved['successful_updates']+saved['amp_skips'] or saved['sample_cursor']!=saved['attempts']*2
        or saved['scheduler']['last_epoch']!=saved['successful_updates']):raise ValueError('Attempt/sampler/scheduler continuity mismatch')
    return saved


def make_optimizer(system,training):
    opt=torch.optim.AdamW(system.optimizer_groups(training),weight_decay=training['weight_decay'])
    def multiplier(i):
        if i<500:return (i+1)/500
        return .5*(1+math.cos(math.pi*min(1.,(i-500)/9500)))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,multiplier)


def step_loss(system,records,xyz,cfg,g,teacher,generator):
    if cfg['phase']=='P1':
        target=torch.cat([ids_to_dense_occupancy(r['parent_ids'],g,device=xyz[0].device) for r in records])
        return system.stage1_loss(xyz,target,generator)
    return stage2_guided_loss(system,records,xyz,cfg,g,teacher,generator)


def run_adaptation(cfg,g,output,*,resume=None,max_updates=10000,smoke=False):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3'):raise ValueError('P1/P2 require one admitted GPU2/3')
    admission=adaptation_admission(int(visible), configure_allocator=True)
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    if not 0<max_updates<=10000 or smoke and max_updates>32:raise ValueError('No budget extension')
    report=json.loads(Path(cfg['engineering_report']).read_text())
    if (report.get('status')!='PASS' or report['source_sha256']!=source_sha()
        or report['config_sha256']!=config_sha(cfg) or report['arm']!=cfg['arm']):
        raise ValueError('Maximum real Batch2/gradient/replay engineering report required')
    if not smoke:
        smoke_summary=json.loads(Path(cfg['smoke_summary']).read_text())
        if (smoke_summary.get('successful_updates')!=32 or smoke_summary.get('successful_frame_exposures')!=64
            or smoke_summary.get('identity',{}).get('source_sha256')!=source_sha()
            or smoke_summary.get('identity',{}).get('config_sha256')!=config_sha(cfg)
            or smoke_summary.get('identity',{}).get('arm')!=cfg['arm']
            or smoke_summary.get('freeze_violation') or smoke_summary.get('amp_skips',0)>2):
            raise ValueError('Own matched32-successful-update smoke required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if not resume and any((output/'checkpoints').glob('*.pt')):raise FileExistsError('Preserve existing run; use exact resume')
    torch.set_num_threads(4);device=torch.device('cuda')
    ds,dev,manifests=datasets(cfg,g,verify=True);seed_all(cfg['training']['seed'])
    system,_,initial=load_sources(cfg,device);seed_all(cfg['training']['seed']);t=cfg['training']
    identity={'experiment':'desescene-encoder-adaptation-v1','phase':cfg['phase'],'arm':cfg['arm'],
        'initialization_mode':'pretrained_ema_adaptation','source_sha256':source_sha(),'config_sha256':config_sha(cfg),
        'geometry_sha256':g.sha256,**manifests,'stage1_source_sha256':STAGE1_SHA,
        'stage2_source_sha256':STAGE2_SHA,'output':str(output),'world_size':1,
        'execution_scope':'engineering_smoke' if smoke else 'additional20k_adaptation'}
    ema=copy.deepcopy(system).eval().requires_grad_(False);optimizer,scheduler=make_optimizer(system,t)
    scaler=torch.cuda.amp.GradScaler(init_scale=t['amp_initial_scale'])
    generator=torch.Generator(device=device).manual_seed(t['seed']+1000)
    teacher=FrozenStage1OccupancyAdapter(**cfg['source_stage1']).to(device) if cfg['phase']=='P2' else None
    successful=attempts=cursor=skips=failures=0
    frozen=initial['frozen_state_sha256'];initial_encoder=initial['encoder_initial_parameter_sha256']
    if resume:
        value=verify_adaptation_resume(torch.load(resume,map_location='cpu'),identity,system)
        system.load_state_dict(value['model'],strict=True);ema.load_state_dict(value['ema'],strict=True)
        optimizer.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler']);scaler.load_state_dict(value['scaler'])
        restore_rng(value['rng'],generator,device)
        successful=value['successful_updates'];attempts=value['attempts'];cursor=value['sample_cursor'];skips=value['amp_skips'];failures=value['consecutive_failures']
    if frozen_state_sha(system)!=frozen or frozen_state_sha(ema)!=frozen:raise RuntimeError('Frozen state changed during load/resume')
    initial.update(shared_initial_parameter_hash=initial_encoder,
        optimizer_ema_scheduler_policy='fresh at cohort update0; exact state restoration on resume')
    atomic_json(output/'config.json',cfg);atomic_json(output/'initialization_audit.json',initial)
    launch={'pid':os.getpid(),'physical_gpu':int(visible),'identity':identity,
        'trainability':system.trainability_audit(),'initialization':initial,'resume':str(resume) if resume else None,
        'starting_updates':successful,'starting_optimizer_entries':len(optimizer.state),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'epoch':time.time()}
    atomic_json(output/'launch.json',launch)
    with (output/'launches.jsonl').open('a') as stream:stream.write(json.dumps(launch)+'\n')
    stopped=[False]
    def stop(signum,frame):stopped[0]=True
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def save(name):
        if frozen_state_sha(system)!=frozen or frozen_state_sha(ema)!=frozen:raise RuntimeError('Frozen state/EMA violation')
        atomic_checkpoint(output/'checkpoints'/name,{'identity':identity,'initialization_mode':'pretrained_ema_adaptation',
            'initialization':initial,'model':system.state_dict(),'ema':ema.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'rng':rng_state(generator,device),'freeze_policy':system.trainability_audit(),
            'successful_updates':successful,'successful_frame_exposures':successful*2,'attempts':attempts,
            'sample_cursor':cursor,'amp_skips':skips,'consecutive_failures':failures,'frozen_state_sha256':frozen,
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    if not resume:save('step000000.pt')
    system.train();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (output/'train.jsonl').open('a',buffering=1) as log:
        while successful<max_updates:
            tick=time.perf_counter();indices=batch_indices(cursor,2,0,1,len(ds),t['seed'])
            records=[ds[i] for i in indices]
            xyz=[torch.from_numpy(input_xyz(Path(cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).to(device) for r in records]
            optimizer.zero_grad(set_to_none=True)
            before_parameters={name:parameter_sha(module) for name,module in (('encoder',system.stage1.encoder),)
                               if successful==0}
            with torch.autocast('cuda',dtype=torch.float16):loss,details=step_loss(system,records,xyz,cfg,g,teacher,generator)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            encoder_norm=module_grad_norm(system.stage1.encoder)
            norm=torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad],t['grad_clip'])
            frozen_grad=any(p.grad is not None for p in system.parameters() if not p.requires_grad)
            if frozen_grad:raise RuntimeError('Frozen parameter received a gradient')
            finite=bool(torch.isfinite(loss) and torch.isfinite(norm) and norm>0)
            if cfg['arm']!='S2-E0':finite=bool(finite and np.isfinite(encoder_norm) and encoder_norm>0)
            attempts+=1;cursor+=2
            if finite:
                scaler.step(optimizer);scaler.update();scheduler.step();successful+=1;failures=0
                update_adaptation_ema(ema,system,t['ema_decay'])
            else:
                skips+=1;failures+=1;scaler.update(new_scale=scaler.get_scale()/2)
            if successful<=1 and finite and cfg['arm']!='S2-E0' and parameter_sha(system.stage1.encoder)==before_parameters.get('encoder'):
                raise RuntimeError('Encoder did not update despite finite gradient')
            if successful and successful%100==0 and frozen_state_sha(system)!=frozen:raise RuntimeError('Frozen parameter/state drift')
            torch.cuda.synchronize();row={'successful_updates':successful,'successful_frame_exposures':successful*2,
                'attempts':attempts,'sample_cursor':cursor,'amp_skips':skips,'finite':finite,
                'loss':float(loss) if torch.isfinite(loss) else None,'encoder_gradient_norm':encoder_norm if np.isfinite(encoder_norm) else None,
                'gradient_norm':float(norm) if torch.isfinite(norm) else None,'seconds':time.perf_counter()-tick,
                'batch_per_gpu':2,'records':[{'sequence':r['sequence'],'frame':r['frame'],'sampler_index':i} for r,i in zip(records,indices)],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'learning_rates':[x['lr'] for x in optimizer.param_groups]}
            if 'ids' in details:row['parent_ids_sha256']=hashlib.sha256(details['ids'].detach().cpu().numpy().tobytes()).hexdigest();row['scene_tokens']=details['scene_lengths']
            log.write(json.dumps(row,allow_nan=False)+'\n');print(f"{cfg['arm']} update={successful}/{max_updates} exposures={successful*2}/20000 fm={row['loss']} seconds={row['seconds']:.3f}",flush=True)
            del details,loss,xyz
            if finite and (successful%500==0 or successful==max_updates) or stopped[0] or failures>=3:
                save('resume_latest.pt')
                if successful%2500==0 or successful==max_updates:save(f'step{successful:06d}.pt')
            if stopped[0] or failures>=3:break
    for s,h in handlers.items():signal.signal(s,h)
    summary={'status':'COMPLETE_REQUESTED_BUDGET' if successful==max_updates else 'STOPPED_HEALTH_FAILURE',
        'identity':identity,'successful_updates':successful,'successful_frame_exposures':successful*2,'attempts':attempts,
        'amp_skips':skips,'freeze_violation':False,'frozen_state_sha256':frozen,'seconds':time.perf_counter()-start,
        'encoder_parameter_sha256':parameter_sha(system.stage1.encoder),'encoder_changed':parameter_sha(system.stage1.encoder)!=initial_encoder}
    atomic_json(output/'summary.json',summary)
    if successful!=max_updates:raise RuntimeError('Adaptation stopped before requested successful budget')

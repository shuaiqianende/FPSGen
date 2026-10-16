"""Successful-update training, atomic identity-checked resume and joint RNG state.

First-stage encoder is frozen. FM and BCE objectives are intentionally separate.
EMA is an optional parameter average for evaluation, never a supervision target.
"""
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import numpy as np
import torch
from .model import HierarchicalOccupancyModel
from .storage import SparseShardDataset,atomic_json,file_sha
from .geometry import SparseHierarchyGeometry
from .dataset import assemble_scene
from .flow_matching import training_pair,fm_loss,bce_loss


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rng_state(generator,device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'source':generator.get_state(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(value,generator,device):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    generator.set_state(value['source'])
    if value['cuda'] is not None: torch.cuda.set_rng_state(value['cuda'],device)


def atomic_checkpoint(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: torch.save(value,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def load_resume(path,identity):
    ckpt=torch.load(path,map_location='cpu')
    if ckpt.get('identity')!=identity: raise ValueError('Resume identity/geometry/dataset/config/world-size mismatch')
    return ckpt


@torch.no_grad()
def update_ema(ema,online,decay):
    for target,source in zip(ema.parameters(),online.parameters()): target.mul_(decay).add_(source,alpha=1-decay)
    for target,source in zip(ema.buffers(),online.buffers()):
        if target.is_floating_point(): target.mul_(decay).add_(source,alpha=1-decay)
        else: target.copy_(source)


def build_model(cfg,g): return HierarchicalOccupancyModel(g,**cfg['model'])


def run_training(cfg,output,*,device_name='cpu',resume=None,max_updates=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    if device_name=='cuda':
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if visible not in ('2','3','2,3'): raise ValueError('Set physical CUDA_VISIBLE_DEVICES=2/3 only')
        from .resources import gpu_preflight
        marker=output/f"gpu_preflight_{os.environ.get('TORCHELASTIC_RUN_ID','single')}.json"
        if rank==0: atomic_json(marker,gpu_preflight(tuple(map(int,visible.split(',')))))
        deadline=time.time()+30
        while not marker.exists():
            if time.time()>deadline: raise RuntimeError('Rank0 preflight timeout')
            time.sleep(.1)
        gate=json.loads(marker.read_text())
        if gate['status']!='AVAILABLE': raise RuntimeError(f'GPU experiment blocked; existing training preserved: {gate}')
        torch.cuda.set_device(local); device=torch.device('cuda',local)
    else:
        if world!=1 or cfg['model']['backbone']!='local' or 'pyramid' in cfg['model']['condition_mode']:
            raise ValueError('CPU mechanics runs support local backbone with input-derived conditions only')
        device=torch.device('cpu')
    if world>1: torch.distributed.init_process_group('nccl')
    g=SparseHierarchyGeometry(**cfg['geometry'])
    dataset=SparseShardDataset(cfg['train_dataset'],g,verify=True)
    train=cfg['training']; steps=train['successful_updates'] if max_updates is None else max_updates
    if steps<1 or steps>train['successful_updates']:
        raise ValueError('--max-updates can only shorten the frozen scheduler budget')
    expected_loss={'fm':1.0} if cfg['model']['objective']=='fm' else {'bce':1.0}
    if cfg.get('loss',expected_loss)!=expected_loss: raise ValueError('First version permits only the configured FM MSE or direct BCE objective')
    if cfg.get('flow',{'source':'gaussian','sigma':1.,'target_mapping':'binary_symmetric'})!={'source':'gaussian','sigma':1.,'target_mapping':'binary_symmetric'}:
        raise ValueError('HS-OccFlow uses unit Gaussian and binary symmetric target, not historical density mapping')
    if steps>5000:
        gate=json.loads(Path(cfg.get('engineering_gate','')).read_text()) if cfg.get('engineering_gate') else {}
        if not gate.get('formal_ready') or gate.get('geometry_sha256')!=g.sha256:
            raise ValueError('Formal training requires new data, resource, resume, and BCE/FM quality gates')
    if not resume and any((output/'checkpoints').glob('*.pt')): raise FileExistsError('Fresh output contains checkpoints')
    identity={'geometry_sha256':g.sha256,'dataset_manifest_sha256':file_sha(Path(cfg['train_dataset'])/'manifest.json'),
        'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'world_size':world,'experiment':'hs-occflow-v1'}
    stage1_needed='pyramid' in cfg['model']['condition_mode'] or cfg['candidates']['mode']=='predicted'
    if stage1_needed: identity['stage1_checkpoint_sha256']=file_sha(cfg['stage1_checkpoint'])
    code_manifest={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}
    source_code_sha=hashlib.sha256(json.dumps(code_manifest,sort_keys=True).encode()).hexdigest()
    seed_all(train['seed']); torch.set_num_threads(train.get('cpu_threads',4))
    raw=build_model(cfg,g).to(device)
    if world>1: model=torch.nn.parallel.DistributedDataParallel(raw,device_ids=[local])
    else: model=raw
    ema=copy.deepcopy(raw).eval().requires_grad_(False) if train.get('ema_decay') is not None else None
    optimizer=torch.optim.AdamW(raw.parameters(),lr=train['lr'],weight_decay=train['weight_decay'])
    def lr_multiplier(u):
        if u<train['warmup_updates']: return (u+1)/max(train['warmup_updates'],1)
        progress=(u-train['warmup_updates'])/max(train['successful_updates']-train['warmup_updates'],1)
        return .5*(1+math.cos(math.pi*min(max(progress,0),1)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and train['precision']=='fp16')
    generator=torch.Generator(device=device).manual_seed(train['seed']+rank+1000)
    encoder=None
    if 'pyramid' in cfg['model']['condition_mode'] or cfg['candidates']['mode']=='predicted':
        from .condition import FrozenVoxDetCondition
        encoder=FrozenVoxDetCondition(cfg['stage1_checkpoint']).to(device)
    successful=attempts=0
    if resume:
        value=load_resume(resume,identity)
        raw.load_state_dict(value['model'],strict=True); optimizer.load_state_dict(value['optimizer'])
        scheduler.load_state_dict(value['scheduler']); scaler.load_state_dict(value['scaler'])
        if ema is not None: ema.load_state_dict(value['ema'],strict=True)
        successful=value['successful_updates']; attempts=value['attempts']
        restore_rng(value['rank_rng'][rank],generator,device)
    if rank==0:
        atomic_json(output/'config.json',cfg)
        atomic_json(output/'source_code_manifest.json',{'sha256':source_code_sha,'files':code_manifest})
    stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    old_handlers={signum:signal.signal(signum,request_stop) for signum in (signal.SIGINT,signal.SIGTERM)}

    def save(name):
        state=rng_state(generator,device)
        states=[None]*world
        if world>1: torch.distributed.all_gather_object(states,state)
        else: states=[state]
        if rank==0:
            atomic_checkpoint(output/'checkpoints'/name,{'identity':identity,'model':raw.state_dict(),
                'ema':ema.state_dict() if ema is not None else None,'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'successful_updates':successful,
                'attempts':attempts,'rank_rng':states,'sampler':{'attempts':attempts,'world_size':world,'seed':train['seed']},
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_code_sha256':source_code_sha,'stage1_checkpoint_sha256':identity.get('stage1_checkpoint_sha256')})
    if not resume: save('step000000.pt')
    raw.train(); start=time.perf_counter(); failures=0
    log=(output/f'train_rank{rank}.jsonl').open('a',buffering=1)
    while successful<steps:
        tick=time.perf_counter(); position=attempts*world+rank; epoch,index=divmod(position,len(dataset))
        order=np.random.default_rng(train['seed']+epoch).permutation(len(dataset)); record=dataset[int(order[index])]
        ids,condition,target,provenance=assemble_scene(record,cfg,g,device,encoder,train['seed']+position)
        optimizer.zero_grad(set_to_none=True)
        precision=torch.autocast('cuda',dtype=torch.float16,enabled=scaler.is_enabled()) if device.type=='cuda' else nullcontext()
        with precision:
            if raw.objective=='fm':
                pair=training_pair(target,condition['valid_child_mask'],generator)
                prediction=model(ids,state=pair['state'],time=pair['time'],**condition)
                loss=fm_loss(prediction,pair['velocity'],condition['valid_child_mask'])
            else:
                prediction=model(ids,**condition); loss=bce_loss(prediction,target,condition['valid_child_mask'])
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(raw.parameters(),train['grad_clip'])
        finite=torch.tensor(bool(torch.isfinite(loss) and torch.isfinite(grad) and grad>0),device=device,dtype=torch.int32)
        if world>1: torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
        attempts+=1
        if not bool(finite):
            failures+=1
            if scaler.is_enabled(): scaler.update(new_scale=scaler.get_scale()/2)
            if failures>=3: stopping=True
        else:
            scaler.step(optimizer); scaler.update(); scheduler.step(); successful+=1; failures=0
            if ema is not None: update_ema(ema,raw,train['ema_decay'])
        row={'successful_updates':successful,'attempts':attempts,'loss':float(loss.detach()) if bool(torch.isfinite(loss)) else None,'objective':raw.objective,
             'gradient_norm':float(grad) if bool(torch.isfinite(grad)) else None,'finite':bool(finite),'lr':optimizer.param_groups[0]['lr'],'tokens':len(ids),
             'sequence':record['sequence'],'frame':record['frame'],'source':provenance,'seconds':time.perf_counter()-tick}
        if device.type=='cuda': row['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        log.write(json.dumps(row,allow_nan=False)+'\n')
        if rank==0: print(f"update={successful}/{steps} {raw.objective}={float(loss):.6f} parents={len(ids)} seconds={row['seconds']:.3f}",flush=True)
        stop_flag=torch.tensor(int(stopping),device=device)
        if world>1: torch.distributed.all_reduce(stop_flag,op=torch.distributed.ReduceOp.MAX)
        stopping=bool(stop_flag)
        if successful and (successful%train['checkpoint_every']==0 or successful==steps) or stopping:
            save('resume_latest.pt')
            if successful%train.get('milestone_every',5000)==0: save(f'step{successful:06d}.pt')
        if bool(finite) and cfg.get('validation',{}).get('every_updates') and successful%cfg['validation']['every_updates']==0:
            # Validation consumes stochastic serialization operations, so restore
            # the training RNG afterward to preserve exact resume/data sequence.
            state=rng_state(generator,device)
            if rank==0:
                from .validation import evaluate_model
                evaluation=evaluate_model(ema if ema is not None else raw,cfg,g,device,encoder)
                evaluation.update(ema_used=ema is not None,successful_updates=successful)
                atomic_json(output/'validation'/f'update_{successful:06d}.json',evaluation)
            if world>1: torch.distributed.barrier()
            restore_rng(state,generator,device)
        if stopping: break
    log.close()
    for signum,handler in old_handlers.items(): signal.signal(signum,handler)
    if rank==0: atomic_json(output/'summary.json',{'status':'PAUSED' if stopping else 'COMPLETE_REQUESTED_BUDGET',
        'successful_updates':successful,'attempts':attempts,'scope':cfg.get('scope','engineering'),
        'seconds':time.perf_counter()-start,'identity':identity,'formal_training_complete':steps==100000 and successful==100000})
    if world>1: torch.distributed.destroy_process_group()

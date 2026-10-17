"""Scratch-only Stage1 occupancy runner; dense binary CFM or matched plain BCE.

No old density mapping, head, teacher loss, shortcut schedule or weight loader.
GPU resource and pilot gates are checked before model CUDA allocation.
"""
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
from .geometry import SparseHierarchyGeometry
from .stage1_occupancy import Stage1DirectOccupancy,ids_to_dense_occupancy,occupancy_flow_pair,masked_occ_loss
from .storage import SparseShardDataset,atomic_json,file_sha
from .dataset import input_xyz
from .trainer import seed_all,rng_state,restore_rng,atomic_checkpoint,update_ema
from .stage1_monitoring import module_grad_norm,gradient_stats,sample_parameters,sample_changes
from .stage1_batch_transition import batch_indices,verify_batch_transition


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_config(cfg):
    g=SparseHierarchyGeometry(**cfg['geometry'])
    if g.parent_resolution!=(384,384,32) or g.child_split!=(2,2,2): raise ValueError('384+8way is locked')
    if g.origin!=(-50.,-50.,-4.) or g.maximum!=(50.,50.,4.4) or g.sphere_radius!=50.:
        raise ValueError('World bounds and50m sphere are fixed for this protocol')
    if cfg['model'].get('initialization')!='scratch' or cfg['model'].get('pretrained_encoder') is not None:
        raise ValueError('Scratch only; historical model weights prohibited')
    allowed={'objective','initialization','pretrained_encoder','widths','activation_checkpointing','input_feature_version'}
    if set(cfg['model'])-allowed:raise ValueError('Unsupported model options; init_cfg/load_from/pretrained adapters are prohibited')
    if cfg['model'].get('input_feature_version')!='coarse_f64_v1':raise ValueError('Require geometry-audited float64 coarse indexing')
    objective=cfg['model']['objective']
    if objective not in ('fm','bce') or cfg.get('loss')!={objective:1.0}: raise ValueError('Only pure FM or direct BCE loss allowed')
    if objective=='fm' and cfg.get('flow')!={'source':'gaussian','sigma':1.0,'target':'binary_symmetric'}:
        raise ValueError('Unit Gaussian to binary -1/+1 occupancy required')
    if cfg['training']['physical_gpus']!=[2,3] or cfg['training']['batch_per_gpu'] not in (1,2):
        raise ValueError('Supported configuration is GPU2/3 B1 or B2/rank')
    if cfg['training']['precision'] not in ('fp16','fp32'): raise ValueError('Unsupported precision')
    if cfg['training'].get('optimizer','AdamW')!='AdamW' or cfg['training'].get('scheduler','warmup_cosine')!='warmup_cosine':
        raise ValueError('This runner supports explicitly registered AdamW and warmup_cosine only')
    if cfg['training']['successful_updates']>100000: raise ValueError('Budget exceeds registration')
    return g


def verify_stage1_resume(value,identity):
    if value.get('identity')!=identity or value.get('initialization_mode')!='scratch':
        raise ValueError('Checkpoint experiment/config/data/geometry/world-size identity mismatch')
    if value.get('parameter_sync_pass') is False:raise ValueError('Cannot resume a checkpoint with detected parameter desynchronization')
    return value


def run_stage1(cfg,output,*,resume=None,max_updates=None,smoke=False):
    g=validate_config(cfg); output=Path(output)
    if not resume and any((output/'checkpoints').glob('*.pt')): raise FileExistsError('Fresh output already contains checkpoints')
    train=cfg['training']; steps=train['successful_updates'] if max_updates is None else max_updates
    if not 0<steps<=train['successful_updates']: raise ValueError('max-updates may only shorten registered budget')
    if cfg['scope']=='formal' and (steps!=100000 or train['successful_updates']!=100000):
        raise ValueError('Formal scope is the full100K budget; use a separately labeled smoke/pilot config for short runs')
    if smoke and steps>32: raise ValueError('Smoke is limited to32 successful updates')
    if steps>1000 or cfg['scope']=='formal':
        gate=json.loads(Path(cfg['engineering_gate']).read_text()) if cfg.get('engineering_gate') else {}
        required=('formal_ready','full_preprocessing','gpu_shapes','ddp_resume','fm_bce_overfit','pilot1000')
        if not all(gate.get(k) is True for k in required) or gate.get('geometry_sha256')!=g.sha256:
            raise ValueError('Formal Stage1 requires independently measured data/GPU/overfit/pilot gates')
        if train['batch_per_gpu']==2 and not gate.get('batch2_ddp_resume'):
            raise ValueError('B2 formal continuation requires actual B2 DDP/resume evidence')
        config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if gate.get('registered_config_sha256')!=config_sha:raise ValueError('Formal config changed after engineering gate freeze')
        for key,split in (('train_dataset','train'),('development_dataset','development')):
            if gate.get(split+'_manifest_sha256')!=file_sha(Path(cfg[key])/'manifest.json'):
                raise ValueError('Formal dataset manifest changed after gate freeze')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3','2,3'): raise ValueError('Only physical GPU2/3 allowed')
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    if world!=len(visible.split(',')): raise ValueError('Use one process per selected physical GPU')
    # Launcher checks admission before torchrun, avoiding self-PID admission races.
    gate_path=os.environ.get('HS_OCCFLOW_ADMISSION')
    if gate_path:
        gate=json.loads(Path(gate_path).read_text())
        if gate.get('status')!='AVAILABLE' or time.time()-gate.get('checked_at_epoch',0)>120:
            raise RuntimeError('GPU admission missing/expired/blocked')
        if gate.get('physical_ids')!=list(map(int,visible.split(','))): raise ValueError('GPU admission identity mismatch')
    else:
        from .resources import gpu_preflight
        gate=gpu_preflight(tuple(map(int,visible.split(','))))
        if gate['status']!='AVAILABLE': raise RuntimeError('Allowed GPUs busy; existing training preserved')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: torch.distributed.init_process_group('nccl')
    if rank==0:print('phase=data_verify: checking sparse dataset versions and checksums',flush=True)
    ds=SparseShardDataset(cfg['train_dataset'],g,verify=True)
    dev=SparseShardDataset(cfg['development_dataset'],g,verify=True)
    train_keys={(r['sequence'].decode(),int(r['frame'])) for r in ds.index}
    dev_keys={(r['sequence'].decode(),int(r['frame'])) for r in dev.index}
    if (train_keys&dev_keys and cfg['scope']!='overfit') or any(s=='08' for s,f in train_keys|dev_keys):
        raise ValueError('Train/development leakage; only labeled overfit scope may evaluate training frames')
    if cfg['scope']=='formal' and (len(ds)!=17870 or len(dev)!=900): raise ValueError('Formal full split incomplete')
    if rank==0:print(f'phase=data_verified train={len(ds)} development={len(dev)} target_updates={steps}',flush=True)
    identity={'experiment':'hs-occflow-stage1-occ384-v1','objective':cfg['model']['objective'],'geometry_sha256':g.sha256,
        'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        'train_manifest_sha256':file_sha(Path(cfg['train_dataset'])/'manifest.json'),
        'development_manifest_sha256':file_sha(Path(cfg['development_dataset'])/'manifest.json'),
        'world_size':world,'output':str(output.resolve())}
    seed_all(train['seed']); torch.set_num_threads(4)
    raw=Stage1DirectOccupancy(g,objective=cfg['model']['objective'],widths=cfg['model']['widths'],
        activation_checkpointing=cfg['model']['activation_checkpointing']).to(device)
    model=torch.nn.parallel.DistributedDataParallel(raw,device_ids=[local]) if world>1 else raw
    # Initial broadcast precedes EMA construction; no pre-existing EMA is loaded.
    ema=copy.deepcopy(raw).eval().requires_grad_(False) if train.get('ema_decay') else None
    optimizer=torch.optim.AdamW([{'params':raw.encoder.parameters(),'lr':train['lr_encoder'],'name':'encoder'},
        {'params':raw.flow.parameters(),'lr':train['lr_flow'],'name':'flow'}],weight_decay=train['weight_decay'])
    def multiplier(u):
        if u<train['warmup_updates']: return (u+1)/max(train['warmup_updates'],1)
        p=(u-train['warmup_updates'])/max(train['successful_updates']-train['warmup_updates'],1)
        return .5*(1+np.cos(np.pi*min(max(p,0),1)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    scaler=torch.cuda.amp.GradScaler(enabled=train['precision']=='fp16',init_scale=train.get('amp_initial_scale',16384.))
    generator=torch.Generator(device=device).manual_seed(train['seed']+1000+rank)
    successful=attempts=failures=0
    counters={'clipping_events':0,'overflow_attempts':0,'invalid_attempts':0}
    initialization={'initialization_mode':'scratch','pretrained_loaded':False,
        'loaded_pretrained_parameter_count':0,'optimizer_initial_state_entries':len(optimizer.state),
        'successful_updates':0,'initial_model_sha256':state_sha(raw),'ema_initialized_from':'current_random_online',
        'initial_ema_sha256':state_sha(ema) if ema is not None else None}
    if ema is not None and initialization['initial_model_sha256']!=initialization['initial_ema_sha256']: raise RuntimeError('EMA initialization mismatch')
    if world>1:
        hashes=[None]*world; torch.distributed.all_gather_object(hashes,initialization['initial_model_sha256'])
        if len(set(hashes))!=1: raise RuntimeError('DDP initialization differs')
    sample_cursor=0;successful_frame_exposures=0;batch_transition_applied=False
    if resume:
        value=torch.load(resume,map_location='cpu')
        batch_transition_applied=bool(cfg.get('batch_transition') and value.get('identity')!=identity)
        value=verify_batch_transition(value,identity,cfg,resume) if batch_transition_applied else verify_stage1_resume(value,identity)
        raw.load_state_dict(value['model'],strict=True); optimizer.load_state_dict(value['optimizer'])
        scheduler.load_state_dict(value['scheduler']); scaler.load_state_dict(value['scaler'])
        if ema is not None: ema.load_state_dict(value['ema'],strict=True)
        successful=value['successful_updates']; attempts=value['attempts']; initialization=value['initialization']
        failures=value.get('consecutive_failures',0);counters=value.get('monitor_counters',counters)
        sample_cursor=value['sampler'].get('sample_cursor',attempts*world)
        successful_frame_exposures=value['sampler'].get('successful_frame_exposures',successful*world)
        restore_rng(value['rank_rng'][rank],generator,device)
    output.mkdir(parents=True,exist_ok=True)
    code_sha={p.name:file_sha(p) for p in Path(__file__).parent.glob('stage1*.py')}
    launch_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if rank==0:
        atomic_json(output/'config.json',cfg); atomic_json(output/'initialization_audit.json',initialization)
        atomic_json(output/'source_code_sha256.json',code_sha)
        atomic_json(output/'launch_parameters.json',{'rank0_pid':os.getpid(),'parent_pid':os.getppid(),
            'physical_gpus':list(map(int,visible.split(','))),'world_size':world,'scope':cfg['scope'],
            'actual_update_budget':steps,'starting_successful_updates':successful,'resume':str(resume) if resume else None,
            'batch_per_gpu':train['batch_per_gpu'],'global_batch':train['batch_per_gpu']*world,
            'epoch_time':time.time(),'identity':identity,'git_commit':launch_commit})
        print(f'phase=model_ready initialization=scratch encoder_params={sum(p.numel() for p in raw.encoder.parameters())} flow_params={sum(p.numel() for p in raw.flow.parameters())}',flush=True)
    stopped=[False]
    def request_stop(signum,frame): stopped[0]=True
    old={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(name):
        states=[None]*world; state=rng_state(generator,device)
        if world>1: torch.distributed.all_gather_object(states,state)
        else: states=[state]
        local_hash={'online':state_sha(raw),'ema':state_sha(ema) if ema is not None else None}
        hashes=[None]*world
        if world>1:torch.distributed.all_gather_object(hashes,local_hash)
        else:hashes=[local_hash]
        sync_pass=len({h['online'] for h in hashes})==1 and len({h['ema'] for h in hashes})==1
        if not sync_pass:stopped[0]=True
        if rank==0:
            atomic_checkpoint(output/'checkpoints'/name,{'identity':identity,'initialization_mode':'scratch',
                'initialization':initialization,'model':raw.state_dict(),'ema':ema.state_dict() if ema is not None else None,
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'successful_updates':successful,'attempts':attempts,'rank_rng':states,
                'monitor_counters':counters,'consecutive_failures':failures,
                'sampler':{'attempts':attempts,'world_size':world,'seed':train['seed'],'batch_per_gpu':train['batch_per_gpu'],
                    'sample_cursor':sample_cursor,'successful_frame_exposures':successful_frame_exposures},'source_code_sha256':code_sha,
                'batch_transition':cfg.get('batch_transition'),
                'git_commit':launch_commit,
                'rank_model_hashes':hashes,'parameter_sync_pass':sync_pass})
            print(f'checkpoint={output / "checkpoints" / name} update={successful} ranks_synchronized={sync_pass}',flush=True)
    if not resume: save('step000000.pt')
    elif batch_transition_applied:
        save('batch_transition_start.pt')
        if rank==0:
            os.link(output/'checkpoints/batch_transition_start.pt',output/'checkpoints/resume_latest.pt')
            atomic_json(output/'batch_transition_audit.json',{'source':cfg['batch_transition'],'starting_updates':successful,
                'starting_attempts':attempts,'starting_sample_cursor':sample_cursor,'starting_successful_frame_exposures':successful_frame_exposures,
                'model_sha256':state_sha(raw),'ema_sha256':state_sha(ema),'optimizer_entries':len(optimizer.state),
                'scheduler_last_epoch':scheduler.last_epoch,'amp_scale':scaler.get_scale(),'learning_rates':[x['lr'] for x in optimizer.param_groups]})
    modules={'encoder':raw.encoder,'flow':raw.flow,'velocity_head':raw.flow.output}
    if raw.objective=='fm': modules['time_embedding']=raw.flow.embedding
    samples={k:sample_parameters(m) for k,m in modules.items()}
    if ema is not None: samples['ema']=sample_parameters(ema)
    raw.train(); start=time.perf_counter(); log=(output/f'train_rank{rank}.jsonl').open('a',buffering=1)
    while successful<steps:
        tick=time.perf_counter()
        records=[ds[index] for index in batch_indices(sample_cursor,train['batch_per_gpu'],rank,world,len(ds),train['seed'])]
        record=records[0]
        data_start=tick  # include sampler and sparse mmap record retrieval above
        scenes=[torch.from_numpy(input_xyz(Path(cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).to(device) for r in records]
        xyz=scenes[0] if len(scenes)==1 else scenes
        target=torch.cat([ids_to_dense_occupancy(r['parent_ids'],g,device=device) for r in records],0)
        data_seconds=time.perf_counter()-data_start
        optimizer.zero_grad(set_to_none=True)
        pair=occupancy_flow_pair(target,raw.valid,generator=generator) if raw.objective=='fm' else None
        with torch.autocast('cuda',dtype=torch.float16,enabled=scaler.is_enabled()):
            prediction=model(xyz,state=pair['state'],time=pair['time']) if pair is not None else model(xyz)
            loss=masked_occ_loss(prediction,pair['velocity'] if pair is not None else target,raw.valid,objective=raw.objective)
        scale_before=scaler.get_scale();scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        norms={name:module_grad_norm(module) for name,module in (('encoder',raw.encoder),('flow',raw.flow))}
        audit=(successful==0 or (successful+1)%train.get('gradient_audit_every',50)==0)
        details={name:gradient_stats(module) for name,module in modules.items()} if audit else None
        grad=torch.nn.utils.clip_grad_norm_(raw.parameters(),train['grad_clip'])
        if bool(torch.isfinite(grad)) and float(grad)>train['grad_clip']: counters['clipping_events']+=1
        finite=torch.tensor(int(bool(torch.isfinite(loss) and torch.isfinite(grad)) and all(np.isfinite(x) and x>0 for x in norms.values())),device=device)
        if world>1: torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
        attempts+=1
        sample_cursor+=world*train['batch_per_gpu']
        if not bool(finite):
            failures+=1
            counters['invalid_attempts']+=1
            if scaler.is_enabled() and not bool(torch.isfinite(grad)):counters['overflow_attempts']+=1
            if scaler.is_enabled(): scaler.update(new_scale=scaler.get_scale()/2)
            if failures>=3: stopped[0]=True
        else:
            scaler.step(optimizer); scaler.update(); scheduler.step(); successful+=1; failures=0
            successful_frame_exposures+=world*train['batch_per_gpu']
            # BN buffers have per-rank sample statistics; choose rank0 before EMA.
            if world>1:
                for buffer in raw.buffers(): torch.distributed.broadcast(buffer,0)
            if ema is not None: update_ema(ema,raw,train['ema_decay'])
        torch.cuda.synchronize(device)
        global_loss=loss.detach().float().clone()
        if world>1:torch.distributed.all_reduce(global_loss);global_loss/=world
        changed={}
        if audit and bool(finite):
            for name,module in modules.items(): samples[name],changed[name]=sample_changes(module,samples[name])
            if ema is not None: samples['ema'],changed['ema']=sample_changes(ema,samples['ema'])
        row={'successful_updates':successful,'attempts':attempts,'finite':bool(finite),
            'loss':float(global_loss) if bool(torch.isfinite(global_loss)) else None,
            'rank_loss':float(loss.detach()) if bool(torch.isfinite(loss)) else None,
            'sampled_t':(float(pair['time']) if pair['time'].numel()==1 else pair['time'].tolist()) if pair is not None else None,
            'batch_records':[{'sequence':r['sequence'],'frame':r['frame']} for r in records],
            'prediction_shape':list(prediction.shape),
            'global_batch':world*train['batch_per_gpu'],'sample_cursor':sample_cursor,'successful_frame_exposures':successful_frame_exposures,
            'gradient_norms':{k:v if np.isfinite(v) else None for k,v in norms.items()},
            'lr_encoder':optimizer.param_groups[0]['lr'],'lr_flow':optimizer.param_groups[1]['lr'],
            'sequence':record['sequence'],'frame':record['frame'],'seconds':time.perf_counter()-tick,'data_seconds':data_seconds,
            'global_gradient_before_clip':float(grad) if bool(torch.isfinite(grad)) else None,
            'gradient_details':details,'parameter_changes':changed,'gradient_clipping_events':counters['clipping_events'],
            'amp_scale':scaler.get_scale(),'amp_scale_before':scale_before,'amp_overflow_count':counters['overflow_attempts'],
            'skipped_attempts':attempts-successful,
            'epoch_time':time.time(),'optimizer_steps':int(next(iter(optimizer.state.values()))['step']) if optimizer.state else 0,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}
        log.write(json.dumps(row,allow_nan=False)+'\n')
        if rank==0: print(f"update={successful}/{steps} {raw.objective}={row['loss']} gradients={norms} seconds={row['seconds']:.3f}",flush=True)
        stop=torch.tensor(int(stopped[0]),device=device)
        if world>1: torch.distributed.all_reduce(stop,op=torch.distributed.ReduceOp.MAX)
        stopped[0]=bool(stop)
        if successful and successful%train['checkpoint_every']==0 or successful==steps or stopped[0]:
            save('resume_latest.pt')
            if successful and successful%train['milestone_every']==0: save(f'step{successful:06d}.pt')
        val=cfg['validation']
        if bool(finite) and successful%val['light_every']==0:
            state=rng_state(generator,device)
            if rank==0:
                from .stage1_evaluation import evaluate_stage1
                frames=None if successful%val['full_every']==0 else val['light_frames']
                print(f'phase=validation update={successful} frames={frames if frames is not None else len(dev)}',flush=True)
                use_ema=ema is not None and val.get('use_ema',True)
                sample_dir=output/'validation'/f'update_{successful:06d}_sample' if val.get('save_sample') else None
                result=evaluate_stage1(ema if use_ema else raw,cfg,g,device,max_frames=frames,sample_output=sample_dir)
                result.update(successful_updates=successful,ema_used=use_ema)
                atomic_json(output/'validation'/f'update_{successful:06d}.json',result)
                print(f'validation update={successful} ema={use_ema} metrics='+str({k:v['summary']['macro'] for k,v in result['nfe_results'].items()}),flush=True)
                if use_ema and val.get('compare_online'):
                    online=evaluate_stage1(raw,cfg,g,device,max_frames=val['light_frames'])
                    online.update(successful_updates=successful,ema_used=False,scope='online_light_diagnostic')
                    atomic_json(output/'validation'/f'online_update_{successful:06d}.json',online)
                if not use_ema and ema is not None and val.get('compare_ema'):
                    averaged=evaluate_stage1(ema,cfg,g,device,max_frames=val['light_frames'])
                    averaged.update(successful_updates=successful,ema_used=True,scope='ema_light_diagnostic')
                    atomic_json(output/'validation'/f'ema_update_{successful:06d}.json',averaged)
                if successful==steps and val.get('train_eval_at_end'):
                    diagnostic=copy.deepcopy(cfg);diagnostic['development_dataset']=cfg['train_dataset']
                    train_metrics=evaluate_stage1(ema if use_ema else raw,diagnostic,g,device)
                    train_metrics.update(scope='training_set_overfit_diagnostic',successful_updates=successful,ema_used=use_ema)
                    atomic_json(output/'validation'/f'train_update_{successful:06d}.json',train_metrics)
            if world>1: torch.distributed.barrier()
            restore_rng(state,generator,device)
        if stopped[0]: break
    log.close()
    for sig,handler in old.items(): signal.signal(sig,handler)
    if rank==0: atomic_json(output/'summary.json',{'status':'PAUSED' if stopped[0] else 'COMPLETE_REQUESTED_BUDGET',
        'successful_updates':successful,'attempts':attempts,'seconds':time.perf_counter()-start,
        'formal_training_complete':cfg['scope']=='formal' and successful==100000})
    if world>1: torch.distributed.destroy_process_group()

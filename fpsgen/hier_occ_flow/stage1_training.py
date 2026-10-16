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


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_config(cfg):
    g=SparseHierarchyGeometry(**cfg['geometry'])
    if g.parent_resolution!=(384,384,32) or g.child_split!=(2,2,2): raise ValueError('384+8way is locked')
    if cfg['model'].get('initialization')!='scratch' or cfg['model'].get('pretrained_encoder') is not None:
        raise ValueError('Scratch only; historical model weights prohibited')
    if cfg['model'].get('input_feature_version')!='coarse_f64_v1':raise ValueError('Require geometry-audited float64 coarse indexing')
    objective=cfg['model']['objective']
    if objective not in ('fm','bce') or cfg.get('loss')!={objective:1.0}: raise ValueError('Only pure FM or direct BCE loss allowed')
    if objective=='fm' and cfg.get('flow')!={'source':'gaussian','sigma':1.0,'target':'binary_symmetric'}:
        raise ValueError('Unit Gaussian to binary -1/+1 occupancy required')
    if cfg['training']['physical_gpus']!=[2,3] or cfg['training']['batch_per_gpu']!=1:
        raise ValueError('Validated starting configuration is GPU2/3 B1/rank')
    if cfg['training']['precision'] not in ('fp16','fp32'): raise ValueError('Unsupported precision')
    if cfg['training']['successful_updates']>100000: raise ValueError('Budget exceeds registration')
    return g


def verify_stage1_resume(value,identity):
    if value.get('identity')!=identity or value.get('initialization_mode')!='scratch':
        raise ValueError('Checkpoint experiment/config/data/geometry/world-size identity mismatch')
    return value


def run_stage1(cfg,output,*,resume=None,max_updates=None,smoke=False):
    g=validate_config(cfg); output=Path(output)
    if not resume and any((output/'checkpoints').glob('*.pt')): raise FileExistsError('Fresh output already contains checkpoints')
    train=cfg['training']; steps=train['successful_updates'] if max_updates is None else max_updates
    if not 0<steps<=train['successful_updates']: raise ValueError('max-updates may only shorten registered budget')
    if smoke and steps>32: raise ValueError('Smoke is limited to32 successful updates')
    if steps>1000:
        gate=json.loads(Path(cfg['engineering_gate']).read_text()) if cfg.get('engineering_gate') else {}
        required=('formal_ready','full_preprocessing','gpu_shapes','ddp_resume','fm_bce_overfit','pilot1000')
        if not all(gate.get(k) is True for k in required) or gate.get('geometry_sha256')!=g.sha256:
            raise ValueError('Formal Stage1 requires independently measured data/GPU/overfit/pilot gates')
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
    ds=SparseShardDataset(cfg['train_dataset'],g,verify=True)
    dev=SparseShardDataset(cfg['development_dataset'],g,verify=True)
    train_keys={(r['sequence'].decode(),int(r['frame'])) for r in ds.index}
    dev_keys={(r['sequence'].decode(),int(r['frame'])) for r in dev.index}
    if (train_keys&dev_keys and cfg['scope']!='overfit') or any(s=='08' for s,f in train_keys|dev_keys):
        raise ValueError('Train/development leakage; only labeled overfit scope may evaluate training frames')
    if cfg['scope']=='formal' and (len(ds)!=17870 or len(dev)!=900): raise ValueError('Formal full split incomplete')
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
    successful=attempts=0; initialization={'initialization_mode':'scratch','pretrained_loaded':False,
        'loaded_pretrained_parameter_count':0,'optimizer_initial_state_entries':len(optimizer.state),
        'successful_updates':0,'initial_model_sha256':state_sha(raw),'ema_initialized_from':'current_random_online',
        'initial_ema_sha256':state_sha(ema) if ema is not None else None}
    if ema is not None and initialization['initial_model_sha256']!=initialization['initial_ema_sha256']: raise RuntimeError('EMA initialization mismatch')
    if world>1:
        hashes=[None]*world; torch.distributed.all_gather_object(hashes,initialization['initial_model_sha256'])
        if len(set(hashes))!=1: raise RuntimeError('DDP initialization differs')
    if resume:
        value=verify_stage1_resume(torch.load(resume,map_location='cpu'),identity)
        raw.load_state_dict(value['model'],strict=True); optimizer.load_state_dict(value['optimizer'])
        scheduler.load_state_dict(value['scheduler']); scaler.load_state_dict(value['scaler'])
        if ema is not None: ema.load_state_dict(value['ema'],strict=True)
        successful=value['successful_updates']; attempts=value['attempts']; initialization=value['initialization']
        restore_rng(value['rank_rng'][rank],generator,device)
    output.mkdir(parents=True,exist_ok=True)
    code_sha={p.name:file_sha(p) for p in Path(__file__).parent.glob('stage1*.py')}
    if rank==0:
        atomic_json(output/'config.json',cfg); atomic_json(output/'initialization_audit.json',initialization)
        atomic_json(output/'source_code_sha256.json',code_sha)
    stopped=[False]
    def request_stop(signum,frame): stopped[0]=True
    old={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(name):
        states=[None]*world; state=rng_state(generator,device)
        if world>1: torch.distributed.all_gather_object(states,state)
        else: states=[state]
        if rank==0:
            atomic_checkpoint(output/'checkpoints'/name,{'identity':identity,'initialization_mode':'scratch',
                'initialization':initialization,'model':raw.state_dict(),'ema':ema.state_dict() if ema is not None else None,
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'successful_updates':successful,'attempts':attempts,'rank_rng':states,
                'sampler':{'attempts':attempts,'world_size':world,'seed':train['seed']},'source_code_sha256':code_sha,
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    if not resume: save('step000000.pt')
    modules={'encoder':raw.encoder,'flow':raw.flow,'velocity_head':raw.flow.output}
    if raw.objective=='fm': modules['time_embedding']=raw.flow.embedding
    samples={k:sample_parameters(m) for k,m in modules.items()}
    if ema is not None: samples['ema']=sample_parameters(ema)
    raw.train(); failures=0; clipped=0; start=time.perf_counter(); log=(output/f'train_rank{rank}.jsonl').open('a',buffering=1)
    while successful<steps:
        tick=time.perf_counter(); position=attempts*world+rank; epoch,index=divmod(position,len(ds))
        order=np.random.default_rng(train['seed']+epoch).permutation(len(ds)); record=ds[int(order[index])]
        data_start=time.perf_counter()
        path=Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
        xyz=torch.from_numpy(input_xyz(path)).to(device)
        target=ids_to_dense_occupancy(record['parent_ids'],g,device=device)
        data_seconds=time.perf_counter()-data_start
        optimizer.zero_grad(set_to_none=True)
        pair=occupancy_flow_pair(target,raw.valid,generator=generator) if raw.objective=='fm' else None
        with torch.autocast('cuda',dtype=torch.float16,enabled=scaler.is_enabled()):
            prediction=model(xyz,state=pair['state'],time=pair['time']) if pair is not None else model(xyz)
            loss=masked_occ_loss(prediction,pair['velocity'] if pair is not None else target,raw.valid,objective=raw.objective)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        norms={name:module_grad_norm(module) for name,module in (('encoder',raw.encoder),('flow',raw.flow))}
        audit=(successful==0 or (successful+1)%train.get('gradient_audit_every',50)==0)
        details={name:gradient_stats(module) for name,module in modules.items()} if audit else None
        grad=torch.nn.utils.clip_grad_norm_(raw.parameters(),train['grad_clip'])
        if bool(torch.isfinite(grad)) and float(grad)>train['grad_clip']: clipped+=1
        finite=torch.tensor(int(bool(torch.isfinite(loss) and torch.isfinite(grad)) and all(np.isfinite(x) and x>0 for x in norms.values())),device=device)
        if world>1: torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
        attempts+=1
        if not bool(finite):
            failures+=1
            if scaler.is_enabled(): scaler.update(new_scale=scaler.get_scale()/2)
            if failures>=3: stopped[0]=True
        else:
            scaler.step(optimizer); scaler.update(); scheduler.step(); successful+=1; failures=0
            # BN buffers have per-rank sample statistics; choose rank0 before EMA.
            if world>1:
                for buffer in raw.buffers(): torch.distributed.broadcast(buffer,0)
            if ema is not None: update_ema(ema,raw,train['ema_decay'])
        torch.cuda.synchronize(device)
        changed={}
        if audit and bool(finite):
            for name,module in modules.items(): samples[name],changed[name]=sample_changes(module,samples[name])
            if ema is not None: samples['ema'],changed['ema']=sample_changes(ema,samples['ema'])
        row={'successful_updates':successful,'attempts':attempts,'finite':bool(finite),
            'loss':float(loss.detach()) if bool(torch.isfinite(loss)) else None,'gradient_norms':{k:v if np.isfinite(v) else None for k,v in norms.items()},
            'lr_encoder':optimizer.param_groups[0]['lr'],'lr_flow':optimizer.param_groups[1]['lr'],
            'sequence':record['sequence'],'frame':record['frame'],'seconds':time.perf_counter()-tick,'data_seconds':data_seconds,
            'global_gradient_before_clip':float(grad) if bool(torch.isfinite(grad)) else None,
            'gradient_details':details,'parameter_changes':changed,'gradient_clipping_events':clipped,
            'amp_scale':scaler.get_scale(),'skipped_attempts':attempts-successful,
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
                use_ema=ema is not None and val.get('use_ema',True)
                sample_dir=output/'validation'/f'update_{successful:06d}_sample' if val.get('save_sample') else None
                result=evaluate_stage1(ema if use_ema else raw,cfg,g,device,max_frames=frames,sample_output=sample_dir)
                result.update(successful_updates=successful,ema_used=use_ema)
                atomic_json(output/'validation'/f'update_{successful:06d}.json',result)
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

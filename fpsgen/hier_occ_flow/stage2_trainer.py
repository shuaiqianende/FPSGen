"""384 Stage2-only training with frozen Occupancy Stage1 and exact own resume."""
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
from .stage2_model import Stage2OccupancyModel as HierarchicalOccupancyModel
from .stage1_occ_adapter import FrozenStage1OccupancyAdapter
from .stage2_dataset import assemble_stage2_scene,ConditionCache
from .stage2_runtime import dataset_identity,config_sha,source_sha,sampler_index,require_training_gate,frame_budget_plan,direct_full_train_policy
from .stage2_evaluation import evaluate_stage2
from .flow_matching import training_pair,fm_loss,bce_loss
from .trainer import seed_all,atomic_checkpoint,rng_state,restore_rng,update_ema
from .stage1_training import state_sha
from .storage import atomic_json,file_sha
from .stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse


def own_resume(value,identity):
    if value.get('identity')!=identity or value.get('initialization')!='scratch_stage2':raise ValueError('Not this384 Stage2 run')
    if value.get('parameter_sync_pass') is not True:raise ValueError('Unsynchronized checkpoint rejected')
    return value


def run_stage2(cfg,g,output,*,resume=None,max_updates=None,transition=None):
    output=Path(output).resolve();train=cfg['training'];require_training_gate(cfg,g)
    steps=train['successful_updates'] if max_updates is None else max_updates
    if not 0<steps<=train['successful_updates']:raise ValueError('--max-updates only shortens a frozen horizon')
    if resume and transition:raise ValueError('Exact resume and explicit batch transition are mutually exclusive')
    if not resume and any((output/'checkpoints').glob('*.pt')):raise FileExistsError('Fresh Stage2 output contains checkpoints')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3','2,3'):raise ValueError('Only physical GPU2/3')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    batch_per_gpu=train['batch_per_gpu'];global_batch=world*batch_per_gpu
    if world!=len(visible.split(',')):raise ValueError('One rank per physical GPU')
    admission=json.loads(Path(os.environ['HS_STAGE2_ADMISSION']).read_text())
    if admission['status']!='AVAILABLE' or admission['physical_ids']!=list(map(int,visible.split(','))) or time.time()-admission['checked_at_epoch']>120:
        raise RuntimeError('Stage2 admission unavailable; never compete with Stage1')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:torch.distributed.init_process_group('nccl')
    torch.set_num_threads(train.get('cpu_threads',4))
    ds,dev,manifests=dataset_identity(cfg,g,verify=True)
    # CPU sparse input labels retain the full split. Cohort only changes sampler.
    cohort=cfg.get('cohort_indices')
    if train['scope']=='overfit' and (not cohort or not 8<=len(cohort)<=16 or len(set(cohort))!=len(cohort)):
        raise ValueError('Overfit requires frozen8-16 distinct real frames')
    if train['scope'] not in ('overfit','engineering_smoke') and cohort is not None:raise ValueError('Pilots must sample the full17870 training set')
    identity={'experiment':'hs-occflow-stage2-384-v1','geometry_sha256':g.sha256,'config_sha256':config_sha(cfg),
        **manifests,'world_size':world,'output':str(output),'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],
        'source_sha256':source_sha()}
    seed_all(train['seed']);adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval()
    if train['scope']=='formal' and not adapter.provenance['stage1_final_100k_frames']:
        raise ValueError('Formal Stage2 requires a fixed final Stage1, not the development intermediate checkpoint')
    seed_all(train['seed']);raw=HierarchicalOccupancyModel(g,**cfg['model']).to(device)
    model=torch.nn.parallel.DistributedDataParallel(raw,device_ids=[local]) if world>1 else raw
    ema=copy.deepcopy(raw).eval().requires_grad_(False) if train.get('ema_decay') else None
    optimizer=torch.optim.AdamW(raw.parameters(),lr=train['lr'],weight_decay=train['weight_decay'])
    def multiplier(update):
        if update<train['warmup_updates']:return (update+1)/max(train['warmup_updates'],1)
        p=(update-train['warmup_updates'])/max(train['successful_updates']-train['warmup_updates'],1)
        return .5*(1+math.cos(math.pi*min(max(p,0),1)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    scaler=torch.cuda.amp.GradScaler(enabled=train['precision']=='fp16',init_scale=16384.)
    generator=torch.Generator(device=device).manual_seed(train['seed']+1000+rank)
    successful=attempts=failures=amp_skips=sample_cursor=successful_frames=0;transition_audit=None
    if resume:
        value=own_resume(torch.load(resume,map_location='cpu'),identity)
        raw.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
        scheduler.load_state_dict(value['scheduler']);scaler.load_state_dict(value['scaler'])
        if ema is not None:ema.load_state_dict(value['ema'],strict=True)
        successful=value['successful_updates'];attempts=value['attempts'];amp_skips=value['amp_skips'];failures=value['consecutive_failures']
        sample_cursor=value['sampler'].get('sample_cursor',attempts*global_batch)
        successful_frames=value['sampler'].get('successful_frame_exposures',successful*global_batch)
        restore_rng(value['rank_rng'][rank],generator,device)
    if transition:
        specification=cfg.get('transition',{})
        if str(Path(transition).resolve())!=specification.get('checkpoint') or file_sha(transition)!=specification.get('checkpoint_sha256'):
            raise ValueError('Explicit transition checkpoint path/SHA mismatch')
        value=torch.load(transition,map_location='cpu');previous=value.get('identity',{})
        previous_cfg=json.loads((Path(previous['output'])/'config.json').read_text())
        common=('geometry_sha256','stage1_checkpoint_sha256','train_manifest_sha256','development_manifest_sha256')
        if (previous.get('experiment')!='hs-occflow-stage2-384-v1' or value.get('initialization')!='scratch_stage2'
            or value.get('parameter_sync_pass') is not True or any(previous.get(k)!=identity[k] for k in common)
            or previous_cfg['model']!=cfg['model'] or previous_cfg['model']['objective']!='fm'
            or config_sha(previous_cfg)!=previous['config_sha256']):raise ValueError('Only verified own384 FM continuation allowed')
        raw.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
        scheduler.load_state_dict(value['scheduler']);scaler.load_state_dict(value['scaler'])
        if ema is not None:ema.load_state_dict(value['ema'],strict=True)
        successful=value['successful_updates'];attempts=value['attempts'];amp_skips=value['amp_skips'];failures=value['consecutive_failures']
        old_batch=previous['world_size']*previous_cfg['training']['batch_per_gpu']
        sample_cursor=value['sampler'].get('sample_cursor',attempts*old_batch)
        successful_frames=value['sampler'].get('successful_frame_exposures',successful*old_batch)
        if previous['world_size']==world and old_batch==global_batch:restore_rng(value['rank_rng'][rank],generator,device)
        elif rank==0:restore_rng(value['rank_rng'][0],generator,device)
        old_cursor=sample_cursor
        cohort_changed=previous_cfg.get('cohort_indices')!=cohort
        if cohort_changed:sample_cursor=0
        # The newly introduced rank gets an independent reproducible source
        # stream. This is a registered batch/world-size transition, not an
        # assertion of bit-identical RNG continuation across different batches.
        before=[p['lr'] for p in optimizer.param_groups]
        for group,base in zip(optimizer.param_groups,scheduler.base_lrs):group['lr']=base*multiplier(successful)
        scheduler._last_lr=[p['lr'] for p in optimizer.param_groups]
        transition_audit={'checkpoint':str(Path(transition).resolve()),'checkpoint_sha256':file_sha(transition),
            'source_identity':previous,'old_global_batch':old_batch,'new_global_batch':global_batch,
            'inherited_successful_updates':successful,'inherited_frame_exposures':successful_frames,
            'sampler_cursor':sample_cursor,'lr_before':before,'lr_after':scheduler.get_last_lr(),
            'previous_sampler_cursor':old_cursor,'cohort_changed_reset_order':cohort_changed,
            'optimizer_ema_scaler_preserved':True,'rng_policy':'rank0 restores source; new rank uses independent registered seed'}
    phase_start=value.get('phase_start_updates',successful) if resume else successful
    phase_start_frames=value.get('phase_start_frame_exposures',successful_frames) if resume else successful_frames
    frame_target=train.get('successful_frame_exposures');budget=None
    if frame_target is not None:
        budget=frame_budget_plan(successful,successful_frames,frame_target,global_batch)
        if train['successful_updates']!=budget['target_successful_updates']:
            raise ValueError('Frozen optimizer horizon must exactly match actual successful-frame budget')
    direct_policy=direct_full_train_policy(cfg,g)
    output.mkdir(parents=True,exist_ok=True);cache=None
    if cfg['cache']['enabled']:
        cache=ConditionCache(Path(cfg['cache']['root'])/output.name/f'rank{rank}',max_bytes=cfg['cache']['max_bytes']//world)
    if rank==0:
        atomic_json(output/'config.json',cfg)
        atomic_json(output/'launch.json',{'rank0_pid':os.getpid(),'parent_pid':os.getppid(),'physical_gpus':list(map(int,visible.split(','))),
            'identity':identity,'stage1':adapter.provenance,'initialization':'scratch_stage2','starting_updates':successful,
            'optimizer_entries':len(optimizer.state),'initial_model_sha256':state_sha(raw),'initial_ema_sha256':state_sha(ema) if ema else None,
            'trainable_parameter_names':[name for name,p in raw.named_parameters() if p.requires_grad],
            'batch_per_gpu':batch_per_gpu,'global_batch':global_batch,'batching':'packed_variable_length','transition':transition_audit})
        if budget or direct_policy:
            launch=json.loads((output/'launch.json').read_text())
            atomic_json(output/'launch.json',{**launch,'frame_budget':budget,'direct_training':direct_policy})
    stopped=[False]
    def request_stop(signum,frame):stopped[0]=True
    handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(name):
        states=[None]*world
        if world>1:torch.distributed.all_gather_object(states,rng_state(generator,device))
        else:states=[rng_state(generator,device)]
        hashes=[None]*world;local_hash={'online':state_sha(raw),'ema':state_sha(ema) if ema else None}
        if world>1:torch.distributed.all_gather_object(hashes,local_hash)
        else:hashes=[local_hash]
        synced=len({r['online'] for r in hashes})==1 and len({r['ema'] for r in hashes})==1
        if not synced:stopped[0]=True
        if rank==0:
            atomic_checkpoint(output/'checkpoints'/name,{'identity':identity,'initialization':'scratch_stage2',
                'model':raw.state_dict(),'ema':ema.state_dict() if ema else None,'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'successful_updates':successful,
                'attempts':attempts,'amp_skips':amp_skips,'consecutive_failures':failures,'rank_rng':states,
                'sampler':{'attempts':attempts,'seed':train['seed'],'world_size':world,'cohort_indices':cohort,
                    'batch_per_gpu':batch_per_gpu,'global_batch':global_batch,'sample_cursor':sample_cursor,'successful_frame_exposures':successful_frames},
                'parameter_sync_pass':synced,'rank_hashes':hashes,'stage1':adapter.provenance,
                'batch_transition':transition_audit,
                'phase_start_updates':phase_start,
                'phase_start_frame_exposures':phase_start_frames,'frame_budget':budget,
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    if not resume:save(f'step{successful:06d}_transition.pt' if transition else 'step000000.pt')
    log=(output/f'train_rank{rank}.jsonl').open('a',buffering=1);start=time.perf_counter();raw.train()
    while successful<steps and (frame_target is None or successful_frames<frame_target):
        tick=time.perf_counter();length=len(cohort) if cohort else len(ds)
        scenes=[];records=[];sources=[]
        for slot in range(batch_per_gpu):
            index=sampler_index(sample_cursor+rank*batch_per_gpu+slot,0,1,length,train['seed']);index=cohort[index] if cohort else index
            record=ds[index];assembled=assemble_stage2_scene(record,cfg,g,device,adapter,cache)
            scenes.append(assembled);sources.append(assembled[3]);records.append({'sampler_index':index,'sequence':record['sequence'],'frame':record['frame'],'tokens':len(assembled[0])})
        ids,c,target,scene_lengths=pack_scenes(scenes);del scenes
        if len(ids)>train.get('max_batch_parent_tokens',batch_per_gpu*train['max_parent_tokens']):raise RuntimeError('Full packed batch exceeds registered tokens; no truncation')
        optimizer.zero_grad(set_to_none=True)
        dtype=torch.bfloat16 if train['precision']=='bf16' else torch.float16
        with torch.autocast('cuda',dtype=dtype,enabled=train['precision']!='fp32'):
            if raw.objective=='fm':
                pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],generator)
                loss=scene_mean_mse(model(ids,state=pair['state'],time=pair['time'],**c),pair['velocity'],c['valid_child_mask'],c['batch'])
            else:loss=bce_loss(model(ids,**c),target,c['valid_child_mask'])
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(raw.parameters(),train['grad_clip'])
        finite=torch.tensor(int(bool(torch.isfinite(loss) and torch.isfinite(grad)) and float(grad)>0),device=device)
        if world>1:torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
        attempts+=1;sample_cursor+=global_batch
        if not bool(finite):
            failures+=1;amp_skips+=1
            if scaler.is_enabled():scaler.update(new_scale=scaler.get_scale()/2)
            if failures>=3:stopped[0]=True
        else:
            scaler.step(optimizer);scaler.update();scheduler.step();successful+=1;successful_frames+=global_batch;failures=0
            if world>1:
                for buffer in raw.buffers():torch.distributed.broadcast(buffer,0)
            if ema is not None:update_ema(ema,raw,train['ema_decay'])
        if any(p.grad is not None or p.requires_grad for p in adapter.parameters()):raise RuntimeError('Stage1 freeze violated')
        torch.cuda.synchronize(device)
        display_loss=loss.detach().float().clone()
        if world>1:torch.distributed.all_reduce(display_loss);display_loss/=world
        row={'successful_updates':successful,'attempts':attempts,**records[0],
            'batch_records':records,'batch_per_gpu':batch_per_gpu,'global_batch':global_batch,
            'successful_frame_exposures':successful_frames,'sample_cursor':sample_cursor,'tokens_per_scene':scene_lengths,
            'target_successful_frame_exposures':frame_target,
            'phase_successful_frame_exposures':successful_frames-phase_start_frames,
            'tokens':len(ids),'loss':float(display_loss) if torch.isfinite(display_loss) else None,'rank_loss':float(loss) if torch.isfinite(loss) else None,
            'gradient_norm':float(grad) if torch.isfinite(grad) else None,'finite':bool(finite),'amp_skips':amp_skips,
            'lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-tick,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'candidate_source':sources[0] if len(sources)==1 else sources}
        log.write(json.dumps(row,allow_nan=False)+'\n')
        if rank==0:print(f'update={successful}/{steps} frames={successful_frames}/{frame_target or "unbounded"} {raw.objective}={row["loss"]} global_batch={global_batch} local_scene_tokens={scene_lengths} seconds={row["seconds"]:.3f}',flush=True)
        stop=torch.tensor(int(stopped[0]),device=device)
        if world>1:torch.distributed.all_reduce(stop,op=torch.distributed.ReduceOp.MAX)
        stopped[0]=bool(stop)
        startup_checkpoint=bool(finite) and successful in train.get('startup_checkpoint_updates',[])
        if (successful and successful%train['checkpoint_every']==0) or successful==steps or stopped[0] or startup_checkpoint:
            save('resume_latest.pt')
            if successful==steps or successful%train['milestone_every']==0 or startup_checkpoint:save(f'step{successful:06d}.pt')
        due=cfg['validation'].get('enabled',True) and bool(finite) and (successful%cfg['validation']['every_updates']==0 or successful==steps)
        if due and not stopped[0] and train['scope']!='engineering_smoke':
            state=rng_state(generator,device)
            if rank==0:
                report=evaluate_stage2(ema if ema is not None else raw,cfg,g,device,adapter,dev)
                report.update(successful_updates=successful,ema_used=ema is not None)
                atomic_json(output/'validation'/f'update_{successful:06d}.json',report)
                if cohort:
                    # The short overfit gate checks the trained online network;
                    # an EMA with decay .999 has barely moved after 200 updates.
                    report=evaluate_stage2(raw,cfg,g,device,adapter,ds,indices=cohort,cache=cache)
                    report.update(successful_updates=successful,scope='training_overfit_diagnostic',ema_used=False)
                    atomic_json(output/'overfit'/f'update_{successful:06d}.json',report)
            if world>1:torch.distributed.barrier()
            restore_rng(state,generator,device)
        if stopped[0]:break
    log.close()
    for s,handler in handlers.items():signal.signal(s,handler)
    if rank==0:atomic_json(output/'summary.json',{'status':'PAUSED' if stopped[0] else 'COMPLETE_REQUESTED_BUDGET',
        'successful_updates':successful,'attempts':attempts,'scope':train['scope'],'seconds':time.perf_counter()-start,
        'global_batch':global_batch,'successful_frame_exposures':successful_frames,
        'phase_start_updates':phase_start,'phase_successful_updates':successful-phase_start,
        'phase_start_frame_exposures':phase_start_frames,'phase_successful_frame_exposures':successful_frames-phase_start_frames,
        'frame_budget':budget,'direct_training':direct_policy,
        'formal_training_complete':train['scope']=='formal' and successful==train['successful_updates']
            and (frame_target is None or successful_frames==frame_target),
        'stage1':adapter.provenance,'identity':identity})
    if world>1:torch.distributed.destroy_process_group()

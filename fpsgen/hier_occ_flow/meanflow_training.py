"""Bounded, independent MeanFlow adaptation; no changes to scratch/FM runners."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import torch
import yaml
from .encoder_adaptation import STAGE1_SHA,STAGE2_SHA,load_stage2_source,tensor_state_sha
from .encoder_adaptation_runtime import datasets,config_sha
from .geometry import SparseHierarchyGeometry
from .stage1_occ_adapter import FrozenStage1OccupancyAdapter
from .stage1_occupancy import occupancy_flow_pair,ids_to_dense_occupancy
from .stage1_batch_transition import batch_indices
from .stage2_batching import pack_scenes,packed_training_pair
from .stage2_dataset import assemble_stage2_scene
from .stage2_runtime import source_sha
from .meanflow_math import meanflow_loss,sample_endpoints
from .meanflow_stage1 import MeanFlowStage1
from .meanflow_stage2 import MeanFlowStage2
from .storage import atomic_json,file_sha
from .trainer import seed_all,rng_state,restore_rng,atomic_checkpoint
from .dataset import input_xyz
from .resources import gpu_preflight


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text());g=SparseHierarchyGeometry(**cfg['geometry']);t=cfg['training']
    if cfg['experiment']!='desescene-meanflow-v1' or cfg['arm'] not in ('S1-MF20K','S2-MF20K'):
        raise ValueError('Independent registered MeanFlow identity required')
    if (g.parent_resolution!=(384,384,32) or g.child_split!=(2,2,2) or g.origin!=(-50.,-50.,-4.)
        or g.maximum!=(50.,50.,4.4) or g.sphere_radius!=50.):raise ValueError('Original geometry locked')
    expected=dict(successful_frame_exposures=20000,successful_updates=10000,batch_per_gpu=2,seed=20261032,
        precision='fp16',optimizer='AdamW',weight_decay=.01,warmup_updates=500,grad_clip=1.,ema_decay=.999,
        amp_initial_scale=16384.)
    if any(t.get(k)!=v for k,v in expected.items()):raise ValueError('20K/B2/single-seed matched protocol locked')
    if t['learning_rate']!=(1e-5 if cfg['arm']=='S1-MF20K' else 2e-4):raise ValueError('Matched LR locked')
    if cfg['meanflow']!={'diagonal_probability':.75,'adaptive_power':0.,'adaptive_epsilon':.001,
        'jvp_backend':'torch.func.jvp','time_direction':'noise0_to_data1','target_sign':'plus'}:
        raise ValueError('Only preregistered original MeanFlow endpoint objective')
    if cfg.get('condition_cache',False) or cfg['source_stage1']['checkpoint_sha256']!=STAGE1_SHA or cfg['source_stage1']['weights']!='ema':
        raise ValueError('Original frozen Stage1 EMA / no cross-step projection cache')
    if cfg['source_stage2']['checkpoint_sha256']!=STAGE2_SHA:raise ValueError('Original online Stage2 source required')
    if file_sha(cfg['specification']['path'])!=cfg['specification']['sha256']:raise ValueError('Specification changed')
    policy=json.loads(Path(cfg['reference_stage2_config']).read_text());c=policy['candidates']
    if (config_sha(policy)!=cfg['reference_stage2_config_sha256'] or c['mode']!='predicted' or c['stage1_nfe']!=2
        or any(c['selection'].get(k)!=v for k,v in {'strategy':'threshold','threshold':.5,'expansion':0,'overflow':'error','budget':500000}.items())
        or policy['stage1']['checkpoint_sha256']!=STAGE1_SHA or policy['data_root']!=cfg['data_root']):
        raise ValueError('Original predicted teacher support identity changed')
    if any(policy['model'].get(k,'none')!='none' for k in ('child_bridge','state_condition_interaction','local_state_encoding')):
        raise ValueError('No experimental Stage2 branches')
    return cfg,g


def admission(physical,configure=False):
    if physical not in (2,3) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical):
        raise ValueError('One admitted physical GPU2/3 required')
    # User permits co-running only when real B2 fits. Existing own predecessor
    # tasks must additionally be complete (enforced by the queue controller).
    shared=os.environ.get('DESESCENE_MF_SHARED')=='1'
    report=gpu_preflight((physical,),minimum_free_gib=16.,allow_shared_compute=shared)
    if configure and report['status']=='AVAILABLE':
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(14.*2**30/total,0)
    return report


class MeanFlowSystem:
    def __init__(self,cfg,g,device):
        self.cfg=cfg;self.g=g;self.device=torch.device(device)
        self.teacher=FrozenStage1OccupancyAdapter(**cfg['source_stage1']).to(device).eval()
        self.teacher.requires_grad_(False)
        if cfg['arm']=='S1-MF20K':
            self.network=MeanFlowStage1(self.teacher.model.flow).to(device).requires_grad_(True)
            self.network.base.checkpointing=False
            self.initialization={'stage1':self.teacher.provenance,'interval':self.network.initialization_audit}
        else:
            base,audit=load_stage2_source(cfg,g)
            if audit['source_successful_exposures']!=99520 or base.backbone_kind!='ptv3':
                raise ValueError('99520-exposure original dynamic PTv3 required')
            self.network=MeanFlowStage2(base).to(device).requires_grad_(True)
            self.initialization={'stage1':self.teacher.provenance,'stage2':audit,'interval':self.network.initialization_audit}
        self.network.train()
        self.policy=json.loads(Path(cfg['reference_stage2_config']).read_text())
        self.initialization.update(initialization_mode='pretrained_online_meanflow_adaptation' if cfg['arm'].startswith('S2') else 'pretrained_ema_meanflow_adaptation',
            initial_network_state_sha256=tensor_state_sha(self.network.state_dict()),
            frozen_state_sha256=self.frozen_sha(),source_exposures=99520 if cfg['arm'].startswith('S2') else 100000)

    def frozen_sha(self):
        state=self.teacher.model.state_dict()
        if self.cfg['arm']=='S1-MF20K':state={k:v for k,v in state.items() if not k.startswith('flow.')}
        return tensor_state_sha(state)

    def prepare(self,records,source_generator,interval_generator,*,force_finite=False):
        """Geometry/frozen Encoder outside JVP; one source/time draw per real B2."""
        if len(records)!=2:raise ValueError('True complete Batch2 required')
        if self.cfg['arm']=='S1-MF20K':
            xyz=[torch.from_numpy(input_xyz(Path(self.cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).to(self.device) for r in records]
            target=torch.cat([ids_to_dense_occupancy(r['parent_ids'],self.g,device=self.device) for r in records])
            pair=occupancy_flow_pair(target,self.teacher.model.valid,generator=source_generator)
            with torch.no_grad():
                pyramid=self.teacher.model.encoder(xyz)
                observed=torch.cat([self.teacher.model.geometry.observed_mask(x) for x in xyz])
            valid=self.teacher.model.valid
            def u(x,t,s):return self.network(x,observed,pyramid,t,time_end=s,validate_times=False)*valid
            rows=None;metadata={'pyramid':pyramid,'observed':observed,'scene_lengths':None}
        else:
            scenes=[assemble_stage2_scene(r,self.policy,self.g,self.device,self.teacher,cache=None) for r in records]
            ids,condition,target,lengths=pack_scenes(scenes)
            pair=packed_training_pair(target,condition['valid_child_mask'],condition['batch'],source_generator)
            context=self.network.prepare_conditions(ids,**condition)
            valid=condition['valid_child_mask'];rows=condition['batch']
            # Warming computes only transient dynamic activations; the retained
            # topology contains static indices/maps, never previous-step features.
            end=(pair['time']+1)/2
            self.network.warm_topology(context,pair['state'],pair['time'],time_end=end)
            def u(x,t,s):return self.network.forward_prepared(context,x,t,time_end=s)*valid
            metadata={'ids':ids,'condition':condition,'context':context,'scene_lengths':lengths,
                'parent_ids_sha256':hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()}
        end,diagonal=sample_endpoints(pair['time'],interval_generator)
        if force_finite:end=(pair['time']+1)/2;diagonal=torch.zeros_like(end,dtype=torch.bool)
        return dict(u=u,pair=pair,end=end,diagonal=diagonal,valid=valid,rows=rows,metadata=metadata)

    def loss(self,batch,*,require_jvp=False):
        p=batch['pair'];m=self.cfg['meanflow']
        return meanflow_loss(batch['u'],p['state'],p['time'],batch['end'],p['velocity'],batch['valid'],
            row_to_scene=batch['rows'],model=self.network,adaptive_power=m['adaptive_power'],
            adaptive_epsilon=m['adaptive_epsilon'],skip_all_diagonal=not require_jvp)


def optimizer_for(network,cfg):
    t=cfg['training'];optimizer=torch.optim.AdamW(network.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    def schedule(i):
        return (i+1)/500 if i<500 else .5*(1+math.cos(math.pi*min(1.,(i-500)/9500)))
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)


@torch.no_grad()
def update_ema(ema,online,decay):
    src=online.state_dict()
    for name,value in ema.state_dict().items():
        if value.is_floating_point():value.mul_(decay).add_(src[name],alpha=1-decay)
        else:value.copy_(src[name])


def verify_resume(value,identity):
    if value.get('identity')!=identity:raise ValueError('Changed MeanFlow run/source/config/data identity')
    s=value['counts']
    if (s['exposures']!=2*s['updates'] or s['attempts']!=s['updates']+s['amp_skips']
        or s['cursor']!=2*s['attempts'] or value['scheduler']['last_epoch']!=s['updates']
        or s['diagonal_exposures']+s['finite_interval_exposures']!=s['exposures']):
        raise ValueError('Corrupt successful-exposure / RNG / scheduler continuity')
    if 'interval_rng' not in value:raise ValueError('Missing independent interval RNG')
    return value


def run_training(cfg,g,output,*,resume=None,max_updates=10000,smoke=False):
    setup_start=time.perf_counter()
    physical=int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'));gpu=admission(physical,True)
    if gpu['status']!='AVAILABLE':raise RuntimeError(json.dumps(gpu))
    if not 0<max_updates<=10000 or smoke and max_updates!=32:raise ValueError('Registered 32-smoke/20K maximum only')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if not resume and (out/'checkpoints').exists():raise FileExistsError('Preserve existing run; exact resume required')
    gate=json.loads(Path(cfg['engineering_report']).read_text())
    if any(gate.get(k)!=v for k,v in {'status':'GPU_PASS','arm':cfg['arm'],'source_sha256':source_sha(),'config_sha256':config_sha(cfg)}.items()):
        raise ValueError('Own exact native JVP/maxB2/replay GPU gate required')
    from .meanflow_replay_bounds import POLICY_SHA
    if (not gate.get('native_resume',{}).get('checkpoint_roundtrip_exact')
        or gate.get('full_jvp_vjp_dot',{}).get('normalized_error',float('inf'))>=3e-4):
        raise ValueError('Exact serialization/RNG and full native reverse/JVP comparison required')
    if gate.get('native_resume_exception'):
        if (gate.get('replay_policy_sha256')!=POLICY_SHA
            or gate.get('original_fm_native_replay',{}).get('bounds',{}).get('status')!='PASS'
            or gate.get('native_resume',{}).get('bounds',{}).get('status')!='PASS'
            or gate.get('resume_classification')!='BOUNDED_NATIVE_NUMERICAL_REPLAY_WITH_OWN_FM_CONTROL'):
            raise ValueError('Own same-GPU unmodified-FM control and versioned strict bounds required for Numerical Replay')
    elif not gate.get('native_resume',{}).get('end_to_end_exact'):
        raise ValueError('No implicit numerical replay exception')
    if not smoke:
        proof=json.loads(Path(cfg['smoke_summary']).read_text())
        if (proof.get('status')!='SMOKE_PASS' or proof['counts']['updates']!=32 or proof['counts']['exposures']!=64
            or proof['identity']['source_sha256']!=source_sha() or proof['identity']['config_sha256']!=config_sha(cfg)):
            raise ValueError('Own separate discarded 32-update Smoke required')
    torch.set_num_threads(4);device=torch.device('cuda');t=cfg['training']
    train,dev,manifests=datasets(cfg,g,verify=True,verified_receipt=cfg['verified_data_receipt'])
    seed_all(t['seed']);system=MeanFlowSystem(cfg,g,device);seed_all(t['seed'])
    network=system.network;ema=copy.deepcopy(network).eval().requires_grad_(False)
    optimizer,scheduler=optimizer_for(network,cfg);scaler=torch.cuda.amp.GradScaler(init_scale=t['amp_initial_scale'])
    source_gen=torch.Generator(device=device).manual_seed(t['seed']+1000)
    interval_gen=torch.Generator(device=device).manual_seed(t['seed']+2000)
    identity=dict(experiment=cfg['experiment'],arm=cfg['arm'],scope='discarded_engineering_smoke' if smoke else 'additional20k_meanflow',
        source_sha256=source_sha(),config_sha256=config_sha(cfg),geometry_sha256=g.sha256,**manifests,
        source_stage1_sha256=STAGE1_SHA,source_stage2_sha256=STAGE2_SHA,spec_sha256=cfg['specification']['sha256'],
        initialization_mode=system.initialization['initialization_mode'],output=str(out),batch=2,world_size=1,
        precision='FP32 parameters + original FP16 autocast',jvp_backend='torch.func.jvp',
        engineering_report_sha256=file_sha(cfg['engineering_report']))
    identity.update(resume_classification=gate['resume_classification'],native_resume_exception=gate['native_resume_exception'],
        replay_policy_sha256=gate.get('replay_policy_sha256'),
        native_jvp_adapters='original-primal SpConv + FlashAttention + native-argmax pooling; analytic tangent')
    counts=dict(updates=0,exposures=0,attempts=0,cursor=0,amp_skips=0,diagonal_exposures=0,finite_interval_exposures=0,jvp_calls=0,attempted_jvp_calls=0,consecutive_amp_skips=0)
    frozen=system.frozen_sha()
    if resume:
        value=verify_resume(torch.load(resume,map_location='cpu'),identity)
        network.load_state_dict(value['model'],strict=True);ema.load_state_dict(value['ema'],strict=True)
        optimizer.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler']);scaler.load_state_dict(value['scaler'])
        restore_rng(value['rng'],source_gen,device);interval_gen.set_state(value['interval_rng']);counts=value['counts']
        if value['frozen_state_sha256']!=frozen:raise ValueError('Frozen source drift on resume')
    atomic_json(out/'config.json',cfg);atomic_json(out/'initialization.json',system.initialization)
    atomic_json(out/'launch.json',dict(pid=os.getpid(),physical_gpu=physical,identity=identity,gpu=gpu,
        resume=str(resume) if resume else None,starting_updates=counts['updates'],epoch=time.time(),startup_seconds=time.perf_counter()-setup_start,
        trainable_parameter_names=[n for n,p in network.named_parameters() if p.requires_grad],
        frozen_teacher_parameter_names=[n for n,p in system.teacher.named_parameters() if not p.requires_grad],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    stopped=[False]
    def stop(signum,frame):stopped[0]=True
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def save():
        if system.frozen_sha()!=frozen:raise RuntimeError('Frozen model/state drift')
        atomic_checkpoint(out/'checkpoints/resume_latest.pt',dict(identity=identity,initialization=system.initialization,
            model=network.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),rng=rng_state(source_gen,device),interval_rng=interval_gen.get_state(),
            counts=counts,frozen_state_sha256=frozen))
    if not resume:save()
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();consecutive_skips=counts['consecutive_amp_skips']
    try:
        with (out/'train.jsonl').open('a',buffering=1) as log:
            while counts['updates']<max_updates and not stopped[0]:
                tick=time.perf_counter();indices=batch_indices(counts['cursor'],2,0,1,len(train),t['seed']);records=[train[i] for i in indices]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    batch=system.prepare(records,source_gen,interval_gen)
                    loss,detail=system.loss(batch)
                counts['attempted_jvp_calls']+=detail['jvp_calls']
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(network.parameters(),t['grad_clip'])
                # Nonfinite prediction/JVP/target raises before backward. Grad
                # overflows are recorded as failed AMP attempts, never exposures.
                finite=bool(torch.isfinite(norm) and norm>0)
                if any(p.grad is not None for n,p in system.teacher.named_parameters() if not p.requires_grad):
                    raise RuntimeError('Frozen teacher received parameter gradients')
                counts['attempts']+=1;counts['cursor']+=2
                if finite:
                    scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()<scale:raise RuntimeError('Scaler skipped despite finite audited gradients')
                    scheduler.step();update_ema(ema,network,t['ema_decay']);counts['updates']+=1;counts['exposures']+=2
                    diagonal=int(detail['diagonal'].sum());counts['diagonal_exposures']+=diagonal;counts['finite_interval_exposures']+=2-diagonal
                    counts['jvp_calls']+=detail['jvp_calls'];consecutive_skips=0;counts['consecutive_amp_skips']=0
                else:
                    counts['amp_skips']+=1;consecutive_skips+=1;counts['consecutive_amp_skips']=consecutive_skips;scaler.update(new_scale=scaler.get_scale()/2)
                d=detail['diagonal'];diag_mse=float(detail['per_scene'][d].mean()) if bool(d.any()) else None
                torch.cuda.synchronize()
                row=dict(**counts,finite=finite,loss=float(loss),diagonal_velocity_mse=diag_mse,
                    jvp_calls_this_attempt=detail['jvp_calls'],target_forward_calls=detail['jvp_calls'],prediction_forward_calls=1,topology_warm_forward_calls=1 if cfg['arm']=='S2-MF20K' else 0,gradient_norm=float(norm) if torch.isfinite(norm) else None,
                    seconds=time.perf_counter()-tick,lr=optimizer.param_groups[0]['lr'],amp_scale=scaler.get_scale(),
                    records=[{'sequence':r['sequence'],'frame':r['frame'],'index':i} for r,i in zip(records,indices)],
                    scene_lengths=batch['metadata']['scene_lengths'],parent_ids_sha256=batch['metadata'].get('parent_ids_sha256'),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(row,allow_nan=False)+'\n');print(f"{cfg['arm']} update={counts['updates']}/{max_updates} exposures={counts['exposures']} loss={float(loss):.6g} seconds={row['seconds']:.3f}",flush=True)
                del batch,detail,loss
                if consecutive_skips>=3 or counts['amp_skips']>max(2,int(counts['attempts']*.01)):
                    raise RuntimeError('AMP/gradient health gate failed; no automatic restart/precision substitution')
                if finite and (counts['updates'] in (32,100,500,10000) or counts['updates']%500==0):
                    save();atomic_json(out/f"health_{counts['updates']:06d}.json",dict(status='PASS',counts=counts,frozen_exact=True))
            save()
            status='STOPPED' if stopped[0] else 'SMOKE_PASS' if smoke else 'TRAINING_COMPLETE' if counts['updates']==10000 else 'HEALTH_PASS'
            atomic_json(out/'summary.json',dict(status=status,identity=identity,counts=counts,frozen_exact=True,
                elapsed_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='Smoke weights discarded; formal always reloads source' if smoke else 'Additional20K; no architecture expansion'))
    except Exception as exc:
        atomic_json(out/'failure.json',dict(status='BLOCKED_JVP' if ('forward AD' in str(exc) or 'forward-mode' in str(exc)) else 'FAILED_REVIEW_REQUIRED',
            reason=f'{type(exc).__name__}: {exc}',counts=counts,epoch=time.time()))
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)

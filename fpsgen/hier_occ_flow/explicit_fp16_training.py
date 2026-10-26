"""Scratch20K explicit FP16 training with separate FP32 Adam parameters and EMA."""
import copy,hashlib,json,math,os,signal,time,tempfile
from pathlib import Path
from types import MethodType
import torch
import yaml
from . import scratch20k_training as base
from . import queued20k_training as queued
from .explicit_fp16_training_policy import apply_explicit_policy,FP32Masters,update_fp32_ema
from .encoder_adaptation import tensor_state_sha
from .encoder_adaptation_runtime import config_sha,datasets
from .geometry import SparseHierarchyGeometry
from .trainer import seed_all,rng_state,restore_rng,atomic_checkpoint
from .storage import atomic_json,file_sha
from .resources import gpu_preflight
from .stage1_batch_transition import batch_indices

CURRENT=('S1-MeanFlow-Scratch20K','S2-FM-NoEncoder-Scratch20K')
ARMS=CURRENT+queued.ARMS


def own_sha():
    root=Path(__file__).resolve().parents[2];h=hashlib.sha256(queued.own_sha().encode())
    for p in sorted((root/'scripts/research_v2/explicit_fp16').glob('*.py')):h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text());g=SparseHierarchyGeometry(**cfg['geometry'])
    if cfg['experiment']!='desescene-explicit-fp16-training-v1' or cfg['arm'] not in ARMS or cfg['initialization_mode']!='scratch':raise ValueError('Independent explicit FP16 scratch identity')
    if cfg['network_precision']!='explicit_fp16_fp32_master' or cfg['normalization_statistics']!='fp32' or cfg['ema_storage']!='fp32':raise ValueError('Explicit network/master/norm/EMA policy required')
    shadow=copy.deepcopy(cfg);shadow['experiment']='desescene-scratch20k-v1' if cfg['arm'] in CURRENT else 'desescene-scratch20k-followups-v1'
    with tempfile.NamedTemporaryFile(mode='w',suffix='.yaml',dir='/tmp') as f:
        yaml.safe_dump(shadow,f);f.flush();(base.load_config if cfg['arm'] in CURRENT else queued.load_config)(f.name)
    return cfg,g


def admission(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):raise ValueError('Own physical GPU only')
    report=gpu_preflight((cfg['physical_gpu'],),minimum_free_gib=22.,allow_shared_compute=False)
    if report['status']!='AVAILABLE':raise RuntimeError(json.dumps(report))
    torch.cuda.set_per_process_memory_fraction(21.5*2**30/torch.cuda.get_device_properties(0).total_memory);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    return report


def velocity(self,x,observed,pyramid,time,step):
    if self.cfg['arm']=='S1-MeanFlow-Scratch20K':return self.stage1.flow(x,observed,pyramid,time,time_end=time+step,validate_times=False)
    return queued.QueueSystem.velocity(self,x,observed,pyramid,time,step)


def make_model(cfg,g,device):
    if cfg['arm'] in CURRENT:
        m=base.ScratchSystem(cfg,g,device);m.is_shortcut=False
    else:m=queued.QueueSystem(cfg,g,device)
    if m.is_s1:m.velocity=MethodType(velocity,m)
    return m


class ExplicitState:
    def __init__(self,cfg,g,device):
        seed_all(20261032);self.model=make_model(cfg,g,device);initial=base.cpu_state(self.model.state_dict())
        self.masters=FP32Masters(self.model,initial);self.precision=apply_explicit_policy(self.model)
        if hasattr(self.model,'frozen_condition'):
            holder=self.model.frozen_condition;source_sha=holder.audit['initial_state_sha256']
            holder.audit['explicit_fp16_policy']=apply_explicit_policy(holder.model)
            holder.audit['original_fp32_state_sha256']=source_sha;holder.audit['initial_state_sha256']=tensor_state_sha(holder.model.state_dict())
        self.model.initialization.update(network_precision=self.precision,master_parameter_dtype='torch.float32',ema_accumulation_dtype='torch.float32',
            original_initial_fp32_state_sha256=tensor_state_sha(initial),initial_state_sha256=tensor_state_sha(self.model.state_dict()),
            encoder_precision='explicit_fp16',flow_precision='explicit_fp16')
        self.ema=copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema_state={n:v.to(device) for n,v in initial.items()}
        if self.model.is_shortcut:object.__setattr__(self.model,'_shortcut_teacher',self.ema)
        groups=self.masters.optimizer_groups(self.model.groups());self.opt=torch.optim.AdamW(groups,weight_decay=.01)
        fn=lambda i:(i+1)/500 if i<500 else .5*(1+math.cos(math.pi*min(1.,(i-500)/9500)))
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,fn);self.scaler=torch.cuda.amp.GradScaler(init_scale=2048.)
        self.gen=torch.Generator(device=device).manual_seed(20262032);self.interval=torch.Generator(device=device).manual_seed(20263032)

    def snapshot(self):
        return base.cpu_state(dict(model=self.model.state_dict(),master_parameters=self.masters.state_dict(),ema=self.ema_state,
            optimizer=self.opt.state_dict(),scheduler=self.scheduler.state_dict(),scaler=self.scaler.state_dict(),
            rng=rng_state(self.gen,self.model.device),interval_rng=self.interval.get_state()))

    def restore(self,state):
        self.model.load_state_dict(state['model'],strict=True);self.masters.load_state_dict(state['master_parameters'])
        self.ema_state={n:v.to(self.model.device) for n,v in state['ema'].items()};self.ema.load_state_dict(self.ema_state,strict=True)
        self.opt.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler']);self.scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'],self.gen,self.model.device);self.interval.set_state(state['interval_rng'])
        for n,p in self.model.named_parameters():
            if not torch.equal(p,self.masters.params[n].to(p.dtype)):raise ValueError('Network/master rounding mismatch')
        if any(v.dtype!=torch.float32 for v in self.masters.params.values()):raise ValueError('Non-FP32 master')
        for s in self.opt.state.values():
            if any(torch.is_tensor(v) and v.is_floating_point() and v.dtype!=torch.float32 for v in s.values()):raise ValueError('Adam floating state must remain FP32')

    def update(self,records,*,force_finite=False,capture=False):
        self.model.zero_grad(set_to_none=True);self.opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):loss,d=self.model.loss(records,self.gen,self.interval,force_finite=force_finite)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite explicitFP16 loss')
        self.scaler.scale(loss).backward();self.masters.copy_gradients();self.scaler.unscale_(self.opt)
        norm=torch.nn.utils.clip_grad_norm_(self.masters.params.values(),1.);finite=bool(torch.isfinite(norm) and norm>0)
        prefixes={'encoder':'stage1.encoder.','flow':'stage1.flow.'} if self.model.is_s1 else {'stage2':'stage2.'}
        audits={}
        for k,prefix in prefixes.items():
            values=[p.grad.float().square().sum() for n,p in self.masters.params.items() if n.startswith(prefix) and p.grad is not None]
            v=torch.stack(values).sum().sqrt() if values else norm.new_zeros(())
            audits[k]=float(v) if torch.isfinite(v) else None
        teacher_forwards=2 if self.model.is_shortcut and bool((~d['diagonal']).any()) else 0
        captured=None
        if capture:
            captured=base.cpu_state(dict(prediction=d['prediction'],target=d['target'],gradient={n:p.grad for n,p in self.masters.params.items() if p.grad is not None},loss=float(loss),gradient_norm=float(norm) if torch.isfinite(norm) else None))
            if 'derivative' in d:captured['derivative']=base.cpu_state(d['derivative'])
        previous=self.scaler.get_scale()
        if finite:
            self.scaler.step(self.opt);self.scaler.update()
            if self.scaler.get_scale()<previous:raise RuntimeError('Hidden AMP skip')
            self.masters.copy_to_network();self.scheduler.step();update_fp32_ema(self.ema_state,self.ema,self.model,self.masters)
        else:self.scaler.update(new_scale=previous/2)
        row=dict(finite=finite,loss=float(loss),gradient_norm=float(norm) if torch.isfinite(norm) else None,gradient_modules=audits,amp_scale=self.scaler.get_scale(),
            scene_lengths=d['scene_lengths'],encoder_forwards=d['encoder_forwards'],jvp_calls=d['jvp_calls'],teacher_flow_forwards=teacher_forwards,
            parameter_dtype='torch.float16',network_gradient_dtype='torch.float16',master_gradient_dtype='torch.float32',optimizer_dtype='torch.float32')
        if self.model.is_s1:row.update(diagonal_exposures=int(d['diagonal'].sum()),finite_interval_exposures=int((~d['diagonal']).sum()),pyramid_shapes=d['pyramid_shapes'])
        if self.model.is_shortcut:
            row['interval_last_gradient_norm']=sum(float(p.grad.norm()) for n,p in self.masters.params.items() if '.embedding.interval.net.2.' in n and p.grad is not None)
        if 'parent_ids_sha256' in d:row['parent_ids_sha256']=d['parent_ids_sha256']
        del loss,d;torch.cuda.synchronize();return row,captured


def run(cfg,g,output,*,smoke=False,resume=None):
    gpu=admission(cfg);out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserve prior run')
    out.mkdir(parents=True,exist_ok=True);pin=own_sha();cpin=config_sha(cfg)
    gate=json.loads(Path(cfg['engineering_report']).read_text())
    if gate['status']!='GPU_PASS' or gate['source_sha256']!=pin or gate['config_sha256']!=cpin:raise ValueError('Own explicit FP16 maxB2/master/resume gate required')
    if not smoke:
        proof=json.loads(Path(cfg['smoke_summary']).read_text())
        if proof['status']!='SMOKE_PASS' or proof['counts']['updates']!=32 or proof['identity']['source_sha256']!=pin or proof['identity']['config_sha256']!=cpin:raise ValueError('Own discarded32 update explicit FP16 Smoke')
    train,_,manifests=datasets(cfg,g,verified_receipt=cfg['verified_data_receipt']);state=ExplicitState(cfg,g,'cuda');initial=state.model.initialization
    if initial!=gate['initialization']:raise ValueError('Initial network/master or frozen source changed')
    identity=dict(experiment=cfg['experiment'],arm=cfg['arm'],initialization_mode='scratch',scope='discarded_smoke' if smoke else 'total20k_from_random',
        source_sha256=pin,config_sha256=cpin,geometry_sha256=g.sha256,**manifests,seed=20261032,batch=2,physical_gpu=cfg['physical_gpu'],output=str(out.resolve()),
        network_precision='explicit_fp16_fp32_master',frozen_encoder_sha256=initial.get('frozen_encoder',{}).get('sha256'))
    counts=dict(updates=0,exposures=0,attempts=0,cursor=0,amp_skips=0,consecutive_amp_skips=0)
    if resume:
        saved=base.verify_resume(torch.load(resume,map_location='cpu'),identity);state.restore(saved);counts=saved['counts']
    atomic_json(out/'initialization.json',initial);atomic_json(out/'config.json',cfg)
    atomic_json(out/'launch.json',dict(pid=os.getpid(),epoch=time.time(),gpu=gpu,identity=identity,resume=str(resume) if resume else None))
    stopped=[False]
    def stop(*_):stopped[0]=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    def save():atomic_checkpoint(out/'checkpoints/resume_latest.pt',dict(**state.snapshot(),counts=counts,identity=identity,initialization=initial))
    save();limit=32 if smoke else 10000;start=time.time()
    try:
        with (out/'train.jsonl').open('a',buffering=1) as log:
            while counts['updates']<limit and not stopped[0]:
                if own_sha()!=pin:raise RuntimeError('Running explicit FP16 source changed')
                begin=time.perf_counter();ix=batch_indices(counts['cursor'],2,0,1,len(train),20261032);records=[train[i] for i in ix]
                row,_=state.update(records);counts['attempts']+=1;counts['cursor']+=2
                if row['finite']:counts['updates']+=1;counts['exposures']+=2;counts['consecutive_amp_skips']=0
                else:counts['amp_skips']+=1;counts['consecutive_amp_skips']+=1
                row.update(**counts,seconds=time.perf_counter()-begin,records=[dict(sequence=r['sequence'],frame=r['frame'],index=i) for r,i in zip(records,ix)],lr=[x['lr'] for x in state.opt.param_groups],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(row,allow_nan=False)+'\n');print(f"ExplicitFP16 {cfg['arm']} update={counts['updates']}/{limit} exposures={counts['exposures']} loss={row['loss']:.6g} seconds={row['seconds']:.3f}",flush=True)
                if counts['consecutive_amp_skips']>=3 or counts['amp_skips']>max(2,int(counts['attempts']*.01)):
                    save();raise RuntimeError('AMP finite/gradient health threshold failed')
                if row['finite'] and (counts['updates'] in (32,100,500) or counts['updates']%500==0):save();atomic_json(out/f"health_{counts['updates']:06d}.json",dict(status='PASS',counts=counts,gradient_modules=row['gradient_modules']))
            save();atomic_json(out/'summary.json',dict(status='STOPPED' if stopped[0] else 'SMOKE_PASS' if smoke else 'TRAINING_COMPLETE',identity=identity,initialization=initial,counts=counts,elapsed_seconds=time.time()-start))
    except Exception as exc:
        atomic_json(out/'failure.json',dict(status='BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',counts=counts));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)

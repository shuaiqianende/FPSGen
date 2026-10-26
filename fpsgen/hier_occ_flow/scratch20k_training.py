"""Independent random-initialized20K pilots. No pretrained loader or legacy-run mutation."""
import copy,hashlib,json,math,os,signal,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from .geometry import SparseHierarchyGeometry
from .stage1_occupancy import Stage1DirectOccupancy,ids_to_dense_occupancy,occupancy_flow_pair
from .meanflow_stage1 import MeanFlowStage1
from .meanflow_math import meanflow_loss,sample_endpoints
from .stage2_model import Stage2OccupancyModel
from .encoder_free_stage2 import assemble_oracle_encoder_free_scene
from .stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from .stage2_condition import scene_seed
from .stage1_batch_transition import batch_indices
from .encoder_adaptation_runtime import datasets,config_sha
from .encoder_adaptation import tensor_state_sha
from .trainer import seed_all,rng_state,restore_rng,atomic_checkpoint
from .storage import atomic_json,file_sha
from .dataset import input_xyz
from .resources import gpu_preflight
from .stage2_runtime import source_sha
from .stage1_monitoring import module_grad_norm


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text());g=SparseHierarchyGeometry(**cfg['geometry'])
    if cfg['experiment']!='desescene-scratch20k-v1' or cfg['arm'] not in ('S1-MeanFlow-Scratch20K','S2-FM-NoEncoder-Scratch20K'):
        raise ValueError('Independent scratch identity required')
    if cfg['initialization_mode']!='scratch' or any(k.startswith('source_stage') for k in cfg):
        raise ValueError('Pretrained sources forbidden in scratch config')
    if g.parent_resolution!=(384,384,32) or g.child_split!=(2,2,2) or g.origin!=(-50.,-50.,-4.) or g.maximum!=(50.,50.,4.4) or g.sphere_radius!=50.:
        raise ValueError('Original geometry required')
    t=cfg['training']
    expected=dict(successful_updates=10000,successful_frame_exposures=20000,batch_per_gpu=2,seed=20261032,
        precision='fp16',optimizer='AdamW',weight_decay=.01,warmup_updates=500,grad_clip=1.,ema_decay=.999,amp_initial_scale=2048.)
    if any(t.get(k)!=v for k,v in expected.items()):raise ValueError('Registered20K/B2/seed/AMP protocol changed')
    if cfg['validation']!={'split':'seq08_41','frames':list(range(0,4001,100)),'nfes':[1,2,4,8],'threshold':.5,'point_mode':'native_fine_centers'}:
        raise ValueError('Only frozen seq08/skip10041 evaluation')
    if cfg['physical_gpu']!=(2 if cfg['arm'].startswith('S1') else 3):raise ValueError('User assigned GPU2 S1/GPU3 S2')
    if t['lr_encoder']!=1e-4 or t['lr_flow']!=2e-4 or t['lr_stage2']!=2e-4:raise ValueError('Scratch learning rates locked')
    if cfg['model']!={'widths':[8,16,32,64],'stage2_width':64,'patch_sizes':[64,128],'checkpointing':True}:raise ValueError('Original full capacity')
    return cfg,g


def own_source_sha():
    root=Path(__file__).resolve().parents[2];h=hashlib.sha256(source_sha().encode())
    for folder in ('scripts/research_v2/scratch20k','fpsgen/voxel_point/vendor/voxdet','fpsgen/voxel_shortcut'):
        for p in sorted((root/folder).rglob('*.py')):h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def admission(cfg):
    gpu=cfg['physical_gpu']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Physical GPU mismatch')
    report=gpu_preflight((gpu,),minimum_free_gib=22.,allow_shared_compute=False)
    if report['status']!='AVAILABLE':raise RuntimeError(json.dumps(report))
    torch.cuda.set_per_process_memory_fraction(21.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(4)
    if cfg['arm'].startswith('S1'):
        # Fix convolution algorithm nondeterminism without changing model or replay bounds.
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    return report


class ScratchSystem(nn.Module):
    def __init__(self,cfg,g,device):
        super().__init__();self.cfg=cfg;self.g=g;self.device=torch.device(device)
        self.is_s1=cfg['arm'].startswith('S1')
        if self.is_s1:
            self.stage1=Stage1DirectOccupancy(g,objective='fm',widths=cfg['model']['widths'],activation_checkpointing=True)
            self.stage1.flow=MeanFlowStage1(self.stage1.flow)
        else:
            self.stage2=Stage2OccupancyModel(g,objective='fm',backbone='ptv3',width=64,
                condition_mode='observed',checkpointing=True,patch_sizes=(64,128))
        self.to(device).train().requires_grad_(True)
        self.initialization=dict(initialization_mode='scratch',pretrained_checkpoint_loaded=False,source_exposures=0,
            encoder_present=self.is_s1,all_network_parameters_trainable=all(p.requires_grad for p in self.parameters()),
            initial_state_sha256=tensor_state_sha(self.state_dict()),
            trainable_modules=['stage1.encoder','stage1.flow'] if self.is_s1 else ['stage2'],
            encoder_precision='fp32_island' if self.is_s1 else 'not_present',flow_precision='fp16_amp',parameters=sum(p.numel() for p in self.parameters()),condition='Partial LiDAR' if self.is_s1 else 'GT coarse+25%nearby empty + input Observed; neutral scores1; no384D')
        if not self.is_s1 and any('interval' in n or 'lidar' in n for n,p in self.named_parameters()):raise ValueError('No interval or384D branch in ordinary FM')

    def loss(self,records,generator,interval_generator,*,force_finite=False):
        if len(records)!=2:raise ValueError('True Batch2 required')
        xyz=[torch.from_numpy(input_xyz(Path(self.cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).to(self.device) for r in records]
        if self.is_s1:
            model=self.stage1
            target=torch.cat([ids_to_dense_occupancy(r['parent_ids'],self.g,device=self.device) for r in records])
            pair=occupancy_flow_pair(target,model.valid,generator=generator)
            with torch.autocast('cuda',enabled=False):
                pyramid=model.encoder(xyz)
            if not all(p.requires_grad for p in pyramid):raise RuntimeError('Encoder gradient disconnected')
            observed=torch.cat([model.geometry.observed_mask(x) for x in xyz])
            end,diagonal=sample_endpoints(pair['time'],interval_generator)
            if force_finite:end=(pair['time']+1)/2;diagonal=torch.zeros_like(end,dtype=torch.bool)
            def u(x,t,s):return model.flow(x,observed,pyramid,t,time_end=s,validate_times=False)*model.valid
            loss,detail=meanflow_loss(u,pair['state'],pair['time'],end,pair['velocity'],model.valid,model=model.flow)
            detail.update(scene_lengths=[len(r['parent_ids']) for r in records],diagonal=diagonal,
                pyramid_shapes=[list(p.shape) for p in pyramid],encoder_forwards=1)
        else:
            scenes=[assemble_oracle_encoder_free_scene(r,x,self.g,
                seed=scene_seed(r['sequence'],r['frame'],20261018),device=self.device,keep_observed=True,negative_fraction=.25) for r,x in zip(records,xyz)]
            ids,c,target,lengths=pack_scenes(scenes);pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],generator)
            prediction=self.stage2(ids,state=pair['state'],time=pair['time'],**c)
            loss=scene_mean_mse(prediction,pair['velocity'],c['valid_child_mask'],c['batch'])
            detail=dict(prediction=prediction,target=pair['velocity'],scene_lengths=lengths,encoder_forwards=0,jvp_calls=0,
                parent_ids_sha256=hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest())
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite scratch loss')
        return loss,detail

    def groups(self):
        t=self.cfg['training']
        if self.is_s1:return [{'params':self.stage1.encoder.parameters(),'lr':t['lr_encoder'],'name':'encoder'},
            {'params':self.stage1.flow.parameters(),'lr':t['lr_flow'],'name':'flow'}]
        return [{'params':self.stage2.parameters(),'lr':t['lr_stage2'],'name':'stage2'}]

    def grad_audit(self):
        modules={'encoder':self.stage1.encoder,'flow':self.stage1.flow} if self.is_s1 else {'stage2':self.stage2}
        return {k:module_grad_norm(v) for k,v in modules.items()}


def optimizer_for(model,cfg):
    opt=torch.optim.AdamW(model.groups(),weight_decay=.01)
    fn=lambda i:(i+1)/500 if i<500 else .5*(1+math.cos(math.pi*min(1.,(i-500)/9500)))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,fn)


@torch.no_grad()
def ema_update(ema,model):
    source=model.state_dict()
    for k,v in ema.state_dict().items():
        if v.is_floating_point():v.mul_(.999).add_(source[k],alpha=.001)
        else:v.copy_(source[k])


def cpu_state(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_state(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_state(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_state(v) for v in x)
    return copy.deepcopy(x)


def exact_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact_state(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


class RunState:
    def __init__(self,cfg,g,device):
        seed_all(20261032);self.model=ScratchSystem(cfg,g,device)
        self.ema=copy.deepcopy(self.model).eval().requires_grad_(False)
        self.opt,self.scheduler=optimizer_for(self.model,cfg);self.scaler=torch.cuda.amp.GradScaler(init_scale=2048.)
        self.gen=torch.Generator(device=device).manual_seed(20262032)
        self.interval=torch.Generator(device=device).manual_seed(20263032)
    def snapshot(self):return cpu_state(dict(model=self.model.state_dict(),ema=self.ema.state_dict(),optimizer=self.opt.state_dict(),scheduler=self.scheduler.state_dict(),
        scaler=self.scaler.state_dict(),rng=rng_state(self.gen,self.model.device),interval_rng=self.interval.get_state()))
    def restore(self,s):
        self.model.load_state_dict(s['model'],strict=True);self.ema.load_state_dict(s['ema'],strict=True)
        self.opt.load_state_dict(s['optimizer']);self.scheduler.load_state_dict(s['scheduler']);self.scaler.load_state_dict(s['scaler'])
        restore_rng(s['rng'],self.gen,self.model.device);self.interval.set_state(s['interval_rng'])
    def update(self,records,*,force_finite=False,capture=False):
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss,d=self.model.loss(records,self.gen,self.interval,force_finite=force_finite)
        self.scaler.scale(loss).backward();self.scaler.unscale_(self.opt)
        grad=self.model.grad_audit();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        finite=bool(torch.isfinite(norm) and norm>0)
        before=self.scaler.get_scale()
        if finite:
            self.scaler.step(self.opt);self.scaler.update();self.scheduler.step();ema_update(self.ema,self.model)
            if self.scaler.get_scale()<before:raise RuntimeError('Hidden AMP skip')
        else:self.scaler.update(new_scale=before/2)
        row=dict(loss=float(loss),gradient_norm=float(norm) if torch.isfinite(norm) else None,gradient_modules=grad,finite=finite,
            amp_scale=self.scaler.get_scale(),scene_lengths=d['scene_lengths'],encoder_forwards=d['encoder_forwards'],jvp_calls=d['jvp_calls'])
        if self.model.is_s1:row.update(diagonal_exposures=int(d['diagonal'].sum()),finite_interval_exposures=int((~d['diagonal']).sum()),pyramid_shapes=d['pyramid_shapes'])
        if 'parent_ids_sha256' in d:row['parent_ids_sha256']=d['parent_ids_sha256']
        captured=None
        if capture:captured=cpu_state(dict(prediction=d['prediction'],target=d['target'],loss=float(loss),gradient_norm=float(norm),
            gradient={n:p.grad for n,p in self.model.named_parameters() if p.grad is not None}))
        del loss,d;torch.cuda.synchronize()
        return row,captured


def verify_resume(saved,identity):
    if saved['identity']!=identity or saved['initialization']['initialization_mode']!='scratch' or saved['initialization']['pretrained_checkpoint_loaded']:
        raise ValueError('Resume must be own scratch identity; never adaptation')
    c=saved['counts']
    if c['exposures']!=2*c['updates'] or c['cursor']!=2*c['attempts'] or c['attempts']!=c['updates']+c['amp_skips'] or saved['scheduler']['last_epoch']!=c['updates']:
        raise ValueError('Corrupt successful budget/RNG/scheduler')
    return saved


def run(cfg,g,output,*,smoke=False,resume=None):
    admission_report=admission(cfg);out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserve previous run')
    out.mkdir(parents=True,exist_ok=True);pin=own_source_sha();cpin=config_sha(cfg)
    gate=json.loads(Path(cfg['engineering_report']).read_text())
    if gate['status']!='GPU_PASS' or gate['source_sha256']!=pin or gate['config_sha256']!=cpin:raise ValueError('Own scratch maximumB2 gate required')
    if not smoke:
        proof=json.loads(Path(cfg['smoke_summary']).read_text())
        if proof['status']!='SMOKE_PASS' or proof['counts']['updates']!=32 or proof['identity']['source_sha256']!=pin or proof['identity']['config_sha256']!=cpin:raise ValueError('Own discarded32-update scratch Smoke required')
    train,_,manifests=datasets(cfg,g,verified_receipt=cfg['verified_data_receipt'])
    state=RunState(cfg,g,'cuda');initial=state.model.initialization
    if initial['initial_state_sha256']!=gate['initialization']['initial_state_sha256']:raise ValueError('Fresh scratch initialization changed')
    identity=dict(experiment=cfg['experiment'],arm=cfg['arm'],initialization_mode='scratch',scope='discarded_smoke' if smoke else 'total20k_from_random',
        source_sha256=pin,config_sha256=cpin,geometry_sha256=g.sha256,**manifests,seed=20261032,batch=2,physical_gpu=cfg['physical_gpu'],output=str(out.resolve()))
    counts=dict(updates=0,exposures=0,attempts=0,cursor=0,amp_skips=0,consecutive_amp_skips=0)
    if resume:
        saved=verify_resume(torch.load(resume,map_location='cpu'),identity);state.restore(saved);counts=saved['counts']
    atomic_json(out/'initialization.json',initial);atomic_json(out/'config.json',cfg)
    atomic_json(out/'launch.json',dict(pid=os.getpid(),epoch=time.time(),gpu=admission_report,identity=identity,resume=str(resume) if resume else None,initialization=initial))
    stopped=[False]
    def stop(*_):stopped[0]=True
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def save():atomic_checkpoint(out/'checkpoints/resume_latest.pt',dict(**state.snapshot(),counts=counts,identity=identity,initialization=initial))
    limit=32 if smoke else 10000;save();start=time.time()
    try:
        with (out/'train.jsonl').open('a',buffering=1) as log:
            while counts['updates']<limit and not stopped[0]:
                if own_source_sha()!=pin:raise RuntimeError('Scratch source changed during training')
                begin=time.perf_counter();ix=batch_indices(counts['cursor'],2,0,1,len(train),20261032);records=[train[i] for i in ix]
                row,_=state.update(records);counts['attempts']+=1;counts['cursor']+=2
                if row['finite']:
                    counts['updates']+=1;counts['exposures']+=2;counts['consecutive_amp_skips']=0
                else:counts['amp_skips']+=1;counts['consecutive_amp_skips']+=1
                row.update(**counts,seconds=time.perf_counter()-begin,records=[dict(sequence=r['sequence'],frame=r['frame'],index=i) for r,i in zip(records,ix)],
                    lr=[x['lr'] for x in state.opt.param_groups],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(row,allow_nan=False)+'\n');print(f"{cfg['arm']} update={counts['updates']}/{limit} exposures={counts['exposures']} loss={row['loss']:.6g} seconds={row['seconds']:.3f}",flush=True)
                if counts['consecutive_amp_skips']>=3 or counts['amp_skips']>max(2,int(counts['attempts']*.01)):
                    save();raise RuntimeError('AMP health gate failed; no automatic restart')
                if counts['updates'] in (32,100,500) or row['finite'] and counts['updates']%500==0:
                    save();atomic_json(out/f"health_{counts['updates']:06d}.json",dict(status='PASS',counts=counts,gradient_modules=row['gradient_modules']))
            save();status='STOPPED' if stopped[0] else 'SMOKE_PASS' if smoke else 'TRAINING_COMPLETE'
            atomic_json(out/'summary.json',dict(status=status,identity=identity,initialization=initial,counts=counts,elapsed_seconds=time.time()-start,
                evaluation='seq08_skip10041_only',new_seed_or_100k=False))
    except Exception as exc:
        atomic_json(out/'failure.json',dict(status='BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',counts=counts));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)

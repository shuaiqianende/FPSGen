"""User-authorized followups: matched scratch FM, scratch Shortcut, frozen20K-Encoder S2."""
import copy,hashlib,json,math,os,signal,time
from pathlib import Path
import torch
from torch import nn
import yaml
from . import scratch20k_training as base
from .geometry import SparseHierarchyGeometry
from .stage1_occupancy import Stage1DirectOccupancy,ids_to_dense_occupancy,occupancy_flow_pair,masked_occ_loss
from .meanflow_stage1 import MeanFlowStage1
from .meanflow_interval import install_interval
from .meanflow_math import scene_squared_errors
from .meanflow_jvp import uncached_current_autocast
from .stage2_model import Stage2OccupancyModel
from .stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from .encoder_free_stage2 import assemble_oracle_encoder_free_scene
from .encoder_adaptation import tensor_state_sha
from .encoder_adaptation_runtime import config_sha,datasets
from .condition import sample_pyramid
from .stage2_condition import scene_seed
from .stage1_batch_transition import batch_indices
from .dataset import input_xyz
from .storage import atomic_json,file_sha
from .trainer import seed_all,atomic_checkpoint
from .resources import gpu_preflight

ARMS=('S1-FM-Scratch20K','S1-Shortcut-Scratch20K','S2-FM-Encoder-Scratch20K')


def own_sha():
    root=Path(__file__).resolve().parents[2];h=hashlib.sha256(base.own_source_sha().encode())
    for p in sorted((root/'scripts/research_v2/queued20k').glob('*.py')):h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text());g=SparseHierarchyGeometry(**cfg['geometry'])
    if cfg['experiment']!='desescene-scratch20k-followups-v1' or cfg['arm'] not in ARMS or cfg['initialization_mode']!='scratch':raise ValueError('Explicit registered scratch followup required')
    if any(k.startswith('source_stage') for k in cfg):raise ValueError('No mature source initialization')
    ref=copy.deepcopy(cfg);ref['experiment']='desescene-scratch20k-v1'
    ref['arm']='S1-MeanFlow-Scratch20K' if cfg['arm'].startswith('S1') else 'S2-FM-NoEncoder-Scratch20K'
    ref['physical_gpu']=2 if cfg['arm'].startswith('S1') else 3
    # Reuse strict non-source geometry/data/B2/20K/seed/optimizer/41-frame rules.
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w',suffix='.yaml',dir='/tmp') as f:
        yaml.safe_dump(ref,f);f.flush();base.load_config(f.name)
    if cfg['physical_gpu'] not in (2,3):raise ValueError('Only authorized GPU2/3')
    if cfg['arm']=='S1-FM-Scratch20K' and cfg['physical_gpu']!=3:raise ValueError('FM control assigned released GPU3')
    if cfg['arm']=='S1-Shortcut-Scratch20K':
        if cfg['physical_gpu']!=2 or cfg.get('shortcut')!={'fm_probability':.75,'half_steps':[.5,.25,.125,.0625,.03125,.015625,.0078125],'teacher':'ema','bootstrap':'two_half_steps','extra_losses':False}:
            raise ValueError('Registered d0 FM/two-half-step EMA Shortcut required')
    if cfg['arm']=='S2-FM-Encoder-Scratch20K':
        if cfg['physical_gpu']!=3 or cfg['frozen_encoder']['arm']!='S1-FM-Scratch20K' or cfg['frozen_encoder']['successful_frame_exposures']!=20000:
            raise ValueError('Only completed current scratch FM20K Encoder is allowed')
    return cfg,g


def admission(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):raise ValueError('Wrong physical GPU')
    report=gpu_preflight((cfg['physical_gpu'],),minimum_free_gib=22.,allow_shared_compute=False)
    if report['status']!='AVAILABLE':raise RuntimeError(json.dumps(report))
    torch.cuda.set_per_process_memory_fraction(21.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(4)
    if cfg['arm'].startswith('S1'):
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    return report


class StepConditionedStage1(nn.Module):
    def __init__(self,flow):
        super().__init__();self.base=flow;self.initialization_audit=install_interval(flow,'embedding',128,128)
    def forward(self,state,observed,pyramid,time,step_size):
        if time.shape!=step_size.shape or len(time)!=len(state) or not bool(((step_size>=0)&(time>=0)&(time+step_size<=1.000001)).all()):raise ValueError('Require0<=t,t+d<=1')
        with self.base.embedding.condition(step_size):return self.base._forward(state,observed,pyramid,time,validate_time_range=False)


def shortcut_pair(time,generator,half_steps,force=False):
    half=torch.tensor(half_steps,device=time.device,dtype=time.dtype)[torch.randint(len(half_steps),time.shape,device=time.device,generator=generator)]
    slots=torch.round((1-2*half)/half).long()+1
    discrete=torch.floor(torch.rand(time.shape,device=time.device,generator=generator)*slots).to(time.dtype)*half
    bootstrap=torch.rand(time.shape,device=time.device,generator=generator)>=.75
    if force:bootstrap=torch.arange(len(time),device=time.device)==len(time)-1
    return torch.where(bootstrap,discrete,time),torch.where(bootstrap,2*half,torch.zeros_like(half)),bootstrap,half


@torch.no_grad()
def shortcut_target(fn,state,time,half,valid):
    if bool(((half<=0)|(time+2*half>1.000001)).any()):raise ValueError('Illegal consistency interval')
    h=half.reshape(-1,*([1]*(state.ndim-1)))
    one=fn(state,time,half)*valid;middle=(state.float()+h*one.float())*valid
    two=fn(middle,time+half,half)*valid
    return (.5*(one.float()+two.float())).detach()*valid


def common_stage1_state(model):
    result={}
    for k,v in model.state_dict().items():
        if '.embedding.interval.' in k:continue
        result[k.replace('stage1.flow.base.','stage1.flow.')]=v
    return result


class FrozenEncoder:
    """Read-only shared holder, excluded from trainable/EMA state; pinned by source SHA."""
    def __init__(self,cfg,g,device):
        spec=cfg['frozen_encoder'];path=Path(spec['checkpoint']);saved=torch.load(path,map_location='cpu')
        identity=saved['identity'];counts=saved['counts'];init=saved['initialization']
        if identity['initialization_mode']!='scratch' or identity['arm']!=spec['arm'] or counts['updates']!=10000 or counts['exposures']!=20000 or init['pretrained_checkpoint_loaded']:
            raise ValueError('Mature/partial/unmatched Encoder is forbidden')
        if identity['config_sha256']!=spec['config_sha256'] or identity['geometry_sha256']!=g.sha256 or identity['seed']!=20261032 or identity['source_sha256']!=spec['source_sha256']:raise ValueError('Encoder dependency identity mismatch')
        for split in ('train','development'):
            if file_sha(Path(cfg[split+'_dataset'])/'manifest.json')!=identity[split+'_manifest_sha256']:raise ValueError('Frozen Encoder used another GT manifest')
        with torch.random.fork_rng(devices=[]):tmp=Stage1DirectOccupancy(g,objective='fm',widths=(8,16,32,64),activation_checkpointing=False)
        self.model=tmp.encoder;del tmp
        prefix='stage1.encoder.';weights={k[len(prefix):]:v for k,v in saved['ema'].items() if k.startswith(prefix)}
        self.model.load_state_dict(weights,strict=True);self.model=self.model.to(device).eval().requires_grad_(False)
        self.audit=dict(checkpoint=str(path),sha256=file_sha(path),identity=identity,source_exposures=20000,trainable=False,
            initial_state_sha256=tensor_state_sha(self.model.state_dict()),compute='FP32 eval/no_grad; no Flow forward')
    def __deepcopy__(self,memo):return self
    @torch.no_grad()
    def pyramid(self,xyz):
        with torch.autocast('cuda',enabled=False):return self.model(xyz)


class QueueSystem(base.ScratchSystem):
    def __init__(self,cfg,g,device):
        nn.Module.__init__(self);self.cfg=cfg;self.g=g;self.device=torch.device(device);self.is_s1=cfg['arm'].startswith('S1');self.is_shortcut='Shortcut' in cfg['arm']
        if self.is_s1:
            self.stage1=Stage1DirectOccupancy(g,objective='fm',widths=(8,16,32,64),activation_checkpointing=True)
            if self.is_shortcut:self.stage1.flow=StepConditionedStage1(self.stage1.flow)
        else:
            # Match every shared initial tensor to current no-Encoder scratch arm,
            # while the newly introduced LiDAR branch remains freshly random.
            with torch.random.fork_rng(devices=[]):
                seed_all(20261032);reference=Stage2OccupancyModel(g,objective='fm',backbone='ptv3',width=64,condition_mode='observed',checkpointing=True,patch_sizes=(64,128))
                reference_sha=tensor_state_sha({'stage2.'+k:v for k,v in reference.state_dict().items()})
                if reference_sha!=cfg['matched_no_encoder_initial_state_sha256']:raise ValueError('Random initialization does not match current no-Encoder arm')
                seed_all(20261032);self.stage2=Stage2OccupancyModel(g,objective='fm',backbone='ptv3',width=64,condition_mode='pyramid_observed',checkpointing=True,patch_sizes=(64,128))
                old=reference.state_dict();new=self.stage2.state_dict()
                with torch.no_grad():
                    for k,v in new.items():
                        if k.startswith('lidar.') or k=='fuse.0.weight':continue
                        v.copy_(old[k])
                    keep=list(range(48))+list(range(112,240));new['fuse.0.weight'][:,keep]=old['fuse.0.weight']
                self.shared_audit=dict(reference_initial_state_sha256=reference_sha,reference_is_fresh_random=True,source_exposures=0,shared_tensors_exact=all(torch.equal(v,old[k]) for k,v in new.items() if k in old and k!='fuse.0.weight'),fuse_common_columns_exact=torch.equal(new['fuse.0.weight'][:,keep],old['fuse.0.weight']),new_lidar_columns='fresh random; not pretrained')
            object.__setattr__(self,'frozen_condition',FrozenEncoder(cfg,g,device))
        self.to(device).train().requires_grad_(True)
        self.initialization=dict(initialization_mode='scratch',pretrained_checkpoint_loaded=False,source_exposures=0,
            initial_state_sha256=tensor_state_sha(self.state_dict()),parameters=sum(p.numel() for p in self.parameters()),all_network_parameters_trainable=all(p.requires_grad for p in self.parameters()),
            trainable_modules=['stage1.encoder','stage1.flow'] if self.is_s1 else ['stage2'],encoder_precision='fp32' if self.is_s1 else 'frozen scratch20K Encoder FP32',flow_precision='fp16_amp')
        if self.is_s1:self.initialization['common_stage1_state_sha256']=tensor_state_sha(common_stage1_state(self))
        else:self.initialization.update(shared_initialization=self.shared_audit,frozen_encoder=self.frozen_condition.audit)

    def velocity(self,x,observed,pyramid,time,step):
        if self.is_shortcut:return self.stage1.flow(x,observed,pyramid,time,step)
        return self.stage1.flow(x,observed,pyramid,time)

    def add_lidar(self,scenes,xyz):
        pyramid=self.frozen_condition.pyramid(xyz)
        result=[]
        for i,(ids,c,target,prov) in enumerate(scenes):
            centers=self.g.centers(self.g.unflatten(ids)).float()
            c['lidar_features']=sample_pyramid(tuple(p[i:i+1] for p in pyramid),centers,self.g.origin,self.g.maximum).detach().half()
            result.append((ids,c,target,prov))
        return result

    def loss(self,records,generator,interval_generator,*,force_finite=False):
        if len(records)!=2:raise ValueError('True Batch2 required')
        xyz=[torch.from_numpy(input_xyz(Path(self.cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).to(self.device) for r in records]
        if self.is_s1:
            valid=self.stage1.valid;target=torch.cat([ids_to_dense_occupancy(r['parent_ids'],self.g,device=self.device) for r in records])
            pair=occupancy_flow_pair(target,valid,generator=generator)
            with torch.autocast('cuda',enabled=False):pyramid=self.stage1.encoder(xyz)
            if not all(p.requires_grad for p in pyramid):raise RuntimeError('Encoder gradient lost')
            observed=torch.cat([self.stage1.geometry.observed_mask(x) for x in xyz]);bootstrap=torch.zeros(2,device=self.device,dtype=torch.bool)
            teacher_forwards=0
            if self.is_shortcut:
                t,step,bootstrap,half=shortcut_pair(pair['time'],interval_generator,self.cfg['shortcut']['half_steps'],force_finite)
                state=(1-t[:,None,None,None,None])*pair['source']+t[:,None,None,None,None]*pair['target']
                desired=pair['velocity'].clone()
                if bool(bootstrap.any()):
                    teacher=self._shortcut_teacher
                    select=torch.nonzero(bootstrap).flatten();points=[xyz[int(i)] for i in select]
                    # Detached teacher has its own EMA Encoder and no BN updates.
                    with torch.no_grad(),uncached_current_autocast('cuda'):
                        with torch.autocast('cuda',enabled=False):tp=teacher.stage1.encoder(points)
                        obs=observed[select]
                        desired[select]=shortcut_target(lambda x,a,d:teacher.velocity(x,obs,tp,a,d),state[select],t[select],half[select],valid)
                    teacher_forwards=2
            else:t=pair['time'];step=torch.zeros_like(t);state=pair['state'];desired=pair['velocity']
            prediction=self.velocity(state,observed,pyramid,t,step)*valid
            loss=masked_occ_loss(prediction,desired.detach(),valid,objective='fm')
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite Shortcut/FM loss')
            return loss,dict(prediction=prediction,target=desired.detach(),scene_lengths=[len(r['parent_ids']) for r in records],encoder_forwards=1,jvp_calls=0,
                diagonal=~bootstrap,pyramid_shapes=[list(p.shape) for p in pyramid],teacher_forwards=teacher_forwards)
        scenes=[assemble_oracle_encoder_free_scene(r,x,self.g,seed=scene_seed(r['sequence'],r['frame'],20261018),device=self.device,keep_observed=True,negative_fraction=.25) for r,x in zip(records,xyz)]
        ids,c,target,lengths=pack_scenes(self.add_lidar(scenes,xyz));pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],generator)
        prediction=self.stage2(ids,state=pair['state'],time=pair['time'],**c)
        loss=scene_mean_mse(prediction,pair['velocity'],c['valid_child_mask'],c['batch'])
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite S2 FM loss')
        return loss,dict(prediction=prediction,target=pair['velocity'],scene_lengths=lengths,encoder_forwards=1,jvp_calls=0,
            parent_ids_sha256=hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest())


class QueueState(base.RunState):
    def __init__(self,cfg,g,device):
        seed_all(20261032);self.model=QueueSystem(cfg,g,device);self.ema=copy.deepcopy(self.model).eval().requires_grad_(False)
        if self.model.is_shortcut:object.__setattr__(self.model,'_shortcut_teacher',self.ema)
        self.opt,self.scheduler=base.optimizer_for(self.model,cfg);self.scaler=torch.cuda.amp.GradScaler(init_scale=2048.)
        self.gen=torch.Generator(device=device).manual_seed(20262032);self.interval=torch.Generator(device=device).manual_seed(20263032)

    def update(self,records,*,force_finite=False,capture=False):
        row,detail=super().update(records,force_finite=force_finite,capture=capture)
        if self.model.is_shortcut:
            last=self.model.stage1.flow.base.embedding.interval.net[-1]
            row['interval_last_gradient_norm']=sum(float(p.grad.float().norm()) for p in last.parameters() if p.grad is not None)
        row['teacher_flow_forwards']=2 if self.model.is_shortcut and row.get('finite_interval_exposures',0)>0 else 0
        row['teacher_encoder_forwards']=int(row['teacher_flow_forwards']>0)
        row['repeated_teacher_passes_count_as_exposure']=False
        return row,detail


def run(cfg,g,output,*,smoke=False,resume=None):
    gpu=admission(cfg);out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserve previous run')
    out.mkdir(parents=True,exist_ok=True);pin=own_sha();cpin=config_sha(cfg)
    gate=json.loads(Path(cfg['engineering_report']).read_text())
    if gate['status']!='GPU_PASS' or gate['source_sha256']!=pin or gate['config_sha256']!=cpin:raise ValueError('Own queued-arm GPU gate required')
    if not smoke:
        proof=json.loads(Path(cfg['smoke_summary']).read_text())
        if proof['status']!='SMOKE_PASS' or proof['counts']['updates']!=32 or proof['identity']['source_sha256']!=pin or proof['identity']['config_sha256']!=cpin:raise ValueError('Own discarded Smoke32 required')
    train,_,manifests=datasets(cfg,g,verified_receipt=cfg['verified_data_receipt'])
    state=QueueState(cfg,g,'cuda');initial=state.model.initialization
    if initial!=gate['initialization']:raise ValueError('Scratch initialization/immutable Encoder source changed')
    identity=dict(experiment=cfg['experiment'],arm=cfg['arm'],initialization_mode='scratch',scope='discarded_smoke' if smoke else 'total20k_from_random',
        source_sha256=pin,config_sha256=cpin,geometry_sha256=g.sha256,**manifests,seed=20261032,batch=2,physical_gpu=cfg['physical_gpu'],output=str(out.resolve()),
        frozen_encoder_sha256=initial.get('frozen_encoder',{}).get('sha256'))
    counts=dict(updates=0,exposures=0,attempts=0,cursor=0,amp_skips=0,consecutive_amp_skips=0)
    if resume:
        saved=base.verify_resume(torch.load(resume,map_location='cpu'),identity);state.restore(saved);counts=saved['counts']
    atomic_json(out/'initialization.json',initial);atomic_json(out/'config.json',cfg)
    atomic_json(out/'launch.json',dict(pid=os.getpid(),epoch=time.time(),gpu=gpu,identity=identity,resume=str(resume) if resume else None,initialization=initial))
    stopped=[False]
    def stop(*_):stopped[0]=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    def save():atomic_checkpoint(out/'checkpoints/resume_latest.pt',dict(**state.snapshot(),counts=counts,identity=identity,initialization=initial))
    limit=32 if smoke else 10000;save();start=time.time()
    try:
        with (out/'train.jsonl').open('a',buffering=1) as log:
            while counts['updates']<limit and not stopped[0]:
                if own_sha()!=pin:raise RuntimeError('Pinned queued source changed')
                begin=time.perf_counter();ix=batch_indices(counts['cursor'],2,0,1,len(train),20261032);records=[train[i] for i in ix]
                row,_=state.update(records);counts['attempts']+=1;counts['cursor']+=2
                if row['finite']:counts['updates']+=1;counts['exposures']+=2;counts['consecutive_amp_skips']=0
                else:counts['amp_skips']+=1;counts['consecutive_amp_skips']+=1
                row.update(**counts,seconds=time.perf_counter()-begin,records=[dict(sequence=r['sequence'],frame=r['frame'],index=i) for r,i in zip(records,ix)],
                    lr=[x['lr'] for x in state.opt.param_groups],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(row,allow_nan=False)+'\n');print(f"{cfg['arm']} update={counts['updates']}/{limit} exposures={counts['exposures']} loss={row['loss']:.6g} seconds={row['seconds']:.3f}",flush=True)
                if counts['consecutive_amp_skips']>=3 or counts['amp_skips']>max(2,int(counts['attempts']*.01)):
                    save();raise RuntimeError('AMP health failure; no automatic retry')
                if row['finite'] and (counts['updates'] in (32,100,500) or counts['updates']%500==0):
                    save();atomic_json(out/f"health_{counts['updates']:06d}.json",dict(status='PASS',counts=counts,gradient_modules=row['gradient_modules']))
            save();status='STOPPED' if stopped[0] else 'SMOKE_PASS' if smoke else 'TRAINING_COMPLETE'
            atomic_json(out/'summary.json',dict(status=status,identity=identity,initialization=initial,counts=counts,elapsed_seconds=time.time()-start,evaluation='seq08_skip10041_only'))
    except Exception as exc:
        atomic_json(out/'failure.json',dict(status='BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',counts=counts));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)

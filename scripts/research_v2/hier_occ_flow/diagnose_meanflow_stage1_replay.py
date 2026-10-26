"""Own unmodified-FM control and MeanFlow native replay, no formal exposures.

Exact restoration is tested independently from native CUDA re-execution. All
numeric differences are recorded; this diagnostic never waives a training gate.
"""
from common import ROOT
import argparse,copy,gc,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.meanflow_training import load_config,MeanFlowSystem,admission,optimizer_for,update_ema
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config,datasets,step_loss,make_optimizer,config_sha
from fpsgen.hier_occ_flow.encoder_adaptation import load_sources,frozen_state_sha,update_adaptation_ema,tensor_state_sha
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.trainer import seed_all,rng_state,restore_rng
from verify_encoder_adaptation_gpu import stress_records
from verify_desescene_m1_gpu import cpu_state
from verify_stage1_flow_control import exact_state


def error(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:return dict(exact=False,shape_or_dtype_mismatch=True)
    if torch.equal(a,b):return dict(exact=True)
    d=a.double()-b.double();scale=torch.linalg.vector_norm(a.double())
    return dict(exact=False,max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),
        relative_l2=float(torch.linalg.vector_norm(d)/(scale+1e-30)),different_elements=int((a!=b).sum()),
        sign_flips=int(((a>0)!=(b>0)).sum()),shape=list(a.shape),dtype=str(a.dtype))


def differences(a,b,path=''):
    rows=[]
    if isinstance(a,torch.Tensor):
        e=error(a,b)
        if not e['exact']:rows.append(dict(path=path,**e))
    elif isinstance(a,dict):
        if a.keys()!=b.keys():return [dict(path=path,structure_changed=True)]
        for k in a:rows+=differences(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):return [dict(path=path,structure_changed=True)]
        for k,(x,y) in enumerate(zip(a,b)):rows+=differences(x,y,path+'/'+str(k))
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):rows.append(dict(path=path,non_tensor_changed=True))
    elif a!=b:rows.append(dict(path=path,non_tensor_changed=True,expected=a,actual=b))
    return rows


def canonical(name):
    for prefix in ('stage1.flow.','base.'):
        if name.startswith(prefix):return name[len(prefix):]
    return name


def audit(kind,cfg,g,records,out):
    seed=cfg['training']['seed'];seed_all(seed);device=torch.device('cuda');amp=cfg['training']['amp_initial_scale']
    source=torch.Generator(device=device).manual_seed(seed+1000)
    interval=torch.Generator(device=device).manual_seed(seed+2000)
    if kind=='original_fm':
        control,_=load_adaptation_config(cfg['control']['source_config']);system,_,initial=load_sources(control,device)
        model=system;ema=copy.deepcopy(model).eval().requires_grad_(False);opt,sched=make_optimizer(model,control['training'])
        frozen=lambda:frozen_state_sha(system)
        xyz=[torch.from_numpy(input_xyz(Path(cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).cuda() for r in records]
        def compute():
            seen=[]
            h=system.stage1.flow.register_forward_hook(lambda m,i,o:seen.append(o.detach().cpu().clone()) if not seen else None)
            try:loss,detail=step_loss(system,records,xyz,control,g,None,source)
            finally:h.remove()
            captures=dict(prediction=seen[0],target=detail['pair']['velocity'].detach().cpu().clone())
            hashes=dict(state=tensor_state_sha({'state':detail['pair']['state']}),time=tensor_state_sha({'t':detail['pair']['time']}),
                pyramid=tensor_state_sha({str(i):p for i,p in enumerate(detail['pyramid'])}))
            return loss,captures,hashes
        ema_update=lambda:update_adaptation_ema(ema,model,.999)
    else:
        system=MeanFlowSystem(cfg,g,device);model=system.network;initial=system.initialization
        ema=copy.deepcopy(model).eval().requires_grad_(False);opt,sched=optimizer_for(model,cfg);frozen=system.frozen_sha
        def compute():
            batch=system.prepare(records,source,interval,force_finite=True);loss,detail=system.loss(batch,require_jvp=True)
            captures={k:detail[k].detach().cpu().clone() for k in ('prediction','target','derivative')}
            hashes=dict(state=tensor_state_sha({'state':batch['pair']['state']}),time=tensor_state_sha({'t':batch['pair']['time']}),
                endpoint=tensor_state_sha({'s':batch['end']}),pyramid=tensor_state_sha({str(i):p for i,p in enumerate(batch['metadata']['pyramid'])}))
            return loss,captures,hashes
        ema_update=lambda:update_ema(ema,model,.999)
    seed_all(seed);frozen_initial=frozen();scaler=torch.cuda.amp.GradScaler(init_scale=amp)
    names={id(p):canonical(n) for n,p in model.named_parameters()}
    def named_optimizer():
        saved=cpu_state(opt.state_dict());mapping={}
        for live,group in zip(opt.param_groups,saved['param_groups']):
            mapping.update({i:names[id(p)] for i,p in zip(group['params'],live['params'])})
        saved['state']={mapping[k]:v for k,v in saved['state'].items()}
        return saved
    def snapshot():return cpu_state(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
        scaler=scaler.state_dict(),rng=rng_state(source,device),interval_rng=interval.get_state()))
    def restore(v):
        model.load_state_dict(v['model'],strict=True);ema.load_state_dict(v['ema'],strict=True);opt.load_state_dict(v['optimizer'])
        sched.load_state_dict(v['scheduler']);scaler.load_state_dict(v['scaler']);restore_rng(v['rng'],source,device);interval.set_state(v['interval_rng'])
    def update():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss,captures,hashes=compute()
        scaler.scale(loss).backward();scaler.unscale_(opt)
        unscaled={canonical(n):p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not bool(torch.isfinite(loss) and torch.isfinite(norm) and norm>0):raise RuntimeError('Nonfinite/zero maximum B2 gradient')
        gradients={canonical(n):p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        before=scaler.get_scale();scaler.step(opt);scaler.update();sched.step();ema_update()
        if scaler.get_scale()<before:raise RuntimeError('Maximum B2 AMP skip')
        if frozen()!=frozen_initial:raise RuntimeError('Frozen state drift')
        return dict(loss=float(loss),gradient_norm=float(norm),captures=captures,hashes=hashes,unscaled_gradients=unscaled,gradients=gradients,
            model={canonical(k):v for k,v in cpu_state(model.state_dict()).items()},ema={canonical(k):v for k,v in cpu_state(ema.state_dict()).items()},
            named_optimizer=named_optimizer(),other=cpu_state(dict(scheduler=sched.state_dict(),scaler=scaler.state_dict(),rng=rng_state(source,device),interval_rng=interval.get_state())))
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();first=update();del first
    saved=snapshot();out.mkdir(parents=True);torch.save(saved,out/'restore_point.pt')
    loaded=torch.load(out/'restore_point.pt',map_location='cpu');restore(loaded)
    roundtrip=exact_state(saved,snapshot())
    if not roundtrip:raise RuntimeError('Checkpoint serialization/RNG restoration not exact')
    reference=update();atomic_json(out/'progress.json',dict(status='REFERENCE_UPDATED',kind=kind))
    probes=[]
    for repeat in (1,2,3):
        restore(loaded)
        if not exact_state(saved,snapshot()):raise RuntimeError('Pre-replay restoration not exact')
        candidate=update();diff={}
        for key in ('captures','unscaled_gradients','gradients','model','ema','named_optimizer','other'):
            diff[key]=differences(reference[key],candidate[key])
        probes.append(dict(repeat=repeat,reference_loss=reference['loss'],replayed_loss=candidate['loss'],
            reference_gradient_norm=reference['gradient_norm'],replayed_gradient_norm=candidate['gradient_norm'],
            hashes_exact=reference['hashes']==candidate['hashes'],diff=diff))
        atomic_json(out/'progress.json',dict(status='REPLAYED',kind=kind,completed=repeat))
        del candidate
    report=dict(status='DIAGNOSED',kind=kind,source_sha256=source_sha(),config_sha256=config_sha(cfg),initialization=initial,
        checkpoint_roundtrip_exact=roundtrip,frozen_exact=True,replays=probes,seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32),
        engineering_only=True,formal_exposures=0,training_gate_waived=False)
    atomic_json(out/'result.json',report)
    del reference,saved,loaded,system,model,ema,opt,sched,scaler,source,interval
    gc.collect();torch.cuda.empty_cache();return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,g=load_config(a.config)
    if cfg['arm']!='S1-MF20K':raise ValueError('S1 only')
    if a.output.exists():raise FileExistsError('Preserve existing diagnostic')
    a.output.mkdir(parents=True);torch.set_num_threads(4);ad=admission(int(os.environ.get('CUDA_VISIBLE_DEVICES','-1')),True)
    if ad['status']!='AVAILABLE':raise RuntimeError(json.dumps(ad))
    report=dict(status='RUNNING',source_sha256=source_sha(),config_sha256=config_sha(cfg),admission=ad,formal_exposures=0)
    try:
        train,dev,manifests=datasets(cfg,g,verify=True,verified_receipt=cfg['verified_data_receipt'])
        records,selection=stress_records(train,dev,Path(cfg['stress']['p0_root']),Path(cfg['stress']['census']))
        report.update(manifests=manifests,selection=selection)
        for kind in ('original_fm','meanflow'):
            audit(kind,cfg,g,records,a.output/kind)
        report['status']='DIAGNOSED_BOTH'
    except Exception as exc:report.update(status='BLOCKED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:atomic_json(a.output/'result.json',report)

if __name__=='__main__':main()

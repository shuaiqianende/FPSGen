"""Real maximum Batch2 and exact checkpoint/state-update audit for Flow-only FM.

Fixed-gradient optimizer replay tests serialization independently of CUDA backward
reduction order. It does not certify bitwise end-to-end native CUDA training.
"""
from common import ROOT
import argparse,copy,json,os
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.encoder_adaptation import load_sources,frozen_state_sha,parameter_sha,update_adaptation_ema
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config,datasets,config_sha,step_loss,make_optimizer
from fpsgen.hier_occ_flow.stage1_monitoring import module_grad_norm
from fpsgen.hier_occ_flow.adaptation_gpu_slot import adaptation_admission
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.trainer import seed_all,rng_state,restore_rng
from fpsgen.hier_occ_flow.dataset import input_xyz
from verify_encoder_adaptation_gpu import stress_records
from verify_desescene_m1_gpu import cpu_state


def exact_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact_state(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact_state(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--p0-root',type=Path,required=True)
    p.add_argument('--census',type=Path,required=True);a=p.parse_args()
    cfg,g=load_adaptation_config(a.config)
    if cfg['arm']!='S1-FM20K' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Registered GPU0 Flow-only control required')
    admission=adaptation_admission(0,configure_allocator=True)
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report=dict(arm=cfg['arm'],source_sha256=source_sha(),config_sha256=config_sha(cfg),
        gpu_admission=admission,true_batch=2,no_token_truncation=True,
        resume_classification='EXACT_CHECKPOINT_AND_FIXED_GRADIENT_STATE_REPLAY_ONLY',
        native_end_to_end_cuda_replay='NOT_CERTIFIED',no_EF_numerical_exception_used=True)
    try:
        train,dev,manifests=datasets(cfg,g,verify=True,verified_receipt=cfg['verified_data_receipt'])
        records,selection=stress_records(train,dev,a.p0_root,a.census)
        seed_all(cfg['training']['seed']);model,_,initial=load_sources(cfg,torch.device('cuda'))
        seed_all(cfg['training']['seed']);ema=copy.deepcopy(model).eval().requires_grad_(False)
        xyz=[torch.from_numpy(input_xyz(Path(cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).cuda() for r in records]
        opt,sched=make_optimizer(model,cfg['training']);scaler=torch.cuda.amp.GradScaler(init_scale=cfg['training']['amp_initial_scale'])
        gen=torch.Generator(device='cuda').manual_seed(cfg['training']['seed']+1000)
        frozen=frozen_state_sha(model);flow_before=parameter_sha(model.stage1.flow)
        torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss,details=step_loss(model,records,xyz,cfg,g,None,gen)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        flow_grad=module_grad_norm(model.stage1.flow)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not bool(torch.isfinite(loss) and torch.isfinite(norm) and norm>0) or not np.isfinite(flow_grad) or flow_grad<=0:
            raise ValueError('Nonfinite/zero maximum real Batch2 Flow gradient')
        if any(p.grad is not None for p in model.stage1.encoder.parameters()):raise ValueError('Frozen Encoder has gradients')
        gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
        scale=scaler.get_scale();scaler.step(opt);scaler.update();sched.step();update_adaptation_ema(ema,model,.999)
        if scaler.get_scale()<scale:raise ValueError('Maximum Batch2 AMP overflow')
        def snapshot():return cpu_state(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),scaler=scaler.state_dict(),rng=rng_state(gen,torch.device('cuda'))))
        saved=snapshot();torch.save(saved,a.output/'checkpoint_roundtrip.pt')
        def restore():
            value=torch.load(a.output/'checkpoint_roundtrip.pt',map_location='cpu')
            model.load_state_dict(value['model'],strict=True);ema.load_state_dict(value['ema'],strict=True)
            opt.load_state_dict(value['optimizer']);sched.load_state_dict(value['scheduler']);scaler.load_state_dict(value['scaler'])
            restore_rng(value['rng'],gen,torch.device('cuda'))
        restore();roundtrip=exact_state(saved,snapshot())
        def fixed_gradient_step():
            opt.zero_grad(set_to_none=True)
            for name,p in model.named_parameters():
                if name in gradients:p.grad=gradients[name].to(device=p.device,dtype=p.dtype).clone()
            opt.step();sched.step();update_adaptation_ema(ema,model,.999)
            return snapshot()
        first=fixed_gradient_step();restore();second=fixed_gradient_step()
        checks=dict(maximum_real_batch2=len(records)==2,finite_nonzero_flow_gradient=flow_grad>0,
            encoder_gradients_absent=all(p.grad is None for p in model.stage1.encoder.parameters()),
            frozen_encoder_and_ema_exact=frozen_state_sha(model)==frozen_state_sha(ema)==frozen,
            flow_changed=parameter_sha(model.stage1.flow)!=flow_before,
            checkpoint_state_restored_exact=roundtrip,fixed_gradient_optimizer_ema_rng_replay_exact=exact_state(first,second))
        report.update(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,initialization=initial,
            manifests=manifests,stress_selection=selection,stress_census_coverage=dict(train=len(train),development=len(dev),complete=True),
            loss=float(loss),flow_gradient_norm=flow_grad,memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()),checkpoint_sha256=file_sha(a.output/'checkpoint_roundtrip.pt'))
    except torch.cuda.OutOfMemoryError as exc:report.update(status='BLOCKED',reason='Real maximum Batch2 OOM; no scene/token/batch reduction',detail=str(exc))
    except Exception as exc:report.update(status='FAIL',reason=f'{type(exc).__name__}: {exc}')
    atomic_json(a.output/'result.json',report)
    print(json.dumps({k:report.get(k) for k in ['status','arm','checks','reason','memory']},indent=2),flush=True)
    if report['status']!='PASS':raise RuntimeError('Flow-only engineering gate failed')


if __name__=='__main__':main()

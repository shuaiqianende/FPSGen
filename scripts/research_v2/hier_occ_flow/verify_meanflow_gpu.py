"""Actual native-JVP, original-output, max real B2 and full resume gate.

Unsupported custom forward AD is BLOCKED_JVP, never a zero/FD derivative.
No native numeric-replay exception is inherited from an earlier experiment.
"""
from common import ROOT
import argparse,copy,json,os,traceback,time
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.meanflow_training import load_config,MeanFlowSystem,admission,optimizer_for,update_ema
from fpsgen.hier_occ_flow.meanflow_jvp import endpoint_jvp,finite_difference_diagnostic
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import datasets,config_sha
from fpsgen.hier_occ_flow.encoder_adaptation import tensor_state_sha
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.storage import atomic_json
from fpsgen.hier_occ_flow.trainer import seed_all,rng_state,restore_rng
from verify_encoder_adaptation_gpu import stress_records
from verify_desescene_m1_gpu import cpu_state
from verify_stage1_flow_control import exact_state
from fpsgen.hier_occ_flow.meanflow_replay_bounds import compare_native_replay,POLICY_SHA


def errors(a,b):
    a=a.detach().float();b=b.detach().float();d=a-b
    return dict(exact=torch.equal(a,b),max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),
        relative_l2=float(torch.linalg.vector_norm(d)/(torch.linalg.vector_norm(b)+1e-12)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,g=load_config(a.config);a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    physical=int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'));gpu=admission(physical,True)
    report=dict(status='IMPLEMENTED',arm=cfg['arm'],source_sha256=source_sha(),config_sha256=config_sha(cfg),
        gpu_admission=gpu,true_batch=2,no_truncation=True,jvp_backend='torch.func.jvp',started=time.time(),
        native_resume_exception=False,precision='FP32 weights + original FP16 autocast')
    stage='admission'
    try:
        if gpu['status']!='AVAILABLE':raise RuntimeError('GPU not admitted; no launch on occupied resource')
        train,dev,manifests=datasets(cfg,g,verify=True,verified_receipt=cfg['verified_data_receipt'])
        records,selection=stress_records(train,dev,Path(cfg['stress']['p0_root']),Path(cfg['stress']['census']))
        report.update(manifests=manifests,stress_selection=selection,census_coverage=dict(train=len(train),development=len(dev)))
        seed_all(cfg['training']['seed']);system=MeanFlowSystem(cfg,g,'cuda');seed_all(cfg['training']['seed'])
        model=system.network;ema=copy.deepcopy(model).eval().requires_grad_(False)
        opt,sched=optimizer_for(model,cfg);scaler=torch.cuda.amp.GradScaler(init_scale=cfg['training']['amp_initial_scale'])
        source_gen=torch.Generator(device='cuda').manual_seed(cfg['training']['seed']+1000)
        interval_gen=torch.Generator(device='cuda').manual_seed(cfg['training']['seed']+2000)
        frozen=system.frozen_sha();original=tensor_state_sha(model.state_dict());torch.cuda.reset_peak_memory_stats()
        stage='maximum_real_batch2_static_conditions'
        with torch.autocast('cuda',dtype=torch.float16):batch=system.prepare(records,source_gen,interval_gen,force_finite=True)
        p0=batch['pair'];u=batch['u'];meta=batch['metadata']
        report['scene_lengths']=meta['scene_lengths'];report['initialization']=system.initialization
        if cfg['arm']=='S2-MF20K' and meta['scene_lengths']!=[r['teacher_parent_tokens'] for r in selection]:
            raise RuntimeError('Maximum-support census differs from actual frozen teacher; recensus required, no smaller-scene substitution')
        stage='source_and_finite_interval_zero_init_parity'
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            if cfg['arm']=='S1-MF20K':
                reference=model.base(p0['state'],meta['observed'],meta['pyramid'],p0['time'])*batch['valid']
            else:
                reference=model.base(meta['ids'],state=p0['state'],time=p0['time'],**meta['condition'])*batch['valid']
            parity=[]
            for end in (p0['time'],batch['end'],torch.ones_like(batch['end'])):
                parity.append(errors(u(p0['state'],p0['time'],end),reference))
        report['zero_init_parity']=parity
        if not all(row['exact'] for row in parity):raise RuntimeError('Step0 source/prepared outputs are not bitwise equal; adapter review required')
        del reference
        if cfg['arm']=='S1-MF20K':
            stage='cached_stage1_endpoint_propagation'
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                model.eval()
                try:
                    cached=model.prepare_inference_context(meta['observed'],meta['pyramid'])
                    value=model.forward_with_inference_context(p0['state'],cached,p0['time'],time_end=batch['end'])*batch['valid']
                    direct=u(p0['state'],p0['time'],batch['end'])
                    report['cached_inference_parity']=errors(value,direct)
                    if not torch.equal(value,direct):raise RuntimeError('Stage1 cached interval/source output mismatch')
                finally:model.train()
                del cached,value,direct
        if cfg['arm']=='S2-MF20K':
            stage='packed_permutation_and_scene_isolation'
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                ids=meta['ids'];condition=meta['condition'];x=p0['state'];t=p0['time'];end=batch['end']
                ordinary=model(ids,state=x,time=t,time_end=end,**condition)
                order=torch.randperm(len(ids),device=ids.device);inverse=torch.argsort(order)
                permuted=model(ids[order],state=x[order],time=t,time_end=end,**{k:v[order] for k,v in condition.items()})[inverse]
                if not torch.equal(ordinary,permuted):raise RuntimeError('Canonical Parent permutation mismatch')
                isolation=[]
                for scene in (0,1):
                    selected=condition['batch']==scene
                    c={k:v[selected] for k,v in condition.items() if k!='batch'}
                    one=model(ids[selected],state=x[selected],time=t[scene:scene+1],time_end=end[scene:scene+1],**c)
                    old=model.base(ids[selected],state=x[selected],time=t[scene:scene+1],**c)
                    if not torch.equal(one,old):raise RuntimeError('Single-scene source interval0-init mismatch')
                    isolation.append(errors(ordinary[selected],one))
                # The source network itself uses different GEMM row shapes
                # for packed/single calls. Record that numerical effect without
                # interpreting it as cross-scene information leakage.
                report['scene_isolation']=dict(permutation_exact=True,single_scene_source_exact=True,packed_vs_single=isolation)
                model.eval()
                try:
                    cached=model.prepare_inference_context(ids,**condition)
                    cached_output=model.forward_with_inference_context(cached,x,t,time_end=end)
                    report['cached_inference_parity']=errors(cached_output,ordinary)
                    if not torch.equal(cached_output,ordinary):raise RuntimeError('Cached inference changed source output')
                finally:model.train()
                del cached,cached_output,ordinary,permuted,one,old
        if cfg['arm']=='S2-MF20K':
            from fpsgen.hier_occ_flow.meanflow_component_probe import probe_stage2_components
            report['native_components']=probe_stage2_components(model,meta['context'])
            atomic_json(a.output/'component_probe.json',report['native_components'])
        stage='full_native_state_and_fixed_endpoint_jvp'
        with torch.autocast('cuda',dtype=torch.float16):
            primal,derivative=endpoint_jvp(u,p0['state'],p0['time'],batch['end'],p0['velocity'],model=model)
        if not bool(torch.isfinite(derivative).all() and derivative.abs().max()>0):raise RuntimeError('Missing/nonfinite/zero complete native derivative')
        report['jvp']=dict(finite=True,nonzero=True,max_abs=float(derivative.abs().max()),
            full_state_time_tangent=True,endpoint_tangent_zero=True,contains_full_dynamic_backbone=True)
        stage='full_native_jvp_vjp_dot_product'
        sx=p0['state'].detach().requires_grad_(True);st=p0['time'].detach().requires_grad_(True)
        ss=batch['end'].detach().requires_grad_(True);cotangent=torch.randn_like(primal)*.01
        with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):vy=u(sx,st,ss)
        vy.backward(cotangent)
        lhs=(derivative.double()*cotangent.double()).sum()
        rhs=(sx.grad.double()*p0['velocity'].double()).sum()+st.grad.double().sum()
        denom=(derivative.double().norm()*cotangent.double().norm()+sx.grad.double().norm()*p0['velocity'].double().norm()+st.grad.double().norm()*len(st)**.5)
        dot_error=float((lhs-rhs).abs()/(denom+1e-30))
        report['full_jvp_vjp_dot']=dict(normalized_error=dot_error,limit=3e-4,
            lhs=float(lhs),rhs=float(rhs),native_reverse_path=True,fixed_endpoint_tangent_zero=True)
        if not dot_error<3e-4:raise RuntimeError('Full native reverse / analytic forward derivative mismatch')
        del sx,st,ss,vy,cotangent
        # FD is only a diagnostic; AMP/max-pooling can have discontinuities.
        # Its error is disclosed, never substituted for the actual derivative.
        stage='central_finite_difference_diagnostic'
        epsilon=1e-3
        with torch.autocast('cuda',dtype=torch.float16):fd=finite_difference_diagnostic(u,p0['state'],p0['time'],batch['end'],p0['velocity'],epsilon)
        report['finite_difference_diagnostic']=dict(epsilon=epsilon,comparison=errors(derivative,fd),training_target='native_jvp_only')
        del primal,derivative,fd,batch,u,meta,p0
        torch.cuda.empty_cache()
        stage='full_maximum_batch2_backward_and_update'
        def snapshot():return cpu_state(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),scaler=scaler.state_dict(),rng=rng_state(source_gen,torch.device('cuda')),
            interval_rng=interval_gen.get_state()))
        def restore(value):
            model.load_state_dict(value['model'],strict=True);ema.load_state_dict(value['ema'],strict=True)
            opt.load_state_dict(value['optimizer']);sched.load_state_dict(value['scheduler']);scaler.load_state_dict(value['scaler'])
            restore_rng(value['rng'],source_gen,torch.device('cuda'));interval_gen.set_state(value['interval_rng'])
        def update(baseline=False):
            opt.zero_grad(set_to_none=True)
            time_audit=[]
            from fpsgen.hier_occ_flow.meanflow_interval import _ACTIVE_INTERVAL
            tm=model.base.embedding if cfg['arm']=='S1-MF20K' else model.base.time
            def time_hook(m,inputs,output):
                active=_ACTIVE_INTERVAL.get()
                time_audit.append(dict(module_id=id(m),active_id=active[0] if active else None,
                    grad_enabled=torch.is_grad_enabled(),output_requires_grad=output.requires_grad,
                    interval_matches=bool(active and active[0]==id(m))))
            handle=tm.register_forward_hook(time_hook)
            try:
                with torch.autocast('cuda',dtype=torch.float16):
                    batch=system.prepare(records,source_gen,interval_gen,force_finite=True)
                    if baseline:
                        p=batch['pair'];meta=batch['metadata']
                        if cfg['arm']=='S1-MF20K':prediction=model.base(p['state'],meta['observed'],meta['pyramid'],p['time'])*batch['valid']
                        else:prediction=model.base(meta['ids'],state=p['state'],time=p['time'],**meta['condition'])*batch['valid']
                        from fpsgen.hier_occ_flow.meanflow_math import scene_squared_errors
                        if cfg['arm']=='S1-MF20K':
                            from fpsgen.hier_occ_flow.stage1_occupancy import masked_occ_loss
                            loss=masked_occ_loss(prediction,p['velocity'],batch['valid'],objective='fm')
                        else:loss=scene_squared_errors(prediction,p['velocity'],batch['valid'],batch['rows']).mean()
                        details=dict(prediction=prediction,target=p['velocity'],derivative=torch.zeros_like(prediction),jvp_calls=0)
                    else:loss,details=system.loss(batch,require_jvp=True)
            finally:handle.remove()
            report['time_module_forward_audit']=time_audit
            scaler.scale(loss).backward();scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm) and norm>0):raise RuntimeError('Nonfinite/zero maximum B2 gradient')
            report['interval_gradient_audit']={n:dict(requires_grad=p.requires_grad,grad_none=p.grad is None,
                grad_max_abs=float(p.grad.abs().max()) if p.grad is not None else None,
                grad_finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None)
                for n,p in model.named_parameters() if '.interval.' in n}
            branch=[p.grad for n,p in model.named_parameters() if '.interval.net.2.' in n]
            if not baseline and (not branch or not any(v is not None and bool(torch.isfinite(v).all() and v.abs().max()>0) for v in branch)):
                raise RuntimeError('Interval final projection disconnected')
            if any(p.grad is not None for p in system.teacher.parameters() if not p.requires_grad):raise RuntimeError('Frozen gradient')
            captures=cpu_state({k:details[k] for k in ('prediction','target','derivative')})
            gradient=cpu_state({n:p.grad for n,p in model.named_parameters() if p.grad is not None})
            scale=scaler.get_scale();scaler.step(opt);scaler.update();sched.step();update_ema(ema,model,.999)
            if scaler.get_scale()<scale:raise RuntimeError('Maximum B2 AMP skip')
            return dict(loss=float(loss),gradient_norm=float(norm),jvp_calls=details['jvp_calls']),dict(gradient=gradient,**captures)
        def checked_pair(saved,baseline):
            left,left_capture=update(baseline);left_state=snapshot();restore(saved)
            if not exact_state(saved,snapshot()):raise RuntimeError('Pre-replay checkpoint/RNG restoration not exact')
            right,right_capture=update(baseline);right_state=snapshot()
            check=compare_native_replay(dict(**left_state,**left_capture,loss=left['loss'],gradient_norm=left['gradient_norm']),
                dict(**right_state,**right_capture,loss=right['loss'],gradient_norm=right['gradient_norm']))
            return left,right,exact_state(left_state,right_state),check
        stage='own_original_fm_native_replay_baseline'
        initial=snapshot();update(True);control_saved=snapshot()
        control_left,control_right,control_exact,control_check=checked_pair(control_saved,True)
        report['original_fm_native_replay']=dict(expected_update=control_left,replayed_update=control_right,
            exact=control_exact,bounds=control_check,same_source_same_gpu_same_maximum_batch2=True)
        atomic_json(a.output/'original_fm_replay.json',report['original_fm_native_replay'])
        if control_check['status']!='PASS':raise RuntimeError('Own unmodified-FM replay exceeds registered bounds')
        restore(initial)
        if not exact_state(initial,snapshot()):raise RuntimeError('Original source/RNG not restored after baseline diagnosis')
        del initial,control_saved
        stage='full_maximum_batch2_backward_and_update'
        first,_=update();saved=snapshot();torch.save(saved,a.output/'resume_audit.pt')
        restored=torch.load(a.output/'resume_audit.pt',map_location='cpu');restore(restored)
        if not exact_state(saved,snapshot()):raise RuntimeError('Checkpoint roundtrip state/RNG not exact')
        stage='native_end_to_end_resume_replay'
        second_update,replayed_update,replay_exact,replay_bounds=checked_pair(restored,False)
        report['native_resume']=dict(checkpoint_roundtrip_exact=True,end_to_end_exact=replay_exact,
            expected_update=second_update,replayed_update=replayed_update,
            numerical_exception_used=not replay_exact,bounds=replay_bounds)
        if replay_bounds['status']!='PASS':raise RuntimeError('Own MeanFlow native replay exceeds registered bounds')
        if system.frozen_sha()!=frozen or tensor_state_sha(model.state_dict())==original:raise RuntimeError('Frozen drift or no trainable update')
        report.update(status='GPU_PASS',max_batch2_backward=first,frozen_exact=True,
            memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()),
            environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
            native_resume_exception=not replay_exact,replay_policy_sha256=POLICY_SHA,
            resume_classification='EXACT_FULL_NATIVE_UPDATE_WITH_INTERVAL_RNG' if replay_exact else 'BOUNDED_NATIVE_NUMERICAL_REPLAY_WITH_OWN_FM_CONTROL')
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status='BLOCKED_MEMORY',reason=str(exc),stage=stage)
    except Exception as exc:
        message=f'{type(exc).__name__}: {exc}'
        jvp_stage='jvp' in stage
        report.update(status='BLOCKED_JVP' if jvp_stage else 'BLOCKED_ENGINEERING',reason=message,stage=stage,
            traceback=traceback.format_exc(),fallback_used=False)
    if torch.cuda.is_initialized():report.setdefault('memory',dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    report['finished']=time.time();atomic_json(a.output/'result.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','arm','stage','reason','memory')},indent=2),flush=True)
    if report['status']!='GPU_PASS':raise SystemExit(2)

if __name__=='__main__':main()

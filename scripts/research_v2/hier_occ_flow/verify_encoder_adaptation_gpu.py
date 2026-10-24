"""Maximum real B2, gradient/state audit and bounded replay; never shrink scenes."""
from common import ROOT
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
import torch

from fpsgen.hier_occ_flow.encoder_adaptation import load_sources,frozen_state_sha,parameter_sha,stage2_differentiable_batch,update_adaptation_ema
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config,datasets,config_sha,step_loss,make_optimizer
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage1_monitoring import module_grad_norm
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.trainer import seed_all,rng_state,restore_rng
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from verify_desescene_m1_gpu import cpu_state,tensor_differences


def flatten_state(value,prefix=''):
    tensors={};scalars={}
    def visit(v,path):
        if isinstance(v,torch.Tensor):tensors[path]=v.detach().cpu()
        elif isinstance(v,np.ndarray):tensors[path]=torch.from_numpy(v.astype(np.int64) if v.dtype.kind in 'ui' else v.copy())
        elif isinstance(v,dict):
            for k,x in v.items():visit(x,path+'/'+str(k))
        elif isinstance(v,(list,tuple)):
            for k,x in enumerate(v):visit(x,path+'/'+str(k))
        else:scalars[path]=v
    visit(value,prefix);return tensors,scalars


def stress_records(train,dev,p0root,census):
    count={}
    for arm in ('position','attention'):
        for line in (p0root/(arm+'_20k/train_rank0.jsonl')).read_text().splitlines():
            if not line.strip():continue
            for row in json.loads(line)['batch_records']:
                key=(row['sequence'],row['frame']);count[key]=max(count.get(key,0),row['tokens'])
    for line in census.read_text().splitlines():
        row=json.loads(line);key=(row['sequence'],row['frame']);count[key]=max(count.get(key,0),row['stage1_threshold_parent_count'])
    lookup={}
    for split,ds in (('train',train),('development',dev)):
        for i,row in enumerate(ds.index):lookup[(row['sequence'].decode(),int(row['frame']))]=(split,i)
    if set(lookup)-set(count):raise ValueError('Complete Train+Development support census required; do not substitute smaller seen scenes')
    selected=sorted(count,key=count.get,reverse=True)[:2]
    records=[(train if lookup[key][0]=='train' else dev)[lookup[key][1]] for key in selected]
    return records,[{'sequence':s,'frame':f,'teacher_parent_tokens':count[(s,f)],'split':lookup[(s,f)][0]} for s,f in selected]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--p0-root',type=Path,required=True);p.add_argument('--census',type=Path,required=True)
    a=p.parse_args();cfg,g=load_adaptation_config(a.config);physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in ('2','3'):raise ValueError('Single GPU2/3 only')
    admission=gpu_preflight((int(physical),))
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    base={'arm':cfg['arm'],'phase':cfg['phase'],'config_sha256':config_sha(cfg),'source_sha256':source_sha(),
          'gpu_admission':admission,'true_batch':2,'no_token_truncation':True}
    try:
        train,dev,manifest=datasets(cfg,g,verify=True);records,selection=stress_records(train,dev,a.p0_root,a.census)
        seed_all(cfg['training']['seed']);model,_,initial=load_sources(cfg,torch.device('cuda'))
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        initial_state=cpu_state(model.state_dict()) if cfg['arm']=='S2-E0' else None
        teacher=FrozenStage1OccupancyAdapter(**cfg['source_stage1']).cuda() if cfg['phase']=='P2' else None
        xyz=[torch.from_numpy(input_xyz(Path(cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).cuda() for r in records]
        opt,scheduler=make_optimizer(model,cfg['training']);scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
        gen=torch.Generator(device='cuda').manual_seed(cfg['training']['seed']+1000)
        frozen=frozen_state_sha(model);encoder_before=parameter_sha(model.stage1.encoder);teacher_sha=frozen_state_sha(model)
        def update():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss,details=step_loss(model,records,xyz,cfg,g,teacher,gen)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            norms={'encoder':module_grad_norm(model.stage1.encoder),'flow':module_grad_norm(model.stage1.flow),
                   'stage2':module_grad_norm(model.stage2) if model.stage2 is not None else 0.}
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm) or norm<=0:raise ValueError('Nonfinite/zero maximum-B2 gradient')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen gradient')
            scale=scaler.get_scale();scaler.step(opt);scaler.update();scheduler.step()
            if scaler.get_scale()<scale:raise ValueError('Maximum-B2 update AMP overflow; no hidden pass')
            update_adaptation_ema(ema,model,cfg['training']['ema_decay'])
            memory={'pyramid_bytes':sum(p.numel()*p.element_size() for p in details['pyramid']),
                'scene_tokens':details.get('scene_lengths'),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'ema_parameter_bytes':sum(p.numel()*p.element_size() for p in ema.parameters()),
                'module_parameter_bytes':{name:sum(p.numel()*p.element_size() for p in module.parameters())
                    for name,module in (('encoder',model.stage1.encoder),('flow',model.stage1.flow),('stage2',model.stage2)) if module is not None},
                'module_gradient_bytes':{name:sum(p.grad.numel()*p.grad.element_size() for p in module.parameters() if p.grad is not None)
                    for name,module in (('encoder',model.stage1.encoder),('flow',model.stage1.flow),('stage2',model.stage2)) if module is not None}}
            return float(loss),norms,memory
        torch.cuda.reset_peak_memory_stats();first,norms,memory=update()
        saved={'model':cpu_state(model.state_dict()),'ema':cpu_state(ema.state_dict()),'optimizer':cpu_state(opt.state_dict()),'scheduler':copy.deepcopy(scheduler.state_dict()),
               'scaler':copy.deepcopy(scaler.state_dict()),'rng':rng_state(gen,torch.device('cuda'))}
        second,_,_=update();reference=cpu_state(model.state_dict())
        expected_aux=cpu_state({'ema':ema.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'rng':rng_state(gen,torch.device('cuda'))})
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'],gen,torch.device('cuda'));replay,_,_=update()
        differences=tensor_differences(reference,model.state_dict())
        actual_aux={'ema':ema.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'rng':rng_state(gen,torch.device('cuda'))}
        expected_tensors,expected_scalars=flatten_state(expected_aux);actual_tensors,actual_scalars=flatten_state(actual_aux)
        auxiliary_differences=tensor_differences(expected_tensors,actual_tensors)
        aux_bounded=all(r['max_abs']<=1e-7 and r['relative_l2']<=1e-6 for r in auxiliary_differences)
        rng_exact=all(torch.equal(v,actual_tensors[k]) for k,v in expected_tensors.items() if k.startswith('/rng/'))
        # Pre-registered strict bounds; failures remain failures. A numerical
        # result is never relabeled bitwise. Frozen state must be exact.
        bounded=all(r['max_abs']<=1e-7 and r['relative_l2']<=1e-6 for r in differences)
        required=['encoder'] if cfg['arm'] in ('EF1','S2-E1') else ['encoder','flow'] if cfg['arm']=='EF2' else ['stage2'] if cfg['arm']=='S2-E0' else ['encoder','stage2']
        checks={'required_gradients':all(norms[k]>0 for k in required),'frozen_state_exact':frozen_state_sha(model)==frozen_state_sha(ema)==frozen,
                'encoder_changed':cfg['arm']=='S2-E0' or parameter_sha(model.stage1.encoder)!=encoder_before,
                'resume_loss_exact':second==replay,'resume_parameter_bound':bounded,'resume_auxiliary_bound':aux_bounded,
                'resume_scalars_exact':expected_scalars==actual_scalars,'resume_rng_exact':rng_exact,
                'finite':all(torch.isfinite(torch.tensor(v)) for v in (first,second,replay))}
        reuse=False
        if cfg['arm']=='S2-E0':
            model.load_state_dict(initial_state,strict=True)
            model.eval()
            # Batch2 eval representation versus the old frozen B1 reference.
            from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
            policy=json.loads(Path(cfg['reference_stage2_config']).read_text())
            scenes=[assemble_stage2_scene(r,policy,g,torch.device('cuda'),teacher,cache=None) for r in records]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                ids,c,target,lengths,pyr=stage2_differentiable_batch(model,records,xyz,cfg,g,teacher,teacher_scenes=scenes)
            old=torch.cat([s[1]['lidar_features'] for s in scenes]);new=c['lidar_features']
            from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
            old_ids,old_c,old_target,_=pack_scenes(scenes)
            pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],torch.Generator(device='cuda').manual_seed(cfg['training']['seed']+1000))
            model.stage2.train()
            gradients=[];velocities=[];all_finite=True
            for condition in (old_c,old_c,c):
                model.stage2.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    velocity=model.stage2(ids,state=pair['state'],time=pair['time'],**condition)
                    loss=scene_mean_mse(velocity,pair['velocity'],condition['valid_child_mask'],condition['batch'])
                (loss*16384).backward()
                grads={k:p.grad.detach().float().cpu()/16384 for k,p in model.stage2.named_parameters() if p.grad is not None}
                all_finite=all_finite and all(bool(torch.isfinite(v).all()) for v in grads.values())
                gradients.append(grads);velocities.append(velocity.detach().cpu())
            control=tensor_differences(gradients[0],gradients[1]);changed=tensor_differences(gradients[0],gradients[2])
            control_by_name={r['name']:r for r in control}
            grad_bounded=all(r['max_abs']<=max(1e-7,2*control_by_name.get(r['name'],{}).get('max_abs',0.))
                and r['relative_l2']<=max(1e-6,2*control_by_name.get(r['name'],{}).get('relative_l2',0.)) for r in changed)
            reuse=(torch.equal(old,new) and torch.equal(old_ids,ids) and torch.equal(old_target,target)
                   and torch.equal(velocities[0],velocities[2]) and all_finite and grad_bounded)
            base['b0_forward_gradient_protocol_comparison']={'features_exact':torch.equal(old,new),
                'max_feature_abs':float((new.float()-old.float()).abs().max()),
                'parent_ids_exact':torch.equal(old_ids,ids),'targets_exact':torch.equal(old_target,target),
                'velocity_exact':torch.equal(velocities[0],velocities[2]),'gradients_finite':all_finite,
                'gradient_differences':changed,'original_baseline_gradient_control':control,
                'source_noise_and_time':'same explicit packed pair; seed20261032+1000',
                'optimizer_schedule':'registered B0 stage2 AdamW2e-4/wd.01/warmup500/10K-cosine/EMA.999',
                'reuse_proven':reuse}
        base.update(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,initialization=initial,
            manifests=manifest,stress_selection=selection,memory=memory,gradient_norms=norms,
            stress_census_coverage={'train':len(train),'development':len(dev),'complete':True},
            losses=[first,second,replay],resume_parameter_differences=differences,
            resume_auxiliary_differences=auxiliary_differences,
            resume_classification='BITWISE' if not differences and not auxiliary_differences and expected_scalars==actual_scalars else 'NUMERICAL_REPLAY',
            replay_bounds={'max_abs':1e-7,'relative_l2':1e-6},
            b0_reuse_proven=reuse)
    except torch.cuda.OutOfMemoryError as exc:
        base.update(status='BLOCKED',reason='True maximum Batch2 OOM; no reduction/truncation attempted',detail=str(exc))
    except Exception as exc:
        base.update(status='FAIL',reason=f'{type(exc).__name__}: {exc}')
    atomic_json(a.output/'result.json',base);print(json.dumps({k:base.get(k) for k in ('status','arm','checks','reason')},indent=2))
    if base['status']!='PASS':raise RuntimeError('Engineering check did not pass; formal run prohibited')


if __name__=='__main__':main()

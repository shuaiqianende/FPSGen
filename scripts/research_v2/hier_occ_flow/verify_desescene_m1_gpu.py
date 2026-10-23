"""Real B2 M1 parity, gradients, ordering and audited resume checks; no long training."""
from common import ROOT
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from fpsgen.hier_occ_flow.desescene_diagnostics import error_stats
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_initialization import initialize_matched_finetune
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,source_sha,config_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def cpu_state(state):
    if isinstance(state,torch.Tensor):return state.detach().cpu().clone()
    if isinstance(state,dict):return {k:cpu_state(v) for k,v in state.items()}
    if isinstance(state,list):return [cpu_state(v) for v in state]
    return copy.deepcopy(state)


def tensor_differences(reference, candidate):
    """Preserve mismatch magnitudes instead of hiding a failed exact check."""
    rows=[]
    for key, expected in reference.items():
        actual=candidate[key].detach().cpu()
        if not torch.equal(expected,actual):
            rows.append({'name':key,'shape':list(expected.shape),
                         'different_elements':int((expected!=actual).sum()),
                         **error_stats(expected,actual)})
    return rows


def probe_resume(model, ids, cond, pair, output):
    """Compare uninterrupted and restored updates, including a B0 control.

    SpConv's implicit-GEMM embedding weight-gradient may reduce in a different
    order. Exactness is reported separately; it must never be inferred from
    matching losses. The numerical allowance below is restricted to that one
    tensor and requires unchanged evaluated endpoints and occupancy.
    """
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            velocity=model(ids,state=pair['state'],time=pair['time'],**cond)
            loss=scene_mean_mse(velocity,pair['velocity'],cond['valid_child_mask'],cond['batch'])
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        interaction=getattr(model,'interaction',None)
        gate_grads=[] if interaction is None else [float(p.grad) for p in interaction.scales]
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradients in maximum-scene B2')
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<before:raise ValueError('AMP skipped maximum-scene update')
        return float(loss),gate_grads
    first,gate_gradients=update()
    saved={'model':cpu_state(model.state_dict()),'optimizer':cpu_state(optimizer.state_dict()),
           'scaler':scaler.state_dict(),'cuda_rng':torch.cuda.get_rng_state(),'cpu_rng':torch.get_rng_state()}
    torch.save(saved,output)
    second,_=update();reference=cpu_state(model.state_dict())
    reference_gradients={k:cpu_state(p.grad) for k,p in model.named_parameters() if p.grad is not None}
    interaction=getattr(model,'interaction',None)
    branch_gradient=(None if interaction is None else
        any(p.grad is not None and bool((p.grad!=0).any()) for p in interaction.static_encoder.parameters()))
    saved=torch.load(output,map_location='cpu')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
    torch.cuda.set_rng_state(saved['cuda_rng']);torch.set_rng_state(saved['cpu_rng'])
    replay,_=update()
    parameters=tensor_differences(reference,model.state_dict())
    gradients=tensor_differences(reference_gradients,{k:p.grad for k,p in model.named_parameters() if p.grad is not None})
    replay_state=cpu_state(model.state_dict())
    model.eval()
    endpoints=[]
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    for weights in (reference,reference,replay_state):
        # Never load new parameter values inside one long-lived AMP weight
        # cache. Replay the same RNG for each independent inference rollout.
        model.load_state_dict(weights,strict=True)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            state=pair['source'].clone()
            for step in range(4):
                velocity=model(ids,state=state,time=torch.full((2,),step/4,device='cuda'),**cond)
                state=state+velocity.float()/4
            endpoints.append(state)
    # Declared before the independent B0 control is run; no blanket allclose.
    limits={'parameter_max_abs':1e-7,'gradient_max_abs':1e-7,'relative_l2':1e-6,
            'allowed_tensor':'trunk.net.embedding.stem.conv.weight'}
    bounded=(all(r['name']==limits['allowed_tensor'] and r['sign_flips']==0 and
                 r['max_abs']<=limits[kind+'_max_abs'] and r['relative_l2']<=limits['relative_l2']
                 for kind,rows in (('parameter',parameters),('gradient',gradients)) for r in rows))
    return {'losses':[first,second,replay],'loss_exact':second==replay,
            'parameters_exact':not parameters,'gradients_exact':not gradients,
            'parameter_differences':parameters,'gradient_differences':gradients,
            'repeated_forward_exact':torch.equal(endpoints[0],endpoints[1]),
            'repeated_forward_error':error_stats(endpoints[0],endpoints[1]),
            'endpoint_exact':torch.equal(endpoints[0],endpoints[2]),'endpoint_error':error_stats(endpoints[0],endpoints[2]),
            'occupancy_exact':torch.equal(endpoints[0]>0,endpoints[2]>0),
            'bounded_embedding_wgrad_only':bounded,'limits':limits,
            'gate_gradients_finite_nonzero':bool(gate_gradients) and all(np.isfinite(gate_gradients)) and any(abs(x)>0 for x in gate_gradients),
            'interaction_gradient_nonzero':branch_gradient}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--b0-config',type=Path,required=True);p.add_argument('--m1-config',type=Path,required=True)
    p.add_argument('--stress-source',type=Path,required=True,help='D3 per_frame.jsonl with NFE2 threshold counts')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in ('2','3'):raise ValueError('One physical GPU2/3 required')
    admission=gpu_preflight((int(physical),))
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    cfg,g=load_stage2_config(a.b0_config);m1cfg,m1g=load_stage2_config(a.m1_config)
    if g!=m1g or cfg['initialization']!=m1cfg['initialization']:raise ValueError('Source weights/geometry differ')
    _,dev,manifests=dataset_identity(cfg,g,verify=True)
    records=[json.loads(x) for x in a.stress_source.read_text().splitlines() if x.strip()]
    if len(records)!=900:raise ValueError('Use complete 900-frame support census for maximum-scene test')
    ranked=sorted(records,key=lambda r:r['stage1_threshold_parent_count'],reverse=True)[:2]
    mapping={(str(r['sequence'].decode()),int(r['frame'])):i for i,r in enumerate(dev.index)}
    indices=[mapping[(r['sequence'],int(r['frame']))] for r in ranked]
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda().eval()
    scenes=[assemble_stage2_scene(dev[index],cfg,g,torch.device('cuda'),adapter) for index in indices]
    ids,cond,target,lengths=pack_scenes(scenes)
    if len(ids)>cfg['training']['max_batch_parent_tokens']:raise ValueError('Stress batch exceeds token contract')
    del adapter;torch.cuda.empty_cache()
    b0=Stage2OccupancyModel(g,**cfg['model']).cuda().eval()
    m1=Stage2OccupancyModel(g,**m1cfg['model']).cuda().eval()
    init0=initialize_matched_finetune(b0,cfg,g,manifests)
    init1=initialize_matched_finetune(m1,m1cfg,g,manifests)
    generator=torch.Generator(device='cuda').manual_seed(cfg['training']['seed'])
    pair=packed_training_pair(target,cond['valid_child_mask'],cond['batch'],generator)
    checks={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        reference=b0(ids,state=pair['state'],time=pair['time'],**cond)
        candidate=m1(ids,state=pair['state'],time=pair['time'],**cond)
        checks['zero_parity']={'exact':torch.equal(reference,candidate),**error_stats(reference,candidate)}
        permutation=torch.randperm(len(ids),device='cuda',generator=generator)
        other_cond={k:v[permutation] for k,v in cond.items()}
        permuted=m1(ids[permutation],state=pair['state'][permutation],time=pair['time'],**other_cond)
        checks['sorting_exact']=torch.equal(candidate[permutation],permuted)
        context=m1.prepare_inference_context(ids,**cond)
        cached=m1(ids,state=pair['state'],time=pair['time'],inference_context=context)
        checks['cached_exact']=torch.equal(candidate,cached)
        endpoints=[]
        for model in (b0,m1):
            state=pair['source'].clone()
            for step in range(4):
                velocity=model(ids,state=state,time=torch.full((2,),step/4,device='cuda'),**cond)
                state=state+velocity.float()/4
            endpoints.append(state)
        checks['endpoint_exact']=torch.equal(*endpoints)
        checks['occupancy_exact']=torch.equal(endpoints[0]>0,endpoints[1]>0)
    del b0,reference,candidate,permuted,cached,context,endpoints
    torch.cuda.empty_cache()
    m1_resume=probe_resume(m1,ids,cond,pair,a.output/'resume_probe.pt')
    del m1;torch.cuda.empty_cache()
    b0=Stage2OccupancyModel(g,**cfg['model']).cuda().eval()
    initialize_matched_finetune(b0,cfg,g,manifests)
    b0_resume=probe_resume(b0,ids,cond,pair,a.output/'b0_resume_probe.pt')
    checks['gate_gradients_finite_nonzero']=m1_resume['gate_gradients_finite_nonzero']
    checks['interaction_gradient_nonzero']=m1_resume['interaction_gradient_nonzero']
    for name,resume in (('m1',m1_resume),('b0',b0_resume)):
        checks[name+'_resume_loss_exact']=resume['loss_exact']
        checks[name+'_repeated_forward_exact']=resume['repeated_forward_exact']
        checks[name+'_resume_bounded_embedding_only']=resume['bounded_embedding_wgrad_only']
    checks['resume_control_explains_nonexact']=(m1_resume['parameters_exact'] or not b0_resume['gradients_exact'])
    checks['resume_endpoint_within_baseline_control']=(
        all(m1_resume['endpoint_error'][k]<=2*b0_resume['endpoint_error'][k]+1e-7
            for k in ('max_abs','mean_abs','relative_l2')) and
        m1_resume['endpoint_error']['sign_flips']<=2*b0_resume['endpoint_error']['sign_flips'])
    passed=checks['zero_parity']['exact'] and all(v is True for k,v in checks.items() if k!='zero_parity')
    result={'status':'PASS' if passed else 'FAIL','checks':checks,'source_sha256':source_sha(),
            'b0_config_sha256':config_sha(cfg),'m1_config_sha256':config_sha(m1cfg),
            'source_initialization':init0,'m1_initialization':init1,'gpu_admission':admission,
            'stress_source_sha256':file_sha(a.stress_source),'stress_indices':indices,'scene_tokens':lengths,
            'stress_scope':'largest two threshold-predicted Parent counts in the fixed 900-frame NFE2 census; not every possible training batch',
            'resume_diagnostics':{'m1':m1_resume,'baseline_control':b0_resume},
            'resume_equivalence':'EXACT' if m1_resume['parameters_exact'] else 'NUMERICAL_EMBEDDING_WGRAD',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    atomic_json(a.output/'result.json',result)
    if not passed:raise RuntimeError('M1 GPU correctness gate failed; see result.json')
    print(json.dumps({'status':result['status'],'checks':checks,'scene_tokens':lengths},indent=2))


if __name__=='__main__':
    main()

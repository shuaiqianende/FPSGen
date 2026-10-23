"""Real B2 M1 parity, gradients, ordering and exact local resume checks; no long training."""
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
    m1.train()
    optimizer=torch.optim.AdamW(m1.parameters(),lr=2e-4,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            velocity=m1(ids,state=pair['state'],time=pair['time'],**cond)
            loss=scene_mean_mse(velocity,pair['velocity'],cond['valid_child_mask'],cond['batch'])
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        gate_grads=[float(p.grad) for p in m1.interaction.scales]
        norm=torch.nn.utils.clip_grad_norm_(m1.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradients in maximum-scene B2')
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<before:raise ValueError('AMP skipped maximum-scene update')
        return float(loss),gate_grads
    first,gradients=update()
    checks['gate_gradients_finite_nonzero']=all(np.isfinite(gradients)) and any(abs(x)>0 for x in gradients)
    saved={'model':cpu_state(m1.state_dict()),'optimizer':cpu_state(optimizer.state_dict()),
           'scaler':scaler.state_dict(),'cuda_rng':torch.cuda.get_rng_state(),'cpu_rng':torch.get_rng_state()}
    torch.save(saved,a.output/'resume_probe.pt')
    second,_=update();reference=cpu_state(m1.state_dict())
    checks['interaction_gradient_nonzero']=any(p.grad is not None and bool((p.grad!=0).any()) for p in m1.interaction.static_encoder.parameters())
    saved=torch.load(a.output/'resume_probe.pt',map_location='cpu')
    m1.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
    torch.cuda.set_rng_state(saved['cuda_rng']);torch.set_rng_state(saved['cpu_rng'])
    replay,_=update()
    checks['resume_loss_exact']=second==replay
    checks['resume_parameters_exact']=all(torch.equal(v.cpu(),reference[k]) for k,v in m1.state_dict().items())
    passed=checks['zero_parity']['exact'] and all(v is True for k,v in checks.items() if k!='zero_parity')
    result={'status':'PASS' if passed else 'FAIL','checks':checks,'source_sha256':source_sha(),
            'b0_config_sha256':config_sha(cfg),'m1_config_sha256':config_sha(m1cfg),
            'source_initialization':init0,'m1_initialization':init1,'gpu_admission':admission,
            'stress_source_sha256':file_sha(a.stress_source),'stress_indices':indices,'scene_tokens':lengths,
            'stress_scope':'largest two threshold-predicted Parent counts in the fixed 900-frame NFE2 census; not every possible training batch',
            'losses':[first,second,replay],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    atomic_json(a.output/'result.json',result)
    if not passed:raise RuntimeError('M1 GPU correctness gate failed; see result.json')
    print(json.dumps({'status':result['status'],'checks':checks,'scene_tokens':lengths},indent=2))


if __name__=='__main__':
    main()

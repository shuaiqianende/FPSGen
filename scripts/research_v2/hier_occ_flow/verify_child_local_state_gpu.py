"""Actual maximum-support B2 checks for the local XYZ/attention pilot pair."""
from common import ROOT
import argparse
import json
import os
import time
from pathlib import Path
import torch

from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_initialization import initialize_matched_finetune
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,source_sha,config_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from verify_desescene_m1_gpu import probe_resume


def endpoint(model,ids,cond,source):
    state=source.clone()
    for step in range(4):
        state=state+model(ids,state=state,time=torch.full((2,),step/4,device='cuda'),**cond).float()/4
    return state


def timing(function):
    for _ in range(3):value=function()
    torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    tick=time.perf_counter();start.record()
    for _ in range(3):value=function()
    end.record();torch.cuda.synchronize()
    return {'cuda_ms':start.elapsed_time(end)/3,'wall_ms':(time.perf_counter()-tick)*1000/3,
            'repetitions':3,'scope':'diagnostic one-step max-scene Batch2, not full generation'}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline-config',type=Path,required=True)
    p.add_argument('--configs',type=Path,nargs=2,required=True)
    p.add_argument('--stress-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in ('1','3'):raise ValueError('Use one of the currently authorized GPU1/3')
    admission=gpu_preflight((int(physical),))
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    cfg,g=load_stage2_config(a.baseline_config);_,dev,manifests=dataset_identity(cfg,g,verify=True)
    torch.manual_seed(cfg['training']['seed'])
    records=[json.loads(x) for x in a.stress_source.read_text().splitlines() if x.strip()]
    if len(records)!=900:raise ValueError('Require the complete NFE2 support census')
    records=sorted(records,key=lambda x:x['stage1_threshold_parent_count'],reverse=True)[:2]
    indices={(str(x['sequence'].decode()),int(x['frame'])):i for i,x in enumerate(dev.index)}
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda().eval()
    scenes=[assemble_stage2_scene(dev[indices[(x['sequence'],x['frame'])]],cfg,g,torch.device('cuda'),adapter) for x in records]
    ids,cond,target,lengths=pack_scenes(scenes)
    del adapter,scenes;torch.cuda.empty_cache()
    generator=torch.Generator(device='cuda').manual_seed(cfg['training']['seed']+1000)
    pair=packed_training_pair(target,cond['valid_child_mask'],cond['batch'],generator)
    baseline=Stage2OccupancyModel(g,**cfg['model']).cuda().eval()
    initialization=initialize_matched_finetune(baseline,cfg,g,manifests)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        reference=baseline(ids,state=pair['state'],time=pair['time'],**cond).detach()
        reference_endpoint=endpoint(baseline,ids,cond,pair['source'])
        baseline_timing=timing(lambda:baseline(ids,state=pair['state'],time=pair['time'],**cond))
    baseline_resume=probe_resume(baseline,ids,cond,pair,a.output/'b0_resume_probe.pt')
    del baseline;torch.cuda.empty_cache()
    results={}
    for path in a.configs:
        current,otherg=load_stage2_config(path)
        if otherg!=g or current['initialization']!=cfg['initialization']:raise ValueError('Source/geometry changed')
        torch.manual_seed(cfg['training']['seed'])
        model=Stage2OccupancyModel(g,**current['model']).cuda().eval()
        init=initialize_matched_finetune(model,current,g,manifests)
        mode=current['model']['local_state_encoding'];checks={}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            value=model(ids,state=pair['state'],time=pair['time'],**cond)
            checks['zero_velocity_exact']=torch.equal(reference,value)
            order=torch.randperm(len(ids),device='cuda',generator=generator)
            shuffled={k:v[order] for k,v in cond.items()}
            checks['sorting_exact']=torch.equal(value[order],model(ids[order],state=pair['state'][order],time=pair['time'],**shuffled))
            context=model.prepare_inference_context(ids,**cond)
            checks['cached_exact']=torch.equal(value,model(ids,state=pair['state'],time=pair['time'],inference_context=context))
            initial_endpoint=endpoint(model,ids,cond,pair['source'])
            checks['zero_endpoint_exact']=torch.equal(reference_endpoint,initial_endpoint)
            checks['zero_occupancy_exact']=torch.equal(reference_endpoint>0,initial_endpoint>0)
            network_timing=timing(lambda:model(ids,state=pair['state'],time=pair['time'],**cond))
        resume=probe_resume(model,ids,cond,pair,a.output/(mode+'_resume_probe.pt'))
        checks['resume_loss_exact']=resume['loss_exact']
        checks['same_weights_repeated_forward_exact']=resume['repeated_forward_exact']
        checks['resume_embedding_gradient_bounded']=resume['bounded_embedding_wgrad_only']
        checks['resume_baseline_control']=(resume['parameters_exact'] or not baseline_resume['gradients_exact'])
        checks['resume_endpoint_within_baseline']=all(resume['endpoint_error'][k]<=2*baseline_resume['endpoint_error'][k]+1e-7
            for k in ('max_abs','mean_abs','relative_l2'))
        checks['branch_gradients_finite_nonzero']=all(p.grad is not None and bool(torch.isfinite(p.grad).all())
            and bool((p.grad!=0).any()) for p in model.child_state_encoder.parameters())
        checks['all_outputs_finite']=bool(torch.isfinite(value).all())
        results[mode]={'checks':checks,'resume':resume,'initialization':init,
            'config_sha256':config_sha(current),'parameters':sum(p.numel() for p in model.parameters()),
            'new_parameters':sum(p.numel() for p in model.child_state_encoder.parameters()),'network_timing':network_timing}
        del model,value,context;torch.cuda.empty_cache()
    passed=all(baseline_resume[k] for k in ('loss_exact','repeated_forward_exact','bounded_embedding_wgrad_only'))
    passed=passed and all(all(v['checks'].values()) for v in results.values())
    report={'status':'PASS' if passed else 'FAIL','source_sha256':source_sha(),'gpu_admission':admission,
        'scene_tokens':lengths,'stress_source_sha256':file_sha(a.stress_source),'source_initialization':initialization,
        'baseline_resume':baseline_resume,'baseline_timing':baseline_timing,'variants':results,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'resume_equivalence':'Numerical embedding weight-gradient replay; not bitwise training reproducibility'}
    atomic_json(a.output/'result.json',report)
    if not passed:raise RuntimeError('Local state GPU gate failed; see retained report')
    print(json.dumps({'status':report['status'],'tokens':lengths,'variants':{k:v['checks'] for k,v in results.items()}},indent=2))


if __name__=='__main__':main()

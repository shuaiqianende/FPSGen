"""Real full384 scene/maximum-token PTv3 gate; strictly refuses busy GPUs."""
from common import ROOT
import argparse
import copy
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel as HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.flow_matching import training_pair,fm_loss,bce_loss
from fpsgen.hier_occ_flow.candidate_selection import candidate_metrics
from fpsgen.hier_occ_flow.stage1_training import state_sha


def model_loss(model,ids,c,target,objective,generator):
    if objective=='fm':
        pair=training_pair(target,c['valid_child_mask'],generator,t=torch.tensor(.4,device=ids.device))
        pred=model(ids,state=pair['state'],time=pair['time'],**c)
        return pred,fm_loss(pred,pair['velocity'],c['valid_child_mask']),pair
    pred=model(ids,**c)
    return pred,bce_loss(pred,target,c['valid_child_mask']),None


def run(args,cfg,g,admission):
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda')
    dataset,_,_=dataset_identity(cfg,g,verify=True)
    audit=json.loads(args.token_audit.read_text())
    if audit['status']!='PASS_CPU_FULL_TRAINING_TOKEN_AUDIT' or audit['train_frames']!=17870 or audit['geometry_sha256']!=g.sha256:
        raise ValueError('Need a real full-training token audit')
    representatives=audit['representatives']
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval();stage1_hash=state_sha(adapter.model)
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference import model as official
    import spconv.pytorch as spconv
    counts={'flash':0,'spconv':0};original_flash=official.flash_attn.flash_attn_varlen_qkvpacked_func
    def flash(*a,**kw):counts['flash']+=1;return original_flash(*a,**kw)
    official.flash_attn.flash_attn_varlen_qkvpacked_func=flash
    report={'status':'RUNNING','executed_gpu_operations':True,'admission':admission,'stage1':adapter.provenance,
        'geometry_sha256':g.sha256,'rows':[],'parity':[],'permutation':[],
        'initialization':'random Stage2; frozen actual384 Occupancy Stage1',
        'parity_tolerance':{'atol':1e-5,'rtol':1e-3},'permutation_atol':1e-4,
        'token_audit':'full17870 training scenes, actual max—not sampled max'}
    try:
        for objective in ('bce','fm'):
            local_cfg=copy.deepcopy(cfg);local_cfg['model']['objective']=objective
            torch.manual_seed(20261018);model=HierarchicalOccupancyModel(g,**local_cfg['model']).to(device).train()
            hooks=[m.register_forward_hook(lambda *unused:counts.__setitem__('spconv',counts['spconv']+1))
                   for m in model.modules() if isinstance(m,spconv.SubMConv3d)]
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
            scaler=torch.cuda.amp.GradScaler(init_scale=16384.)
            for representative in representatives:
                for mode in ('oracle','predicted'):
                    local_cfg['candidates']['mode']=mode;torch.cuda.reset_peak_memory_stats(device)
                    record=dataset[representative['index']];start=time.perf_counter();before=adapter.encoder_forward_count
                    ids,c,target,source=assemble_stage2_scene(record,local_cfg,g,device,adapter)
                    torch.cuda.synchronize();condition_seconds=time.perf_counter()-start
                    if mode=='oracle' and len(ids)!=representative['oracle_tokens']:raise ValueError('Full-scene oracle token count mismatch')
                    if adapter.encoder_forward_count-before!=1 or adapter.last_feature_shapes!=adapter.feature_shapes:
                        raise ValueError('Stage1 encoder repeated or actual pyramid shapes incorrect')
                    optimizer.zero_grad(set_to_none=True);counts['flash']=counts['spconv']=0
                    start=time.perf_counter()
                    with torch.autocast('cuda',dtype=torch.float16):
                        pred,loss,pair=model_loss(model,ids,c,target,objective,torch.Generator(device=device).manual_seed(5))
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    finite=bool(torch.isfinite(loss) and torch.isfinite(grad)) and float(grad)>0
                    if not finite:raise RuntimeError('Nonfinite/zero full-scene gradients')
                    if pred.shape!=(len(ids),8) or counts['flash']==0 or counts['spconv']==0:raise RuntimeError('Expected8D parent output with actual FlashAttention/SpConv execution')
                    scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
                    if scaler.get_scale()<scale_before:raise RuntimeError('AMP skipped the benchmark optimizer update')
                    row={'objective':objective,'candidate_mode':mode,'sequence':record['sequence'],'frame':record['frame'],
                        'quantile':representative['quantile'],'tokens':len(ids),'gt_parents':len(record['parent_ids']),
                        'condition_seconds':condition_seconds,'forward_backward_step_seconds':time.perf_counter()-start,
                        'loss':float(loss),'gradient_norm':float(grad),'finite_gradients':finite,
                        'amp_scale':scaler.get_scale(),'optimizer_update_executed':True,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                        'flash_calls':counts['flash'],'spconv_calls':counts['spconv'],'actual_feature_shapes':adapter.last_feature_shapes,
                        'sampled_feature_shape':list(c['lidar_features'].shape),'actual_output_shape':list(pred.shape),
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'candidate_metrics':candidate_metrics(ids.cpu().numpy(),record['parent_ids'],record['child_occ_mask'],g),
                        'no_token_subsampling':True,'stage1_encoder_forwards':1,'source':source}
                    report['rows'].append(row);atomic_json(args.output,report)
                    print(f'GPU{args.physical_gpu} {objective} {mode} Q{representative["quantile"]} parents={len(ids)} peak={row["peak_allocated_bytes"]/1024**3:.2f}GiB',flush=True)
                    if representative['quantile']==50 and mode=='oracle':
                        model.eval()
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                            kwargs={'state':pair['state'],'time':pair['time']} if pair is not None else {}
                            expected=model(ids,**kwargs,**c)
                            order=torch.randperm(len(ids),device=device);shuffled={k:v[order] for k,v in c.items()}
                            kwargs={'state':pair['state'][order],'time':pair['time']} if pair is not None else {}
                            actual=model(ids[order],**kwargs,**shuffled)[torch.argsort(order)]
                        error=float((expected-actual).abs().max());report['permutation'].append({'objective':objective,'tokens':len(ids),'max_abs_error':error,'pass':error<1e-4})
                        model.train();del expected,actual,kwargs
                        # Parity is a2048-token arithmetic diagnostic; all
                        # resource rows above retain every real scene token.
                        small=ids[:2048];small_c={k:v[:2048] for k,v in c.items()};small_target=target[:2048]
                        ref_cfg=copy.deepcopy(local_cfg['model']);ref_cfg['checkpointing']=False
                        reference=HierarchicalOccupancyModel(g,**ref_cfg).to(device).train();reference.load_state_dict(model.state_dict(),strict=True)
                        gradients=[];predictions=[]
                        for net in (model,reference):
                            net.zero_grad(set_to_none=True)
                            with torch.autocast('cuda',dtype=torch.float16):
                                value,local_loss,_=model_loss(net,small,small_c,small_target,objective,torch.Generator(device=device).manual_seed(11))
                            (local_loss*16384.).backward();predictions.append(value.detach().cpu())
                            gradients.append({k:p.grad.detach().float().cpu().clone()/16384. if p.grad is not None else None for k,p in net.named_parameters()})
                        passed=torch.allclose(*predictions,atol=1e-5,rtol=1e-3)
                        differences=[]
                        for name,a in gradients[0].items():
                            b=gradients[1][name]
                            if a is None or b is None:passed&=(a is None and b is None)
                            else:passed&=torch.allclose(a,b,atol=1e-5,rtol=1e-3);differences.append(float((a-b).abs().max()))
                        report['parity'].append({'objective':objective,'diagnostic_tokens':len(small),'pass':bool(passed),
                            'max_gradient_abs_error':max(differences),'checkpointing_not_used_to_reduce_training_tokens':True})
                        del reference,gradients,predictions,small,small_c,small_target,value,local_loss,net,shuffled,order
                    del ids,c,target,pred,loss,pair
            for hook in hooks:hook.remove()
            del optimizer,model;torch.cuda.empty_cache()
        frozen=state_sha(adapter.model)==stage1_hash and all(p.grad is None and not p.requires_grad for p in adapter.parameters())
        passed=frozen and len(report['rows'])==16 and all(r['pass'] for r in report['parity']+report['permutation'])
        report.update(status='PASS_GPU_FULL_SCENE' if passed else 'FAIL_GPU_FULL_SCENE',stage1_unchanged=frozen,
            maximum_oracle_tokens=max(r['tokens'] for r in report['rows'] if r['candidate_mode']=='oracle'))
    except Exception as exc:
        report.update(status='FAIL_GPU_FULL_SCENE',exception=repr(exc),traceback=traceback.format_exc(),
            no_oom_scene_skipped=True,no_automatic_token_reduction=True)
        atomic_json(args.output,report);raise
    finally:official.flash_attn.flash_attn_varlen_qkvpacked_func=original_flash
    atomic_json(args.output,report)
    if report['status']!='PASS_GPU_FULL_SCENE':raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--token-audit',type=Path,required=True);p.add_argument('--physical-gpu',type=int,choices=(2,3),default=2)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    admission=gpu_preflight((args.physical_gpu,))
    if admission['status']!='AVAILABLE' or not args.execute:
        atomic_json(args.output,{'status':admission['status'] if admission['status']!='AVAILABLE' else 'NOT_EXECUTED',
            'executed_gpu_operations':False,'admission':admission,'reason':'Stage1/other compute process preserved'})
        print('Stage2 GPU validation not executed:',admission,flush=True);return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise ValueError('Select the same single physical GPU explicitly')
    cfg,g=load_stage2_config(args.config);run(args,cfg,g,admission)


if __name__=='__main__':main()

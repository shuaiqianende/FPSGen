"""Actual B2 ragged PTv3 validation including two complete maximum scenes."""
from common import ROOT
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--physical-gpu',type=int,choices=(2,3),required=True)
    args=p.parse_args();admission=gpu_preflight((args.physical_gpu,))
    if admission['status']!='AVAILABLE':atomic_json(args.output,{'status':'BLOCKED','admission':admission});return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise ValueError('Single admitted GPU required')
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda')
    cfg,g=load_stage2_config(args.config);dataset,_,_=dataset_identity(cfg,g,verify=True)
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval()
    torch.manual_seed(cfg['training']['seed']);model=Stage2OccupancyModel(g,**cfg['model']).to(device)
    if cfg.get('transition'):
        transition=cfg['transition']
        if file_sha(transition['checkpoint'])!=transition['checkpoint_sha256']:raise ValueError('Changed own FM source checkpoint')
        own=torch.load(transition['checkpoint'],map_location='cpu')
        if own['identity']['stage1_checkpoint_sha256']!=cfg['stage1']['checkpoint_sha256']:raise ValueError('Stage1 lineage differs')
        model.load_state_dict(own['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr']);scaler=torch.cuda.amp.GradScaler(init_scale=16384.)
    report={'status':'RUNNING','physical_gpu':args.physical_gpu,'geometry_sha256':g.sha256,
            'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],'rows':[],
            'tolerance':{'control_atol':1e-4,'control_rtol':1e-3,'amp_relative_l2':torch.finfo(torch.float16).eps,
                'amp_max_units':2,'amp_sign_disagreement':1e-4},'no_token_truncation':True,
            'precision_contract':'Strict AMP-disabled control required; AMP row-count GEMMs may differ by half-precision rounding. Original cellwise AMP check remains reported.'}
    try:
        for kind,indices in (('unequal_real_scenes',[9511,6147]),('two_complete_maximum_scenes',[6147,6147])):
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            scenes=[assemble_stage2_scene(dataset[i],cfg,g,device,adapter) for i in indices]
            ids,c,target,lengths=pack_scenes(scenes)
            pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],torch.Generator(device=device).manual_seed(111))
            checks={};model.eval()
            if kind=='unequal_real_scenes':
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    packed=model(ids,state=pair['state'],time=pair['time'],**c)
                    isolated=[];offset=0
                    for index,scene in enumerate(scenes):
                        local_ids,local_c,_,_=scene;n=len(local_ids)
                        isolated.append(model(local_ids,state=pair['state'][offset:offset+n],time=pair['time'][index],**local_c));offset+=n
                    independent=torch.cat(isolated)
                    checks['original_strict_amp_comparison_met']=torch.allclose(packed,independent,atol=1e-4,rtol=1e-3)
                    checks['packed_vs_separate_max_abs']=float((packed-independent).abs().max())
                    checks['packed_vs_separate_mean_abs']=float((packed-independent).abs().float().mean())
                    checks['packed_vs_separate_relative_l2']=float((packed-independent).float().norm()/independent.float().norm())
                    checks['packed_vs_separate_sign_disagreement']=float(((packed>0)!=(independent>0)).float().mean())
                    units=(packed-independent).abs().float()/(torch.finfo(torch.float16).eps*independent.abs().float().clamp_min(1))
                    checks['amp_max_rounding_units']=float(units.max())
                    changed=pair['state'].clone();changed[c['batch']==1]+=3
                    changed_time=pair['time'].clone();changed_time[1]=1-changed_time[1]
                    altered=model(ids,state=changed,time=changed_time,**c)
                    first=c['batch']==0
                    checks['scene_isolation_pass']=torch.allclose(packed[first],altered[first],atol=1e-4,rtol=1e-3)
                    checks['scene_isolation_max_abs']=float((packed[first]-altered[first]).abs().max())
                    permutation=torch.randperm(len(ids),device=device);shuffled={k:v[permutation] for k,v in c.items()}
                    reordered=model(ids[permutation],state=pair['state'][permutation],time=pair['time'],**shuffled)[torch.argsort(permutation)]
                    checks['permutation_pass']=torch.allclose(packed,reordered,atol=1e-4,rtol=1e-3)
                    checks['permutation_max_abs']=float((packed-reordered).abs().max())
                del packed,isolated,independent,altered,reordered,changed,changed_time,shuffled,permutation
                # Control for AMP GEMM row-count dependent rounding, without
                # changing the registered FP32 comparison tolerance.
                tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    packed32=model(ids,state=pair['state'],time=pair['time'],**c)
                    isolated32=[];offset=0
                    for index,scene in enumerate(scenes):
                        local_ids,local_c,_,_=scene;n=len(local_ids)
                        isolated32.append(model(local_ids,state=pair['state'][offset:offset+n],time=pair['time'][index],**local_c));offset+=n
                    independent32=torch.cat(isolated32)
                    checks['fp32_packed_vs_separate_pass']=torch.allclose(packed32,independent32,atol=1e-4,rtol=1e-3)
                    checks['fp32_packed_vs_separate_max_abs']=float((packed32-independent32).abs().max())
                    checks['fp32_packed_vs_separate_mean_abs']=float((packed32-independent32).abs().mean())
                checks['packed_vs_separate_pass']=(checks['fp32_packed_vs_separate_pass']
                    and checks['packed_vs_separate_relative_l2']<=torch.finfo(torch.float16).eps
                    and checks['amp_max_rounding_units']<=2 and checks['packed_vs_separate_sign_disagreement']<=1e-4)
                torch.backends.cuda.matmul.allow_tf32=tf32
                del packed32,isolated32,independent32
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                prediction=model(ids,state=pair['state'],time=pair['time'],**c)
                loss=scene_mean_mse(prediction,pair['velocity'],c['valid_child_mask'],c['batch'])
            scaler.scale(loss).backward();scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            finite=bool(torch.isfinite(loss) and torch.isfinite(grad)) and float(grad)>0
            if not finite:raise RuntimeError('Nonfinite packed batch gradient')
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            row={'kind':kind,'indices':indices,'tokens_per_scene':lengths,'tokens_total':len(ids),'batch_per_gpu':2,
                 'loss':float(loss),'gradient_norm':float(grad),'finite':finite,'optimizer_update_executed':True,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                 'seconds':time.perf_counter()-start,**checks}
            report['rows'].append(row);atomic_json(args.output,report);print(row,flush=True)
            del scenes,ids,c,target,pair,prediction,loss
        passed=all(row['finite'] and all(v for k,v in row.items() if k.endswith('_pass')) for row in report['rows'])
        report['status']='PASS_PACKED_BATCH2' if passed else 'FAIL_PACKED_BATCH2'
    except Exception as exc:
        report.update(status='FAIL_PACKED_BATCH2',exception=repr(exc),traceback=traceback.format_exc())
        atomic_json(args.output,report);raise
    atomic_json(args.output,report)
    if report['status']!='PASS_PACKED_BATCH2':raise SystemExit(2)


if __name__=='__main__':main()

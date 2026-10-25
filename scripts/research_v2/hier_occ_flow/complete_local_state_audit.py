"""P0 completion audit and measured branch activity; requires all existing evaluations."""
from common import ROOT
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from audit_desescene_evidence import paired_delta
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,config_sha
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.inference import sample
from fpsgen.hier_occ_flow.adaptation_gpu_slot import adaptation_admission
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.encoder_adaptation import STAGE2_SHA


def tree_sha(root):
    digest=hashlib.sha256()
    for folder in ('fpsgen/hier_occ_flow','fpsgen/voxel_point/vendor/pwm/ptv3_reference','scripts/research_v2/hier_occ_flow'):
        for path in sorted((root/folder).rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def comparison(reference,candidate):
    for a,b in zip(reference,candidate):
        if (a['sequence'],a['frame'],a['input_sha256'],a['parent_ids_sha256'])!=(b['sequence'],b['frame'],b['input_sha256'],b['parent_ids_sha256']):
            raise ValueError('P0 paired input/support mismatch')
        if a['continuous']['raw_gt_xyz_sha256']!=b['continuous']['raw_gt_xyz_sha256']:raise ValueError('P0 raw geometry GT changed')
    out={'fine_iou':paired_delta(reference,candidate,nested=True)}
    out.update({'fine_'+metric:paired_delta(reference,candidate,nested=True,metric=metric) for metric in ('f1','precision','recall')})
    for metric in ('cd_m','iou05','iou02','iou01','jsd_3d','jsd_bev'):
        a=[dict(sequence=r['sequence'],frame=r['frame'],**r['continuous']) for r in reference]
        b=[dict(sequence=r['sequence'],frame=r['frame'],**r['continuous']) for r in candidate]
        value=paired_delta(a,b,metric=metric)
        delta=sorted([{'sequence':x['sequence'],'frame':x['frame'],'delta':y[metric]-x[metric]} for x,y in zip(a,b)],
                     key=lambda r:r['delta'],reverse=metric in ('cd_m','jsd_3d','jsd_bev'))
        value['worst_frames']=delta[:10];value['worst_decile_mean']=float(np.mean([r['delta'] for r in delta[:90]]));out[metric]=value
    return out


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--p0-root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    scheduler=json.loads((a.p0_root/'gpu2_transfer/summary.json').read_text())
    if scheduler['status']!='COMPLETE':raise ValueError('Existing P0 queue has not completed')
    baseline=json.loads((a.baseline_root/'evaluation/primary900_nfe4.json').read_text())['nfe_results']['4']
    if len(baseline['rows'])!=900:raise ValueError('B0 primary900 required')
    results={};activities=[];checkpoint_audits={};evaluated={}
    running_source_sha=tree_sha(a.p0_root.parents[2])
    b0_identity=torch.load(a.baseline_root/'checkpoints/resume_latest.pt',map_location='cpu')['identity']
    for arm in ('position','attention'):
        folder=a.p0_root/(arm+'_20k');cfg,g=load_stage2_config(folder/'config.json')
        saved=torch.load(folder/'checkpoints/resume_latest.pt',map_location='cpu')
        if (saved['successful_updates']!=10000 or saved['sampler']['successful_frame_exposures']!=20000
            or saved['identity']['config_sha256']!=config_sha(cfg) or saved['ema'] is None
            or not saved['optimizer']['state'] or cfg['training']['seed']!=20261032
            or saved['identity']['source_sha256']!=running_source_sha
            or saved.get('initialization_audit',{}).get('checkpoint_sha256')!=STAGE2_SHA
            or not all(bool(torch.isfinite(v).all()) for v in saved['ema'].values() if v.is_floating_point())):
            raise ValueError('P0 final checkpoint identity/source/budget/EMA/optimizer failure')
        evaluation=json.loads((folder/'evaluation/primary900_nfe4.json').read_text())['nfe_results']['4']
        for key in ('train_manifest_sha256','development_manifest_sha256','stage1_checkpoint_sha256','geometry_sha256'):
            if saved['identity'][key]!=b0_identity[key]:raise ValueError('P0/B0 identity mismatch: '+key)
        evaluated[arm]=evaluation['rows']
        diagnostic=json.loads((folder/'evaluation/mechanisms64/summary.json').read_text())
        if len(evaluation['rows'])!=900 or diagnostic['frames']!=64:raise ValueError('P0 evaluation incomplete')
        checkpoint_audits[arm]={'checkpoint_sha256':file_sha(folder/'checkpoints/resume_latest.pt'),
            'identity':saved['identity'],'successful_updates':10000,'successful_exposures':20000,
            'amp_skips':saved['amp_skips'],'optimizer_entries':len(saved['optimizer']['state']),
            'scheduler':saved['scheduler'],'scaler':saved['scaler'],
            'new_branch_output_weight_norm':float(saved['ema']['child_state_encoder.output.weight'].norm())}
        results[arm]={'summary':evaluation['summary'],'continuous_macro':evaluation['continuous_macro'],
                     'paired_vs_b0':comparison(baseline['rows'],evaluation['rows']),
                     'mechanisms64':diagnostic,
                     'occupied_children_mean':float(np.mean([r['continuous']['predicted_points'] for r in evaluation['rows']]))}
        with (folder/'evaluation/mechanisms64/sensitivity.csv').open() as stream:sensitivity=list(csv.DictReader(stream))
        results[arm]['condition_sensitivity']={name:{'rows':sum(r['perturbation']==name for r in sensitivity),
            'mean_velocity_relative_l2':float(np.mean([float(r['velocity_relative_l2']) for r in sensitivity if r['perturbation']==name]))}
            for name in sorted({r['perturbation'] for r in sensitivity})}
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in ('2','3') or adaptation_admission(int(physical), configure_allocator=True)['status']!='AVAILABLE':raise RuntimeError('Admitted GPU2/3 required for independent activity diagnosis')
    torch.set_num_threads(4)
    for arm in results:
        folder=a.p0_root/(arm+'_20k');cfg,g=load_stage2_config(folder/'config.json');_,dev,_=dataset_identity(cfg,g,verify=False)
        saved=torch.load(folder/'checkpoints/resume_latest.pt',map_location='cpu')
        teacher=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda()
        model=Stage2OccupancyModel(g,**cfg['model']).cuda().eval();model.load_state_dict(saved['ema'],strict=True)
        state={};handles=[]
        def base_hook(module,args,out):state['base_rms']=float(out.float().square().mean().sqrt())
        def branch_hook(module,args,out):
            rms=float(out.float().square().mean().sqrt());state['residual_rms']=rms
            state['relative_residual_rms']=rms/max(state['base_rms'],1e-12)
        handles.extend((model.local.register_forward_hook(base_hook),model.child_state_encoder.register_forward_hook(branch_hook)))
        if arm=='attention':
            def attention_hook(module,args,out):
                n=len(out);q,k,v=out.reshape(n,8,3,2,8).permute(2,0,3,1,4).unbind(0)
                scores=(q@k.transpose(-2,-1)).float()/8**.5;keys=state['valid'][:,None,None,:]
                prob=scores.masked_fill(~keys,torch.finfo(scores.dtype).min).softmax(-1)*keys
                prob=prob/prob.sum(-1,keepdim=True).clamp_min(1e-12)
                query=state['valid'][:,None,:].expand(-1,2,-1)
                state['attention_entropy']=float((-(prob*prob.clamp_min(1e-12).log()).sum(-1))[query].mean())
                state['attention_max_weight']=float(prob.max(-1).values[query].mean())
                state['attention_diagonal_mass']=float(prob.diagonal(dim1=-2,dim2=-1)[query].mean())
            handles.append(model.child_state_encoder.qkv.register_forward_hook(attention_hook))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            for index in np.linspace(0,len(dev)-1,64,dtype=int):
                record=dev[int(index)];ids,c,_,_=assemble_stage2_scene(record,cfg,g,torch.device('cuda'),teacher,cache=None)
                state['valid']=c['valid_child_mask'];x=torch.randn(c['valid_child_mask'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20261031+int(index)))*c['valid_child_mask']
                for step in range(4):
                    velocity=model(ids,state=x,time=x.new_tensor(step/4),**c);x=x+velocity.float()/4
                    activities.append({'arm':arm,'sequence':record['sequence'],'frame':record['frame'],'step':step,
                        **{k:v for k,v in state.items() if k!='valid'}})
        for h in handles:h.remove()
        del teacher,model;torch.cuda.empty_cache()
    gain={arm:r['paired_vs_b0']['fine_iou']['mean'] for arm,r in results.items()}
    atomic_json(a.output/'summary.json',{'status':'COMPLETE','results':results,'checkpoint_audits':checkpoint_audits,
        'b0_summary':baseline['summary'],'b0_continuous':baseline['continuous_macro'],
        'attention_vs_position':comparison(evaluated['position'],evaluated['attention']),
        'branch_activity':activities,'single_training_seed_only':True,
        'decision':{arm:'NO_GO_EXPANSION' if d<.01 else 'QUALITY_RUNTIME_REVIEW_REQUIRED' for arm,d in gain.items()},
        'timing_scope':'Separate dedicated3-round5+20 timing artifacts required; evaluation seconds are diagnostic'})
    lines=['# P0 completed L0/L1 audit','','| Arm | Native Fine IoU | CD m | IoU0.1 | Decision |','|---|---:|---:|---:|---|']
    for arm,r in results.items():lines.append(f"| {arm} | {r['summary']['macro']['iou']:.8f} | {r['continuous_macro']['cd_m']:.8f} | {r['continuous_macro']['iou01']:.8f} | {'NO-GO expansion' if gain[arm]<.01 else 'Review'} |")
    lines+=['','Paired900 inputs and Parent identities are equal toB0. Checkpoint/EMA/optimizer/scaler/counters and real branch-activity diagnostics are in summary.json. CI resamples sequence/time blocks, not independent training seeds. Dedicated timing is in the sibling timing directory; no runtime gain is inferred from loss or parameter count.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

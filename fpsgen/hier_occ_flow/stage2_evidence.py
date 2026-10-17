"""CPU evidence checks for real GPU runs; missing results always stay pending."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .storage import file_sha


# Registered before GPU execution. Integer/RNG/sampler identities remain exact.
RESUME_ATOL=1e-5
RESUME_RTOL=1e-3
OVERFIT_LOSS_RATIO=.9
OVERFIT_F1_MARGIN=.02


def compare_tree(a,b,path='root',errors=None):
    errors=[] if errors is None else errors
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype:
            errors.append(path+': tensor shape/dtype differs')
        elif a.is_floating_point() and not path.endswith('.step'):
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a,b,atol=RESUME_ATOL,rtol=RESUME_RTOL):
                errors.append(path+': floating tensor differs beyond registered tolerance')
        elif not torch.equal(a,b):errors.append(path+': exact tensor differs')
    elif isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
        if not np.array_equal(a,b):errors.append(path+': exact NumPy/RNG array differs')
    elif isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():errors.append(path+': keys differ')
        for key in a.keys() & b.keys():compare_tree(a[key],b[key],f'{path}.{key}',errors)
    elif isinstance(a,(tuple,list)) and isinstance(b,type(a)):
        if len(a)!=len(b):errors.append(path+': lengths differ')
        for i,(x,y) in enumerate(zip(a,b)):compare_tree(x,y,f'{path}[{i}]',errors)
    elif isinstance(a,float) and isinstance(b,(float,int)):
        if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,abs_tol=RESUME_ATOL,rel_tol=RESUME_RTOL):errors.append(path+': scalar differs')
    elif a!=b:errors.append(path+': exact value differs')
    return errors


def compare_resume(reference,resumed):
    """Both checkpoints must be from the SAME output/identity and two ranks."""
    errors=[]
    required=('identity','initialization','successful_updates','attempts','amp_skips',
              'consecutive_failures','model','ema','optimizer','scheduler','scaler','rank_rng','sampler','parameter_sync_pass')
    for key in required:
        if key not in reference or key not in resumed:errors.append(key+': missing checkpoint state')
        else:compare_tree(reference[key],resumed[key],key,errors)
    identity=reference.get('identity',{})
    if identity.get('experiment')!='hs-occflow-stage2-384-v1' or identity.get('world_size')!=2:errors.append('Need this384 Stage2 two-rank checkpoint')
    if reference.get('successful_updates')!=21:errors.append('Registered GPU restore test ends at successful update21')
    for value in (reference,resumed):
        if value.get('initialization')!='scratch_stage2' or value.get('parameter_sync_pass') is not True:errors.append('Scratch/synchronized state required')
        if len(value.get('rank_rng',[]))!=2:errors.append('Two rank RNG snapshots required')
        hashes=value.get('rank_hashes',[])
        if len(hashes)!=2 or hashes[0]!=hashes[1]:errors.append('Actual online/EMA rank hashes must agree')
    return {'status':'PASS_DDP_RESUME' if not errors else 'FAIL_DDP_RESUME',
            'geometry_sha256':identity.get('geometry_sha256'),
            'stage1_checkpoint_sha256':identity.get('stage1_checkpoint_sha256'),
            'successful_updates':reference.get('successful_updates'),
            'global_batch':reference.get('sampler',{}).get('global_batch',identity.get('world_size',1)),
            'batch_per_gpu':reference.get('sampler',{}).get('batch_per_gpu',1),
            'tolerance':{'atol':RESUME_ATOL,'rtol':RESUME_RTOL,'rng_sampler_integer':'exact'},'errors':errors}


def overfit_evidence(directory,objective):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text());cfg=json.loads((root/'config.json').read_text())
    identity=summary['identity'];n=summary['successful_updates']
    evaluation=json.loads((root/'overfit'/f'update_{n:06d}.json').read_text())
    logs=[json.loads(line) for line in (root/'train_rank0.jsonl').read_text().splitlines()]
    launch=json.loads((root/'launch.json').read_text()) if (root/'launch.json').exists() else {}
    transition=launch.get('transition')
    if transition:
        previous_root=Path(transition['source_identity']['output'])
        inherited=[json.loads(line) for line in (previous_root/'train_rank0.jsonl').read_text().splitlines()]
        logs=[row for row in inherited if row['successful_updates']<=transition['inherited_successful_updates']]+logs
    losses=[r['loss'] for r in logs if r['finite']]
    beginning=float(np.mean(losses[:20])) if len(losses)>=40 else float('inf')
    ending=float(np.mean(losses[-20:])) if len(losses)>=40 else float('inf')
    scores=[]
    for nfe,result in evaluation['nfe_results'].items():
        rows=result['rows'];baseline=float(np.mean([r['all_candidate_children_occupied_f1'] for r in rows]))
        scores.append({'nfe':int(nfe),'f1':result['summary']['macro']['f1'],'all_occupied_f1':baseline,
                       'pass':result['summary']['macro']['f1']>baseline+OVERFIT_F1_MARGIN})
    passed=(summary['status']=='COMPLETE_REQUESTED_BUDGET' and cfg['training']['scope']=='overfit'
            and cfg['candidates']['mode']=='oracle' and cfg['model']['objective']==objective
            and evaluation.get('ema_used') is False and n>=200
            and len(losses)>=40 and all(r['finite'] for r in logs)
            and ending<beginning*OVERFIT_LOSS_RATIO and any(s['pass'] for s in scores))
    return {'status':'PASS_ORACLE_OVERFIT' if passed else 'FAIL_ORACLE_OVERFIT','objective':objective,
            'geometry_sha256':identity['geometry_sha256'],'stage1_checkpoint_sha256':identity['stage1_checkpoint_sha256'],
            'first20_mean_loss':beginning,'last20_mean_loss':ending,'scores':scores,
            'registered_thresholds':{'last_first_loss_ratio':OVERFIT_LOSS_RATIO,'f1_margin_over_all_occupied':OVERFIT_F1_MARGIN}}


def matched_pilot_evidence(bce_directory,fm_directory):
    runs=[]
    for directory,objective in ((bce_directory,'bce'),(fm_directory,'fm')):
        root=Path(directory);cfg=json.loads((root/'config.json').read_text());summary=json.loads((root/'summary.json').read_text())
        report=json.loads((root/'validation'/'update_001000.json').read_text());identity=summary['identity']
        logs=[json.loads(line) for line in (root/'train_rank0.jsonl').read_text().splitlines()]
        values=[r['loss'] for r in logs if r['finite']]
        latest=report['nfe_results']['1' if objective=='bce' else '8']
        baseline=float(np.mean([r['all_candidate_children_occupied_f1'] for r in latest['rows']]))
        passed=(cfg['model']['objective']==objective and cfg['training']['scope']=='pilot1000'
                and summary['status']=='COMPLETE_REQUESTED_BUDGET' and summary['successful_updates']==1000
                and not cfg.get('cohort_indices') and all(r['finite'] for r in logs) and len(values)>=1000
                and np.mean(values[-50:])<np.mean(values[:50])
                and latest['summary']['macro']['f1']>baseline)
        runs.append({'objective':objective,'pass':bool(passed),'identity':identity,'cfg':cfg,
                     'frame_order':[(r['sequence'],r['frame']) for r in logs],
                     'macro_f1':latest['summary']['macro']['f1'],'all_occupied_f1':baseline,
                     'seconds':summary['seconds'],'root':str(root.resolve())})
    a,b=runs
    # Capacity differences due to FM state/time modules are explicitly reported;
    # the parent PTv3 backbone and frozen conditions must be identical.
    model_a={k:v for k,v in a['cfg']['model'].items() if k!='objective'}
    model_b={k:v for k,v in b['cfg']['model'].items() if k!='objective'}
    matched=(model_a==model_b and a['cfg']['candidates']==b['cfg']['candidates']
             and a['cfg']['training']==b['cfg']['training'] and a['cfg']['validation']==b['cfg']['validation']
             and a['frame_order']==b['frame_order'] and a['identity']['world_size']==b['identity']['world_size']
             and all(a['identity'][k]==b['identity'][k] for k in ('geometry_sha256','stage1_checkpoint_sha256',
                       'train_manifest_sha256','development_manifest_sha256')))
    return {'status':'PASS_MATCHED_1K' if matched and all(r['pass'] for r in runs) else 'FAIL_MATCHED_1K',
            'geometry_sha256':a['identity']['geometry_sha256'],'stage1_checkpoint_sha256':a['identity']['stage1_checkpoint_sha256'],
            'matched_protocol':matched,'runs':[{k:v for k,v in r.items() if k not in ('cfg','frame_order')} for r in runs],
            'fm_superiority_claim':False,'interpretation':'This gate verifies matched stable learning, not statistical superiority.'}


def build_engineering_gate(geometry_sha,stage1_sha,*,gpu_reports=(),resume_report=None,overfit_reports=(),pilot_report=None):
    gate={'schema':'hs-occflow-stage2-384-engineering-v1','geometry_sha256':geometry_sha,
          'stage1_checkpoint_sha256':stage1_sha,'evidence':[],
          **{key:False for key in ('full_scene_gpu','maximum_scene_gpu','ptv3_permutation','checkpointing_parity',
             'ddp_resume','oracle_bce_overfit','oracle_fm_overfit','matched_bce_fm_1k','pilot5000',
             'stage1_final_checkpoint','development_candidate_calibration','formal_budget_registered')}}
    def read(path):
        value=json.loads(Path(path).read_text())
        actual=value.get('stage1_checkpoint_sha256',value.get('stage1',{}).get('checkpoint_sha256'))
        if value.get('geometry_sha256')!=geometry_sha or actual!=stage1_sha:raise ValueError('Evidence identity mismatch')
        gate['evidence'].append({'path':str(Path(path).resolve()),'sha256':file_sha(path),'status':value['status']})
        return value
    physical=set();passed=[]
    for path in gpu_reports:
        value=read(path)
        if value.get('executed_gpu_operations') is not True or value['status']!='PASS_GPU_FULL_SCENE':continue
        rows=value['rows']
        if (value.get('stage1_unchanged') is not True or len(rows)!=16
                or not all(r.get('flash_calls',0)>0 and r.get('spconv_calls',0)>0
                           and r.get('optimizer_update_executed') is True and r.get('no_token_subsampling') is True for r in rows)):continue
        physical.update(row['physical_gpu'] for row in value['admission']['gpus'])
        passed.append(value)
    both=physical=={2,3} and len(passed)>=2
    gate['full_scene_gpu']=both
    gate['maximum_scene_gpu']=both and all(v.get('maximum_oracle_tokens',0)>=399672 for v in passed)
    gate['ptv3_permutation']=both and all(len(v['permutation'])==2 and all(r['pass'] for r in v['permutation']) for v in passed)
    gate['checkpointing_parity']=both and all(len(v['parity'])==2 and all(r['pass'] for r in v['parity']) for v in passed)
    if resume_report:gate['ddp_resume']=read(resume_report)['status']=='PASS_DDP_RESUME'
    for path in overfit_reports:
        v=read(path)
        if v.get('objective') not in ('fm','bce'):raise ValueError('Unknown overfit objective')
        gate[f'oracle_{v["objective"]}_overfit']=v['status']=='PASS_ORACLE_OVERFIT'
    if pilot_report:gate['matched_bce_fm_1k']=read(pilot_report)['status']=='PASS_MATCHED_1K'
    gate['status']='READY_FOR_1K_PILOTS' if all(gate[k] for k in ('full_scene_gpu','maximum_scene_gpu','ptv3_permutation',
        'checkpointing_parity','ddp_resume','oracle_bce_overfit','oracle_fm_overfit')) else 'PENDING_REAL_GPU_EVIDENCE'
    return gate

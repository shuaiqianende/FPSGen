"""Build a measured engineering gate; never turn missing artifacts into PASS."""
from train_stage1_occ import load_config
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def read(path):return json.loads(path.read_text()) if path.exists() else None


def pilot_check(root):
    summary=read(root/'summary.json');metrics=read(root/'validation/update_001000.json')
    train=read(root/'validation/train_update_001000.json');initial=read(root/'initialization_audit.json')
    if not all((summary,metrics,train,initial)):return {'pass':False,'reason':'pilot not complete'}
    rows=[json.loads(s) for s in (root/'train_rank0.jsonl').read_text().splitlines()]
    good=[r for r in rows if r['finite']];audits=[r for r in good if r.get('gradient_details')]
    def quality(data):
        results=[]
        for key,values in data['nfe_results'].items():
            native=values['rows'];macro=values['summary']['macro']
            trivial=float(np.mean([2*r['gt_occupied_ratio']/(1+r['gt_occupied_ratio']) for r in native]))
            results.append({'nfe':key,'macro_f1':macro['f1'],'all_occupied_f1':trivial,
                'pass':macro['f1']>2*trivial and macro['recall']>.1
                    and all(0<r['predicted_occupied_ratio']<.5 for r in native)})
        return results
    development=quality(metrics);overfit=quality(train)
    finite=summary['successful_updates']==1000 and summary['status']=='COMPLETE_REQUESTED_BUDGET'
    finite &= all(r['loss'] is not None for r in rows) and sum(not r['finite'] for r in rows[-100:])<=1
    gradients=bool(audits) and all(all(v['nonfinite_elements']==0 and v['l2'] and v['l2']>0 for v in r['gradient_details'].values()) for r in audits)
    changed=bool(audits) and all(any(r.get('parameter_changes',{}).get(name,{}).get('sample_changed_elements',0)>0 for r in audits)
        for name in ('encoder','flow','velocity_head','ema'))
    scratch=initial['initialization_mode']=='scratch' and initial['loaded_pretrained_parameter_count']==0 and initial['optimizer_initial_state_entries']==0
    trend=float(np.mean([r['loss'] for r in good[-100:]]))<.9*float(np.mean([r['loss'] for r in good[:100]]))
    passed=finite and gradients and changed and scratch and trend and any(x['pass'] for x in development) and any(x['pass'] for x in overfit)
    return {'pass':bool(passed),'finite_and_completed':bool(finite),'nonzero_gradients':bool(gradients),'parameter_updates':bool(changed),
        'scratch':bool(scratch),'loss_learning_trend':bool(trend),'development':development,'training_diagnostic':overfit,
        'summary_sha256':file_sha(root/'summary.json'),'metrics_sha256':file_sha(root/'validation/update_001000.json')}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,default=Path('configs/research_v2/hier_occ_flow/stage1_occ384_fm.yaml'))
    p.add_argument('--run-root',type=Path,default=Path('experiments/hier_occ_flow'));p.add_argument('--output',type=Path,default=Path('experiments/hier_occ_flow/stage1_occ384_engineering_gate.json'));args=p.parse_args()
    cfg=load_config(args.config);run=args.run_root;sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    data=read(run/'occ384_full_parallel/full_data_audit.json');resume=read(run/'stage1_f64_same_origin_resume_audit.json')
    sync=read(run/'stage1_parameter_sync_audit.json')
    gpu=[read(run/f'stage1_gpu{n}_full_gate.json') for n in (2,3)]
    checks={'full_preprocessing':bool(data and data['status']=='PASS_FULL_DATA_AUDIT'),
        'gpu_shapes':all(x and x['status']=='PASS' and x['maximum_scope']=='dataset maximum' for x in gpu),
        'ddp_resume':bool(resume and resume['status']=='PASS_CUDA_TOLERANCE' and sync and sync['parameter_sync_pass'])}
    pilots={o:pilot_check(run/f'stage1_{o}_pilot1000_f64') for o in ('fm','bce')}
    checks['fm_bce_overfit']=all(v['pass'] for v in pilots.values());checks['pilot1000']=checks['fm_bce_overfit']
    g=cfg['geometry'];from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
    result={**checks,'formal_ready':all(checks.values()),'geometry_sha256':SparseHierarchyGeometry(**g).sha256,
        'registered_config_sha256':sha,'pilots':pilots,
        'engineering_quality_rule':'F1 exceeds twice the all-occupied baseline; positive recall/noncollapsed support; genuine loss/parameter learning; not a claim of method superiority'}
    if data:
        result['train_manifest_sha256']=data['manifest_sha256']['train'];result['development_manifest_sha256']=data['manifest_sha256']['development']
    atomic_json(args.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

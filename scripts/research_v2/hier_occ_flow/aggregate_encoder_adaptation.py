"""Paired full-development decisions; never schedules additional training."""
from common import ROOT
import argparse,json
from pathlib import Path
import numpy as np
from audit_desescene_evidence import paired_delta
from fpsgen.hier_occ_flow.storage import atomic_json


def aligned(left,right):
    if len(left)!=900 or len(right)!=900:raise ValueError('Both complete900 evaluations required')
    if [(r['sequence'],r['frame']) for r in left]!=[(r['sequence'],r['frame']) for r in right]:raise ValueError('Development identities differ')


def fine_comparison(left,right,*,fixed_support):
    aligned(left,right)
    for a,b in zip(left,right):
        if a['input_sha256']!=b['input_sha256']:raise ValueError('Input mismatch')
        if a['continuous'].get('raw_gt_xyz_sha256')!=b['continuous'].get('raw_gt_xyz_sha256'):raise ValueError('Raw geometry GT changed')
        if fixed_support and a['parent_ids_sha256']!=b['parent_ids_sha256']:raise ValueError('Teacher support drift')
    result={'native_fine_iou':paired_delta(left,right,nested=True)}
    for metric in ('f1','precision','recall'):
        result['native_fine_'+metric]=paired_delta(left,right,metric=metric,nested=True)
    for metric in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d'):
        a=[dict(sequence=r['sequence'],frame=r['frame'],**r['continuous']) for r in left]
        b=[dict(sequence=r['sequence'],frame=r['frame'],**r['continuous']) for r in right]
        result[metric]=paired_delta(a,b,metric=metric)
        if metric in ('cd_m','jsd_bev','jsd_3d'):
            delta=sorted([dict(sequence=x['sequence'],frame=x['frame'],delta=y[metric]-x[metric]) for x,y in zip(a,b)],key=lambda r:r['delta'],reverse=True)
            result[metric]['worst_frames']=delta[:10]
            result[metric]['worst_decile_mean']=float(np.mean([r['delta'] for r in delta[:90]]))
    return result


def compare(reference,candidate):
    if reference['frames']!=900 or candidate['frames']!=900:raise ValueError('No partial evaluation decisions')
    stage1={}
    for n in ('1','2','4','8'):
        a=reference['stage1'][n]['rows'];b=candidate['stage1'][n]['rows'];aligned(a,b)
        stage1[n]={m:paired_delta(a,b,metric=m) for m in ('iou','f1','precision','recall','fine_recall_ceiling','active_parents')}
    fixed={}
    for k in ('84963','120000'):
        a=reference['fixed_k_rows'][k];b=candidate['fixed_k_rows'][k];aligned(a,b)
        fixed[k]={m:paired_delta(a,b,metric=m) for m in ('iou','fine_recall_ceiling')}
    fine={key:fine_comparison(reference['stage2'][key]['rows'],r['rows'],fixed_support=key.startswith('teacher')) for key,r in candidate['stage2'].items()}
    return {'stage1':stage1,'fixed_k':fixed,'fine':fine}


def timing_summary(folder):
    rows=[]
    for path in sorted(folder.glob('round*/*.json')):
        value=json.loads(path.read_text())
        if 'rows' not in value:continue
        rows.append({'file':str(path),'gpu':value['physical_gpu'],'variant':path.stem,'summary':value['summary'],
                     'peak_allocated_gib':value['peak_allocated_gib'],'peak_reserved_gib':value['peak_reserved_gib']})
    by={}
    for row in rows:by.setdefault(row['variant'],[]).append(row)
    if any(len(v)!=3 for v in by.values()) or not by:raise ValueError('Three rounds per timed arm required')
    if len({r['gpu'] for r in rows})!=1:raise ValueError('Same physical GPU required for paired speedups')
    paired={}
    for name in by:
        if name=='EF0':continue
        values=[];round_means=[]
        for i in (1,2,3):
            base=json.loads((folder/f'round{i}/EF0.json').read_text())['rows']
            alt=json.loads((folder/f'round{i}/{name}.json').read_text())['rows']
            if [(r['sequence'],r['frame'],r['input_sha256']) for r in base]!=[(r['sequence'],r['frame'],r['input_sha256']) for r in alt]:raise ValueError('Timing pair mismatch')
            delta=[b['generation_wall_ms']-a['generation_wall_ms'] for a,b in zip(base,alt)]
            values.append(delta);round_means.append(float(np.mean(delta)))
        # Resample whole paired rounds;20 different scenes are not20 repeats.
        rng=np.random.default_rng(20261034);means=np.asarray(round_means)
        boot=means[rng.integers(0,3,size=(4000,3))].mean(1)
        paired[name]={'paired_round_mean_ms':round_means,'mean_ms':float(means.mean()),
            'ci95_three_round_bootstrap_ms':np.quantile(boot,[.025,.975]).tolist(),
            'per_frame_pair_ms':values,'speedup_established':bool(np.quantile(boot,.975)<0),
            'limitation':'Only3 paired repetitions; CI describes timing, not training-seed stability'}
    return {'rounds':rows,'paired_generation':paired}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--phase',choices=('P1','P2'),required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--arms',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=json.loads(a.reference.read_text());results={}
    for arm in a.arms:
        value=json.loads((a.root/'evaluations'/f'{arm}.json').read_text());delta=compare(ref,value)
        deploy=delta['fine']['deploy_nfe4'];teacher=delta['fine']['teacher_nfe4']
        baseline_cd=ref['stage2']['deploy_nfe4']['continuous_macro']['cd_m']
        compatibility=(deploy['native_fine_iou']['mean']>=-.005 and deploy['cd_m']['mean']<=baseline_cd*.005
                       and all(deploy[m]['mean']>=-.002 for m in ('iou05','iou02','iou01')))
        if a.phase=='P1':
            parent=delta['stage1']['2']['iou'];ceilings=[v['fine_recall_ceiling'] for v in delta['fixed_k'].values()]
            target=(parent['mean']>=.003 and parent['ci95'][0]>0) or any(v['mean']>=.005 and v['ci95'][0]>0 for v in ceilings)
        else:
            target=deploy['native_fine_iou']['mean']>=.01 and deploy['native_fine_iou']['ci95'][0]>0
        jsd_review=any(deploy[m]['mean']>0 for m in ('jsd_bev','jsd_3d'))
        decision='QUALITY_RUNTIME_REVIEW' if target and compatibility else 'NO_GO'
        if a.phase=='P2' and teacher['native_fine_iou']['mean']>=.01 and not target:decision='TEACHER_SUPPORT_ONLY_RESEARCH_RESULT'
        results[arm]={'delta':delta,'summary':{k:v['summary'] for k,v in value['stage1'].items()},
            'decision':decision,'target_reached':bool(target),'deployment_compatibility':bool(compatibility),
            'jsd_requires_review':bool(jsd_review),'no_automatic_model_adoption':True}
    result={'status':'COMPLETE','phase':a.phase,'results':results,'timing':timing_summary(a.root/'timing'),
        'single_training_seed':20261032,'confidence_scope':'Sequence/time-block paired900 frames; not multiple training seeds',
        'p3_formal_training_authorized':False,'policy':'Preserve every negative result; no budget extension or automatic second seed'}
    atomic_json(a.output,result)
    lines=[f'# {a.phase} Encoder adaptation results','','| Arm | Decision | Target | Deploy compatibility |','|---|---|---|---|']
    lines += [f"| {arm} | {v['decision']} | {v['target_reached']} | {v['deployment_compatibility']} |" for arm,v in results.items()]
    lines += ['','Full paired differences, grouped CI, worst frames and three-round timing are in the companion JSON. These are mature-checkpoint+20K adaptation results, not scratch20K architecture results. P3 formal training remains unstarted.']
    a.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

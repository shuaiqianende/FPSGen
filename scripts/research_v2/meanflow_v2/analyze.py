"""Loss-tail audit and final matched 41-frame comparison; never trains a model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,csv,json,hashlib
import numpy as np
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-explicit-fp16-training-v1/experiments/desescene_explicit_fp16_v1')
SNAPSHOT=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-all-experiment-data-publish-20261027/docs/research_v2/all_experiment_data_20261027')


def summary(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if not len(x):return dict(n=0,mean=None,median=None,p95=None,p99=None,max=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=float(x.max()))


def baseline(out):
    out.mkdir(parents=True,exist_ok=False)
    files=['README.md','stage1_scratch20k_full_metrics.csv','training_loss_200_500.csv','dataset_parts.json','training_run_registry.csv']
    identity={f:dict(path=str(SNAPSHOT/f),sha256=file_sha(SNAPSHOT/f),bytes=(SNAPSHOT/f).stat().st_size) for f in files}
    rows=[json.loads(x) for x in (BASE/'S1-MeanFlow-Scratch20K/formal/train.jsonl').open()]
    groups={k:[] for k in ('diagonal_only','mixed','finite_only')}
    for row in rows:
        key='diagonal_only' if row['finite_interval_exposures']==0 else 'finite_only' if row['diagonal_exposures']==0 else 'mixed'
        groups[key].append(row)
    r=dict(snapshot_commit='d6a3a60',inputs=identity,attempts=len(rows),successful_updates=sum(x['finite'] for x in rows),
           amp_skips=sum(not x['finite'] for x in rows),loss=summary([x['loss'] for x in rows]),
           last200_successful=summary([x['loss'] for x in rows if x['finite']][-200:]),
           last200_attempts=summary([x['loss'] for x in rows][-200:]),
           groups={k:dict(attempts=len(a),loss=summary([x['loss'] for x in a]),loss_gt10=sum(x['loss']>10 for x in a),
               amp_skips=sum(not x['finite'] for x in a)) for k,a in groups.items()},
           limitation='Original logs omit t/s/delta, per-scene MSE and JVP norm; long-gap/JVP causality NOT TESTED',
           top10=sorted(rows,key=lambda x:x['loss'],reverse=True)[:10],
           source_log_sha256=file_sha(BASE/'S1-MeanFlow-Scratch20K/formal/train.jsonl'))
    atomic_json(out/'original_loss_audit.json',r)
    (out/'README.md').write_text('# Original MeanFlow evidence audit\n\n'
       'Snapshot d6a3a60; original files remain read-only.\n\n'
       + '\n'.join(f'- {k}: {v}' for k,v in r['groups'].items())
       +'\n\nNo logged t/s or JVP norms: cannot establish a causal long-gap mechanism.\n')


def gap_name(d):
    if d==0:return '0'
    edges=[0.,.125,.25,.5,.75,1.]
    for a,b in zip(edges[:-1],edges[1:]):
        if a<d<=b:return f'({a},{b}]'
    raise ValueError('Invalid temporal gap')


def losses(run,out):
    rows=[json.loads(x) for x in (run/'train.jsonl').open()]
    bins={}
    for r in rows:
        for s in r['scene_diagnostics']:
            q=dict(s,gradient_norm=r['gradient_norm'],gradient_clipped=r['gradient_clipped'],
                   optimizer_succeeded=r['finite'],update=r['updates'],attempt=r['attempts'])
            bins.setdefault(gap_name(s['delta']),[]).append(q)
    fields=['raw_mse','weighted_loss','weight','jvp_rms','jvp_max_abs','conditional_velocity_rms','v_est_rms',
            'v_est_conditional_error_rms','derivative_target_term_rms','gradient_norm','positive_mse','negative_mse','t','s','delta']
    result=dict(attempts=len(rows),successful_updates=sum(r['finite'] for r in rows),
        amp_skips=sum(not r['finite'] for r in rows),
        per_delta={b:dict(scenes=len(q),amp_skip_scene_count=sum(not r['optimizer_succeeded'] for r in q),
            gradient_clipped_scene_count=sum(bool(r['gradient_clipped']) for r in q),
            **{k:summary([r.get(k) for r in q]) for k in fields}) for b,q in bins.items()},
        note='Gradient norm is a batch statistic associated with each scene, not a separate per-scene gradient.',
        last200_raw=summary([r['raw_unweighted_mse'] for r in rows if r['finite']][-200:]),
        last200_weighted=summary([r['weighted_loss'] for r in rows if r['finite']][-200:]),
        update_seconds=summary([r['seconds'] for r in rows]),
        measured_gpu_process_seconds=float(sum(r['seconds'] for r in rows)))
    atomic_json(out/'loss_interval_summary.json',result)
    for width in (200,500):
        table=[]
        successful=[r for r in rows if r['finite']]
        for i in range(0,len(successful),width):
            q=successful[i:i+width]
            table.append(dict(first=q[0]['updates'],last=q[-1]['updates'],n=len(q),
                raw_mse_mean=float(np.mean([r['raw_unweighted_mse'] for r in q])),
                weighted_loss_mean=float(np.mean([r['weighted_loss'] for r in q])),
                gradient_norm_p99=float(np.percentile([r['gradient_norm'] for r in q],99))))
        with (out/f'loss_{width}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)


def bootstrap(values):
    # Time-block resampling on the one seq08 sequence; no new training seed.
    blocks=[np.asarray(values[i:i+5]) for i in range(0,len(values),5)]
    rng=np.random.default_rng(20261032)
    samples=[float(np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]).mean()) for _ in range(4000)]
    return dict(mean=float(np.mean(values)),ci95=np.percentile(samples,[2.5,97.5]).tolist(),
        method='paired contiguous5-frame block bootstrap; one training seed; not cross-seed confidence')


def compare(evaluation,out):
    actual=[json.loads(x) for x in (evaluation/'per_frame.jsonl').open()]
    result={}
    for name in ('S1-FM-Scratch20K','S1-MeanFlow-Scratch20K'):
        old=[json.loads(x) for x in (BASE/name/'evaluation/seq08_41/per_frame.jsonl').open()]
        lookup={(r['frame'],r['nfe']):r for r in old}
        result[name]={}
        for nfe in (1,2,4,8):
            q=sorted([r for r in actual if r['nfe']==nfe],key=lambda r:r['frame'])
            if len(q)!=41:raise ValueError('Incomplete frozen41 evaluation')
            fields={'parent_iou':lambda r:r['parent']['iou'],'recall':lambda r:r['parent']['recall'],
                'ceiling':lambda r:r['parent']['fine_recall_ceiling'],'cd_m':lambda r:r['continuous']['cd_m'],
                'fixed84963_iou':lambda r:r['fixed_k'][0]['iou'],'fixed120000_iou':lambda r:r['fixed_k'][1]['iou']}
            result[name][nfe]={}
            for key,fn in fields.items():
                values=[fn(r)-fn(lookup[(r['frame'],nfe)]) for r in q]
                result[name][nfe][key]=dict(**bootstrap(values),worst_frame=q[int(np.argmax(values) if key=='cd_m' else np.argmin(values))]['frame'])
    atomic_json(out/'paired_comparison.json',result)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline',action='store_true');p.add_argument('--run',type=Path)
    p.add_argument('--evaluation',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.baseline:baseline(a.output);return
    a.output.mkdir(parents=True,exist_ok=False)
    if a.run:losses(a.run,a.output)
    if a.evaluation:compare(a.evaluation,a.output)


if __name__=='__main__':main()

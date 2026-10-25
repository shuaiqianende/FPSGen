"""Grouped paired uncertainty, long tails and preregistered noninferiority screens."""
import numpy as np


def grouped_interval(values,groups,*,seed=20261032,draws=2000):
    values=np.asarray(values,dtype=float);unique=sorted(set(groups));lookup={g:[] for g in unique}
    for i,g in enumerate(groups):lookup[g].append(i)
    rng=np.random.default_rng(seed);means=[]
    for _ in range(draws):
        selected=rng.integers(0,len(unique),len(unique));ix=np.concatenate([lookup[unique[i]] for i in selected])
        means.append(values[ix].mean())
    return list(map(float,np.quantile(means,[.025,.975])))


def paired_analysis(rows,arm):
    lookup={(r['sequence'],r['frame'],r['nfe'],r['variant']):r for r in rows};comparisons=[]
    metrics={'parent_iou':lambda r:r['parent']['iou'],'fine_recall_ceiling':lambda r:r['parent']['fine_recall_ceiling'],
        'native_iou':lambda r:r['native']['fine']['iou'],
        **{k:(lambda r,k=k:r['continuous'][k]) for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')}}
    reference_nfe=2 if arm=='S1-MF20K' else 4
    for nfe in (1,2,4,8):
        for control_nfe in sorted({nfe,reference_nfe}):
            pairs=[]
            for (s,f,n,v),r in lookup.items():
                if n==nfe and v=='meanflow':pairs.append((r,lookup[(s,f,control_nfe,'matched_fm')]))
            result=dict(meanflow_nfe=nfe,control_nfe=control_nfe,frames=len(pairs),metrics={})
            groups=[(a['sequence'],int(a['frame'])//100) for a,b in pairs]
            for key,fn in metrics.items():
                delta=np.array([fn(a)-fn(b) for a,b in pairs]);order=np.argsort(delta)
                # Higher scores are good, lower CD/JSD are good.
                regression=delta if key in ('cd_m','jsd_bev','jsd_3d') else -delta
                worst=np.argsort(regression)[::-1]
                result['metrics'][key]=dict(delta_mean=float(delta.mean()),grouped_95ci=grouped_interval(delta,groups),
                    worst_tenth_mean_delta=float(delta[worst[:max(1,int(np.ceil(.1*len(delta))))]].mean()),
                    worst_frames=[dict(sequence=pairs[i][0]['sequence'],frame=pairs[i][0]['frame'],delta=float(delta[i])) for i in worst[:10]])
            if control_nfe==reference_nfe:
                m=result['metrics'];cd_ref=np.mean([b['continuous']['cd_m'] for a,b in pairs]);cdrel=m['cd_m']['delta_mean']/max(cd_ref,1e-12)
                if arm=='S1-MF20K':
                    passed=(m['parent_iou']['delta_mean']>=-.003 and m['fine_recall_ceiling']['delta_mean']>=-.005)
                else:passed=m['native_iou']['delta_mean']>=-.005 and cdrel<=.02
                result['screen']=dict(quality_mean_screen=bool(passed),relative_cd_change=float(cdrel),
                    decision='REQUIRES_PAIRED_RUNTIME_AND_GEOMETRY_REVIEW' if passed else 'NO_GO',
                    only_one_training_seed=True,ci_scope='Sequence/100-frame blocks, not cross-training-seed uncertainty')
            comparisons.append(result)
    return comparisons

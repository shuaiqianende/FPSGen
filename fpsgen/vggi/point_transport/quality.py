"""Full-cloud set, correspondence and collapse diagnostics (no training).

Exact CPU KD-tree queries avoid an NxN matrix. Metres throughout. DCD follows
fpsgen.ops.dcd (equal cardinality, lambda=1, count epsilon=1e-6). Set metrics
cannot evaluate a permutation, and interpolants are not learned model outputs.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

THRESHOLDS = (.05, .1, .2, .5)


def set_metrics(points, target, *, tree=None, target_tree=None, workers=4):
    if len(points) != len(target):
        raise ValueError('This DCD audit requires equal cardinality')
    if not len(points) or not np.isfinite(points).all() or not np.isfinite(target).all():
        raise ValueError('Finite nonempty clouds required')
    tree = tree or cKDTree(points)
    target_tree = target_tree or cKDTree(target)
    dp, ip = target_tree.query(points, workers=workers)
    dt, it = tree.query(target, workers=workers)
    cp = np.bincount(ip, minlength=len(target))[ip]
    ct = np.bincount(it, minlength=len(points))[it]
    result = dict(cd_l1_mean_m=float((dp.mean()+dt.mean())/2),
                  cd_l2_sum_m2=float(np.mean(dp*dp)+np.mean(dt*dt)))
    for alpha in (1., 100.):
        dcd = .5*(np.mean(1-np.exp(-alpha*dp*dp)/(cp+1e-6))+
                   np.mean(1-np.exp(-alpha*dt*dt)/(ct+1e-6)))
        result[f'dcd_alpha_{alpha:g}'] = float(dcd)
    for threshold in THRESHOLDS:
        p, r = float(np.mean(dp < threshold)), float(np.mean(dt < threshold))
        result[f'prf_{threshold:g}m'] = dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)
    return result


def collapse_metrics(points, *, tree=None, workers=4):
    tree = tree or cKDTree(points)
    unique, counts = np.unique(points, axis=0, return_counts=True)
    nn = tree.query(points, k=2, workers=workers)[0][:, 1]
    out = dict(duplicate_excess_fraction=float(1-len(unique)/len(points)),
               points_in_duplicate_groups_fraction=float(counts[counts>1].sum()/len(points)),
               max_coordinate_multiplicity=int(counts.max()),
               nn_distance_quantiles_m={str(q): float(np.quantile(nn,q)) for q in (.01,.05,.5,.95,.99)},
               nn_below_fraction={str(r): float(np.mean(nn<r)) for r in (.001,.01,.05)})
    out['ball_other_points'] = {}
    for radius in (.01,.05,.1):
        other = tree.query_ball_point(points, radius, return_length=True, workers=workers)-1
        out['ball_other_points'][str(radius)] = dict(mean=float(other.mean()),
            p99=float(np.quantile(other,.99)), maximum=int(other.max()),
            fraction_at_least_8=float(np.mean(other>=8)), fraction_at_least_32=float(np.mean(other>=32)))
    vox = np.floor(np.asarray(points)/.05).astype(np.int64)
    _, occupancy = np.unique(vox, axis=0, return_counts=True)
    out['voxel_005m_origin_zero'] = dict(occupied=len(occupancy), maximum=int(occupancy.max()),
        p99=float(np.quantile(occupancy,.99)), points_in_bins_ge8=float(occupancy[occupancy>=8].sum()/len(points)))
    return out


def pair_metrics(x, endpoint, neighbors):
    from .summarize import path_diagnostics
    d = np.linalg.norm(endpoint-x, axis=1)
    out = dict(mean_m=float(d.mean()), rms_m=float(np.sqrt(np.mean(d*d))),
        cost_m2=float(np.mean(d*d)), p95_m=float(np.quantile(d,.95)),
        p99_m=float(np.quantile(d,.99)), maximum_m=float(d.max()),
        over_1m=float(np.mean(d>1)), over_3m=float(np.mean(d>3)), over_5m=float(np.mean(d>5)))
    out.update(path_diagnostics(x, endpoint, neighbors))
    return out


def evaluate(scene, root):
    folder = root/scene
    output = folder/'quality_v1.json'
    if output.exists():
        return json.loads(output.read_text())
    started = time.monotonic()
    with np.load(folder/'clouds.npz') as f: x,y = f['source'],f['target']
    endpoints = {'gt_self':y, 'source':x}
    for name, filename in [('fast','online_v3_b64_i5000_e0.05_r12.permutation.npy'),
                            ('slow_geometry','round_none_b256_p16_permutation.npy'),
                            ('slow_guided','round_feasible_k16_i5000_e0.05_r1.5_b256_p16_permutation.npy'),
                            ('scaffold','scaffold.npy')]:
        p = np.load(folder/filename)
        np.testing.assert_array_equal(np.sort(p),np.arange(len(y)))
        endpoints[name] = y[p]
    yt, xt = cKDTree(y), cKDTree(x)
    endpoints['nearest'] = y[yt.query(x, workers=4)[1]]
    endpoints['random_permutation_control'] = y[np.random.default_rng(20261107).permutation(len(y))]
    with np.load(folder/'feasible_k16_i5000_e0.05_r1.5_endpoints.npz') as f:
        endpoints['sinkhorn_barycenter'] = f['barycentric']
        endpoints['sinkhorn_categorical'] = y[f['categorical_target_ids']]
    neighbors = xt.query(x,k=9,workers=4)[1][:,1:]
    result = dict(scene=scene,count=len(x),no_training=True,results={},intermediate={},fast_vs_slow_intermediate={})
    for name, cloud in endpoints.items():
        print(scene, 'endpoint', name, flush=True)
        tree = cKDTree(cloud)
        row = dict(set=set_metrics(cloud,y,tree=tree,target_tree=yt),
                   collapse=collapse_metrics(cloud,tree=tree))
        if name not in ('gt_self','source'):
            row['pair'] = pair_metrics(x,cloud,neighbors)
        result['results'][name] = row
    for tau in (.25,.5,.75):
        middle = {}
        for name in ('fast','slow_geometry','sinkhorn_barycenter','nearest','random_permutation_control'):
            print(scene,'intermediate',tau,name,flush=True)
            cloud = (1-tau)*x+tau*endpoints[name]
            tree = cKDTree(cloud)
            result['intermediate'].setdefault(name,{})[str(tau)] = dict(
                set_to_final_gt=set_metrics(cloud,y,tree=tree,target_tree=yt),
                collapse=collapse_metrics(cloud,tree=tree))
            if name in ('fast','slow_geometry'): middle[name] = cloud
        result['fast_vs_slow_intermediate'][str(tau)] = set_metrics(middle['fast'],middle['slow_geometry'])
    result['seconds'] = time.monotonic()-started
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('fpsgen/vggi/runs/point_transport_sinkhorn_v1'))
    p.add_argument('--scenes',nargs='+',default=['00_000500','00_001000','01_000500'])
    args=p.parse_args()
    results=[evaluate(scene,args.root) for scene in args.scenes]
    summary=dict(protocol=dict(cd_l1='half-sum of directional mean Euclidean distances, metres',
        cd_l2='sum of directional mean squared distances, metres squared',
        dcd='equal cardinality; alpha=1 and 100 m^-2, lambda=1, epsilon=1e-6; NN tie breaking may penalize duplicate GT coordinates',
        f1='Euclidean distance strictly below 0.05/0.1/0.2/0.5 metres; no scene normalization',
        scope='all 180K points; same source/target samples across methods; intermediate metrics are diagnostics, not learned outputs'),scenes=results)
    (args.root/'quality_summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=[]
    for r in results:
        for name,data in r['results'].items():
            s,c=data['set'],data['collapse']
            row=dict(scene=r['scene'],method=name,tau=1,cd_l1_m=s['cd_l1_mean_m'],cd_l2_m2=s['cd_l2_sum_m2'],
                dcd_a1=s['dcd_alpha_1'],dcd_a100=s['dcd_alpha_100'],f1_01=s['prf_0.1m']['f1'],
                duplicate_excess=c['duplicate_excess_fraction'],nn_p01=c['nn_distance_quantiles_m']['0.01'],
                nn_p50=c['nn_distance_quantiles_m']['0.5'],ball005_p99=c['ball_other_points']['0.05']['p99'])
            row.update({k:data.get('pair',{}).get(k) for k in ('rms_m','p99_m','over_5m','neighbor_velocity_rms_m','interior_path_approach_fraction')})
            rows.append(row)
    with (args.root/'quality_endpoints_v1.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote quality_summary_v1.json and quality_endpoints_v1.csv',flush=True)


if __name__=='__main__':main()

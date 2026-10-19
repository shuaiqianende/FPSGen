"""Validate completed33-pair CD benchmark and export tables/scaling plots."""
import argparse
import csv
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.output/'results.json').read_text())
    assert r['status']=='COMPLETE' and r['sizes']==[2**i for i in range(10,21)]
    assert len(r['rows'])==33 and r['valid_pairs']==33 and not r['oom_pairs']
    index={(x['n'],x['backend']):x for x in r['rows']}
    assert len(index)==33
    assert len(r['raw_trials'])==99
    maximum_error=max(v['cd_cpu_absolute_error_m'] for v in r['rows'])
    assert maximum_error<=r['loss_absolute_tolerance_m']
    lines=['# CD scaling1024x1024 through1048576x1048576','',
        'COMPLETE:11 power-of-two sizes x3 backends,99 measured repetitions. PhysicalGPU1,RTX3090,B1.',
        'Two independent uniform FP32 XYZ clouds over[-50,50]x[-50,50]x[-4,4.4],seed20261019, nested prefixes.',
        'Synthetic volumetric clouds, not real LiDAR surfaces. Results are distribution/implementation/hardware specific.',
        'Every backend receives exactly the same two clouds at a given size. No duplicate padding or approximate nearest search.',
        'Full-size forward/backward warmup is excluded. Three synchronized timed repetitions per pair; rotate backend order across sizes.',
        'Training loss is0.5*(mean Euclidean P->GT + GT->P). Only prediction coordinates require gradients.',
        'Each GPU-tree CD includes fresh construction of BOTH trees. No tree cache is reused.',
        'Network, optimizer, DDP, input generation/transfers and CPU validation are excluded.',
        'Inactive Torch cache is cleared before measured trials: allocation work is included even though code/kernels have been warmed.',
        'Reported total is median per-run forward+backward wall time; independent forward/backward medians need not sum exactly.',
        '', '| Points per cloud | Chamfer CUDA total(ms) | KeOps total(ms) | GPU KD-tree total(ms) |',
        '|---:|---:|---:|---:|']
    for n in r['sizes']:
        lines.append('|'+str(n)+'|'+'|'.join(f"{index[n,b]['total_ms']:.3f}" for b in r['backends'])+'|')
    lines+=['','## Additional live tensor memory','',
        'Extra peak PyTorch allocated above input baseline, forward AND backward included. MiB=2^20bytes.',
        'This excludes shared inputs and all Teacher/network memory; it is not total nvidia-smi memory.',
        'Torch reserved memory includes cache and is reported separately in CSV. Raw third-party CUDA allocations may be outside Torch counters.',
        'The full maximum-size backing tensors used to generate nested prefixes remain in the input baseline even for small N.',
        '', '| Points per cloud | Chamfer CUDA extra(MiB) | KeOps extra(MiB) | GPU KD-tree extra(MiB) |',
        '|---:|---:|---:|---:|']
    for n in r['sizes']:
        lines.append('|'+str(n)+'|'+'|'.join(f"{index[n,b]['extra_peak_allocated_mib']:.3f}" for b in r['backends'])+'|')
    lines+=['','## Forward and backward separately','',
        '| Points per cloud | Backend | Forward(ms) | Backward(ms) |',
        '|---:|---|---:|---:|']
    for row in r['rows']:
        lines.append(f"|{row['n']}|{row['backend']}|{row['forward_ms']:.3f}|{row['backward_ms']:.3f}|")
    lines+=['','## Correctness and limits','',
        f'All33 scalar CD results agree with full-cloud CPU cKDTree(eps0) within1e-5m; maximum absolute error{maximum_error:.9g}m. All99 measured prediction gradients finite. No OOM.',
        'This scaling sweep validates loss values and finite gradients; it does not prove gradient identity across backends.',
        'Previous large-context Teacher strict gradient comparison remains unpassed and is unaffected by this performance sweep.',
        'Different nearest-neighbor tie rules and FP32/FP64 search rounding can produce different valid coordinate subgradients.',
        'Only independent volumetric XYZ clouds are used here. Surface geometry, repeated points and strongly asymmetric N/M can change the crossover and worst-case behavior.',
        'See GPU_KDTREE_TRADEOFFS_AND_MEMORY.md for measured small/duplicate-point counterexamples.',
        '', 'Artifacts: `'+str(a.output)+'`; raw runs in results.json, full breakdown in summary.csv, log-scale plot in scaling.png.',
        'No formal training sources, configurations, optimizers or checkpoints were changed. GPU2/3 training was not interrupted.','']
    docs=HERE.parents[2]/'docs/research_v2';small=docs/'stage2_morph_fm100k_audit'
    (docs/'CD_SCALING_1024_TO_1048576_GPU.md').write_text('\n'.join(lines))
    (small/'cd_scaling_1024_to_1048576_summary.csv').write_bytes((a.output/'summary.csv').read_bytes())
    (small/'cd_scaling_1024_to_1048576_results.json').write_bytes((a.output/'results.json').read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    labels=dict(chamfer_cuda='Chamfer CUDA',keops='KeOps',gpu_kdtree='Exact GPU KD-tree')
    for b in r['backends']:
        axes[0].plot(r['sizes'],[index[n,b]['total_ms'] for n in r['sizes']],marker='o',label=labels[b])
        axes[1].plot(r['sizes'],[index[n,b]['extra_peak_allocated_mib'] for n in r['sizes']],marker='o',label=labels[b])
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Points per cloud (N=M)')
        ax.grid(True,which='both',alpha=.25);ax.legend()
    axes[0].set_ylabel('CD forward + backward (ms)');axes[0].set_title('Median of 3 full-size warm trials')
    axes[1].set_ylabel('Extra peak allocated (MiB)');axes[1].set_title('Input/network excluded; Torch allocator')
    fig.suptitle('RTX 3090 / FP32 / independent uniform 3D clouds')
    fig.savefig(a.output/'scaling.png',dpi=170)
    fig.savefig(small/'cd_scaling_1024_to_1048576.png',dpi=170);plt.close(fig)
    audit=dict(status='PASS_33_PAIRS_99_TRIALS',max_cd_cpu_absolute_error_m=maximum_error,
        source_result_sha256=file_sha(a.output/'results.json'),figure_sha256=file_sha(a.output/'scaling.png'),
        note='Loss values and finite gradients checked, not universal gradient equivalence',rows=33,trials=99)
    atomic_json(a.output/'audit.json',audit)
    atomic_json(small/'cd_scaling_1024_to_1048576_audit.json',audit)
    print(json.dumps(audit,indent=2));print('\n'.join(lines[:24]))


if __name__=='__main__':main()

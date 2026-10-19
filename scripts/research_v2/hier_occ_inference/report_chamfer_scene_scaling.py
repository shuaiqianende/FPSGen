"""Scene CUDA scaling report: separate brute-force optimization from BVH gain."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.output/'results.json').read_text());assert r['status']=='COMPLETE' and r['valid_pairs']==55 and r['oom_pairs']==0 and r['legacy_sources_unchanged']
    assert len(r['rows'])==55 and len(r['raw_trials'])==165
    index={(v['n'],v['backend']):v for v in r['rows']};assert len(index)==55
    error=max(v['cd_cpu_absolute_error_m'] for v in r['rows']);assert error<=1e-5
    docs=HERE.parents[2]/'docs/research_v2';small=docs/'stage2_morph_fm100k_audit'
    labels=dict(chamfer_cuda='Original CUDA',scene_tiled='Scene tiled',scene_bvh='Scene BVH',keops='KeOps',gpu_kdtree='Prior GPU KD-tree')
    lines=['# Chamfer CUDA for Scene: independent implementation and scaling','',
        'COMPLETE:55 combinations,165 measured trials on physicalGPU1 /RTX3090. All original Chamfer source hashes unchanged.',
        'Original upstream README benchmarks B32x1000/2000; it does not impose an8192-point limit or prove an object-only design.',
        'Inspection shows the original dim3(32,16) grid activates only the16 query-axis blocks for B1, and scans all support points.',
        'This is evidence of a scene workload mismatch, not a claim that CUDA itself should lose to CPU.',
        '', '## Independent implementation','',
        '`fpsgen/ops/chamfer_scene.py` and `chamfer_scene_cuda.cu` define a separate extension/cache, not a patch to chamfer_3D.',
        '- tiled: query-sized dynamic CUDA grid; supports staged in shared SoA tiles; keep nearest minimum in registers through all tiles.',
        '- bvh: single GPU Morton sort per cloud, balanced leaf hierarchy, conservative bounding-box pruning, spatially ordered queries, original-order distance/index restoration.',
        '- CUDA coordinate backward uses saved original nearest indices with atomic accumulation; denseB and ragged lists keep scene identities separate.',
        '- Exact search, no point sampling, finite radius, approximate visit budget or dense NxM tensor. The BVH result uses FP32 distance arithmetic.',
        '- Zero-distance sqrt loss has explicitly finite zero coordinate derivative. First-order coordinate gradients only; no double-backward promise.',
        '- Deterministic minimum-original-ID tie rule; arbitrary ties/FP32 rounding may still differ from another implementation.',
        '- Default auto dispatch uses point-count pair budget, not labels. Explicit modes remain available and are benchmarked independently.',
        '', '## Controlled benchmark','',
        'Two independent uniform FP32 3D clouds, same seed/range/nested prefixes as preceding scaling study. Synthetic volume, not LiDAR geometry.',
        'Full-shape forward/backward warmup then3 measured repetitions. Synchronized wall time, includes allocation, loss and prediction-only backward; excludes model/I/O/DDP.',
        'Every BVH/KD call builds fresh trees. Backend order rotates by size. This measures no-grad index selection plus differentiable loss, not Teacher optimizer throughput.',
        'The tiled improvement and the additional BVH algorithmic gain are reported separately.',
        '', '| N=M | Original CUDA(ms) | Scene tiled(ms) | Scene BVH(ms) | KeOps(ms) | Prior KD-tree(ms) |',
        '|---:|---:|---:|---:|---:|---:|']
    for n in r['sizes']:lines.append('|'+str(n)+'|'+'|'.join(f"{index[n,b]['total_ms']:.3f}" for b in r['backends'])+'|')
    lines+=['','## Extra peak PyTorch allocated memory','',
        'Includes CD forward and backward workspace, excludes shared input baseline and network. Third-party raw CUDA allocations may not appear in Torch counters.',
        'Full maximum-size backing point tensors are in the common input baseline for all sizes. Reserved cache is separately recorded in CSV.',
        '', '| N=M | Original CUDA(MiB) | Scene tiled(MiB) | Scene BVH(MiB) | KeOps(MiB) | Prior KD-tree(MiB) |',
        '|---:|---:|---:|---:|---:|---:|']
    for n in r['sizes']:lines.append('|'+str(n)+'|'+'|'.join(f"{index[n,b]['extra_peak_allocated_mib']:.3f}" for b in r['backends'])+'|')
    large=index[1048576,'scene_bvh'];old=index[1048576,'chamfer_cuda'];tile=index[1048576,'scene_tiled']
    lines+=['',f'At1048576x1048576, pure tiled optimization is{old["total_ms"]/tile["total_ms"]:.2f}x versus original; BVH adds spatial pruning to reach{old["total_ms"]/large["total_ms"]:.2f}x in this run.',
        'This is not a universal speedup: original timing fluctuated strongly and all min/max repetitions remain in CSV/JSON.',
        f'Max scalar CD difference versus CPU exact KD-tree across55 cases:{error:.9g}m. All165 measured coordinate gradients finite; no OOM.',
        'Scalar agreement and finite gradients do not establish gradient identity or convergence equivalence. Real Teacher endpoint parity is reported separately.',
        'Small inputs may favor original CUDA. Pathological spatial distributions can weaken BVH pruning; nearest-neighbor gradients are nonunique at ties.',
        'The custom CUDA backward accumulates atomically and is not guaranteed bitwise deterministic.',
        '', 'Artifacts: `'+str(a.output)+'` contains complete raw timing/memory runs,CSV,original hashes,and source snapshot.',
        'No formal training pipeline was switched automatically.','']
    (docs/'CHAMFER_CUDA_FOR_SCENE_SCALING.md').write_text('\n'.join(lines))
    (small/'chamfer_scene_scaling_results.json').write_bytes((a.output/'results.json').read_bytes())
    (small/'chamfer_scene_scaling_summary.csv').write_bytes((a.output/'summary.csv').read_bytes())
    (small/'chamfer_scene_scaling_source_hashes.json').write_bytes((a.output/'source_hashes.json').read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for b in r['backends']:
        axes[0].plot(r['sizes'],[index[n,b]['total_ms'] for n in r['sizes']],marker='o',label=labels[b])
        axes[1].plot(r['sizes'],[index[n,b]['extra_peak_allocated_mib'] for n in r['sizes']],marker='o',label=labels[b])
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(True,which='both',alpha=.25)
        ax.set_xlabel('Points per cloud (N=M)');ax.legend(fontsize=9)
    axes[0].set_ylabel('CD forward + backward(ms)');axes[1].set_ylabel('Extra peak allocated(MiB)')
    fig.suptitle('Independent Chamfer CUDA for Scene / RTX3090 / FP32 uniform XYZ')
    fig.savefig(a.output/'scene_scaling.png',dpi=170);fig.savefig(small/'chamfer_scene_scaling.png',dpi=170);plt.close(fig)
    audit=dict(status='PASS_55_PAIRS_165_TRIALS',legacy_sources_unchanged=True,maximum_cd_error_m=error,
        note='Performance/loss/finite-gradient checks only; real Teacher gradient parity separate',
        results_sha256=file_sha(a.output/'results.json'),original_source_hashes=r['legacy_sources_before'])
    atomic_json(a.output/'audit.json',audit);atomic_json(small/'chamfer_scene_scaling_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()

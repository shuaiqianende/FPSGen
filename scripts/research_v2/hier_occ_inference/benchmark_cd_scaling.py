"""CD scaling1024..1048576: warm synchronized forward/backward and workspace."""
import argparse
import csv
import gc
import json
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import numpy as np
from scipy.spatial import cKDTree
import torch
from export_stage2_snapshot import admit_gpu1
from fpsgen.ops.gpu_kdtree import load_extension
from fpsgen.ops.chamfer import Chamfer3DDist
from fpsgen.utils.gpu_kdtree_training import chamfer_sqrt_gpu_kdtree
from fpsgen.voxel_point.metrics import chamfer_sqrt
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def timed(fn):
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    tick=time.perf_counter();start.record();v=fn();end.record();end.synchronize()
    return v,dict(wall_seconds=time.perf_counter()-tick,cuda_event_seconds=start.elapsed_time(end)/1000.)


def median(values):return statistics.median(values)


def write_csv(output,rows):
    if not rows:return
    with output.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    a=parser.parse_args()
    if a.repeats<3:raise ValueError('At least3 warm measured repetitions required')
    if a.output.exists():raise ValueError('Preserve prior benchmark')
    a.output.mkdir(parents=True)
    admission=admit_gpu1();torch.set_num_threads(4)
    report=dict(status='RUNNING',admission=admission,seed=20261019,
        sizes=[2**p for p in range(10,21)],backends=['chamfer_cuda','keops','gpu_kdtree'],
        distribution='Two independently generated FP32 uniform XYZ clouds in [-50,50]x[-50,50]x[-4,4.4]; nested prefixes, separate RNG draws; no ROI crop; B1',
        timing='Full-shape warmup then3 measured trials; synchronized wall time and CUDA events; prediction-coordinate backward only; fresh GPU tree builds included',
        excluded=['network','optimizer','DDP','input_generation','CPU_reference','cold_extension_compilation','host_transfer'],
        memory='Extra peak PyTorch allocated over identical live input baseline, includes forward/backward; reserved separately. May exclude raw allocations outside Torch by third-party libraries.',
        precision='FP32 input and differentiable norm/reduction. GPU tree uses FP64 search distances; CPU exact KD-tree reference FP64. Native CUDA squared-distance/sqrt arithmetic FP32.',
        repeats=a.repeats,rows=[],raw_trials=[],loss_absolute_tolerance_m=1e-5,
        reference='SciPy cKDTree eps0,workers4 scalar CD on every full-sized point set, outside GPU timing',
        source_sha256=file_sha(Path(__file__)))
    atomic_json(a.output/'results.json',report)
    extension=load_extension('/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/gpu_kdtree_v1')
    native=Chamfer3DDist()
    def cuda_loss(p,q):
        dp,dq,_,_=native(p[None],q[None]);return .5*(dp.sqrt().mean()+dq.sqrt().mean())
    fns=dict(chamfer_cuda=cuda_loss,keops=lambda p,q:chamfer_sqrt(p,q,backend='keops'),
        gpu_kdtree=lambda p,q:chamfer_sqrt_gpu_kdtree(p,q,extension=extension))
    gen=torch.Generator(device='cuda').manual_seed(report['seed'])
    scale=torch.tensor([100.,100.,8.4],device='cuda');origin=torch.tensor([-50.,-50.,-4.],device='cuda')
    maximum=max(report['sizes'])
    fullp=torch.rand((maximum,3),device='cuda',generator=gen)*scale+origin
    fullq=torch.rand((maximum,3),device='cuda',generator=gen)*scale+origin
    for position,n in enumerate(report['sizes']):
        pbase,q=fullp[:n].contiguous(),fullq[:n].contiguous()
        cpu_p,cpu_q=pbase.cpu().numpy(),q.cpu().numpy()
        tick=time.perf_counter()
        cpu_cd=.5*(cKDTree(cpu_q,leafsize=32).query(cpu_p,eps=0.,workers=4)[0].mean()+
                   cKDTree(cpu_p,leafsize=32).query(cpu_q,eps=0.,workers=4)[0].mean())
        reference_seconds=time.perf_counter()-tick
        # Rotate backend order per size to avoid always giving the same backend cold clocks.
        names=report['backends'][position%3:]+report['backends'][:position%3]
        for name in names:
            fn=fns[name];trials=[]
            try:
                # Warm BOTH forward and backward on the actual complete shape.
                p=pbase.clone().requires_grad_()
                value=fn(p,q);torch.autograd.grad(value,p);torch.cuda.synchronize();del p,value
                for repeat in range(a.repeats):
                    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                    p=pbase.clone().requires_grad_()
                    baseline=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
                    torch.cuda.reset_peak_memory_stats()
                    value,forward=timed(lambda:fn(p,q))
                    gradient,backward=timed(lambda:torch.autograd.grad(value,p)[0])
                    loss=float(value.detach());finite=bool(torch.isfinite(gradient).all())
                    row=dict(n=n,backend=name,repeat=repeat,loss=loss,gradient_finite=finite,
                        forward_wall_seconds=forward['wall_seconds'],backward_wall_seconds=backward['wall_seconds'],
                        forward_cuda_seconds=forward['cuda_event_seconds'],backward_cuda_seconds=backward['cuda_event_seconds'],
                        total_wall_seconds=forward['wall_seconds']+backward['wall_seconds'],
                        baseline_allocated_bytes=baseline,baseline_reserved_bytes=reserved,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-baseline,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    trials.append(row);report['raw_trials'].append(row)
                    del p,value,gradient
                value=dict(n=n,backend=name,status='PASS' if all(t['gradient_finite'] for t in trials) and abs(trials[-1]['loss']-cpu_cd)<=1e-5 else 'FAIL',
                    forward_ms=1000*median(t['forward_wall_seconds'] for t in trials),
                    backward_ms=1000*median(t['backward_wall_seconds'] for t in trials),
                    total_ms=1000*median(t['total_wall_seconds'] for t in trials),
                    total_min_ms=1000*min(t['total_wall_seconds'] for t in trials),
                    total_max_ms=1000*max(t['total_wall_seconds'] for t in trials),
                    extra_peak_allocated_mib=median(t['extra_peak_allocated_bytes'] for t in trials)/1024**2,
                    total_peak_reserved_mib=median(t['peak_reserved_bytes'] for t in trials)/1024**2,
                    cpu_reference_cd_m=float(cpu_cd),cd_m=trials[-1]['loss'],
                    cd_cpu_absolute_error_m=abs(trials[-1]['loss']-cpu_cd),cpu_reference_seconds=reference_seconds,
                    cuda_event_forward_ms=1000*median(t['forward_cuda_seconds'] for t in trials),
                    cuda_event_backward_ms=1000*median(t['backward_cuda_seconds'] for t in trials))
            except torch.cuda.OutOfMemoryError as error:
                value=dict(n=n,backend=name,status='OOM',error=str(error))
                torch.cuda.empty_cache()
            report['rows'].append(value);atomic_json(a.output/'results.json',report)
            write_csv(a.output/'summary.csv',report['rows'])
            atomic_json(a.output/'progress.json',dict(completed_pairs=len(report['rows']),requested_pairs=33,latest=value,status='RUNNING'))
            print(json.dumps(value),flush=True)
            if value['status']=='FAIL':raise RuntimeError('Loss or finite-gradient check failed; preserve logs')
        del cpu_p,cpu_q,pbase,q
    report['rows'].sort(key=lambda r:(r['n'],report['backends'].index(r['backend'])))
    report.update(status='COMPLETE',valid_pairs=sum(r['status']=='PASS' for r in report['rows']),
        oom_pairs=sum(r['status']=='OOM' for r in report['rows']))
    atomic_json(a.output/'results.json',report);write_csv(a.output/'summary.csv',report['rows'])
    atomic_json(a.output/'progress.json',dict(status='COMPLETE',completed_pairs=len(report['rows']),requested_pairs=33))
    print('COMPLETE',a.output,flush=True)


if __name__=='__main__':main()

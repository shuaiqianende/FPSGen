"""Final scene-release smoke after the valid-index FP32 overflow guard fix."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import torch
from export_stage2_snapshot import admit_gpu1
from fpsgen.ops.chamfer_scene import SceneChamferDistance,load_scene_extension,BUILD
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve release audit')
    report=dict(status='RUNNING',admission=admit_gpu1(),checks=[],
        patch='Initialize tiled nearest index to valid0 when all squared distances overflow FP32; normal finite d2 algorithm unchanged')
    atomic_json(a.output,report);torch.set_num_threads(4);load_scene_extension()
    previous=Path('/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/chamfer_scene_scaling_1024_to_1048576_gpu1/results.json')
    old=json.loads(previous.read_text());index={(r['n'],r['backend']):r for r in old['rows']}
    gen=torch.Generator(device='cuda').manual_seed(20261019)
    scale=torch.tensor([100.,100.,8.4],device='cuda');origin=torch.tensor([-50.,-50.,-4.],device='cuda')
    fullp=torch.rand((1048576,3),device='cuda',generator=gen)*scale+origin
    fullq=torch.rand((1048576,3),device='cuda',generator=gen)*scale+origin
    for n in [8192,65536,1048576]:
        for mode in ['tiled','bvh']:
            metric=SceneChamferDistance(mode);trials=[]
            for i in range(4):
                gc.collect();torch.cuda.empty_cache();p=fullp[:n].clone().requires_grad_();q=fullq[:n]
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                loss=metric.sqrt_loss(p,q);gradient=torch.autograd.grad(loss,p)[0];torch.cuda.synchronize()
                trials.append(dict(warmup=i==0,seconds=time.perf_counter()-tick,loss=float(loss),gradient_finite=bool(torch.isfinite(gradient).all()),
                    extra_peak_allocated_mib=(torch.cuda.max_memory_allocated()-base)/1024**2))
                del p,loss,gradient
            ref=index[n,'scene_'+mode]
            row=dict(n=n,mode=mode,trials=trials,median_ms=1000*statistics.median(v['seconds'] for v in trials[1:]),
                reference_loss=ref['cd_m'],max_loss_difference=max(abs(t['loss']-ref['cd_m']) for t in trials),
                pass_test=all(t['gradient_finite'] for t in trials) and max(abs(t['loss']-ref['cd_m']) for t in trials)<=1e-6)
            report['checks'].append(row);atomic_json(a.output,report);print(json.dumps(row),flush=True)
            if not row['pass_test']:raise RuntimeError('Final release smoke failed')
    paths=[HERE.parents[2]/'fpsgen/ops/chamfer_scene.py',HERE.parents[2]/'fpsgen/ops/chamfer_scene_cuda.cu',BUILD/'fpsgen_chamfer_scene_v1.so']
    report.update(status='PASS_FINAL_RELEASE_6_CASES',final_source_and_binary_hashes={str(p):file_sha(p) for p in paths},
        original_sources_unchanged=all(file_sha(Path(n))==sha for n,sha in old['legacy_sources_before'].items()))
    assert report['original_sources_unchanged'];atomic_json(a.output,report)


if __name__=='__main__':main()

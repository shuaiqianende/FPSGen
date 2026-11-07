"""Compare CPU/GPU continuous rejection sampling, including setup separately."""
import json,time,numpy as np,torch
from pathlib import Path
from fpsgen.vggi.field_generation.sample import load_cache
from fpsgen.vggi.field_generation.sample_cuda import CUDAFieldSampler,extension
torch.set_num_threads(2);extension();results=[]
root=Path('fpsgen/vggi/runs/field_sampler_cuda_v1');root.mkdir(exist_ok=True,parents=True)
for scene in ['00_000500','00_001000','01_000500']:
 t=time.perf_counter();cpu,meta,mass=load_cache(f'fpsgen/vggi/runs/amplitude_gram_dataset_v1/cache/train/{scene}.npz');load=time.perf_counter()-t
 t=time.perf_counter();gpu=CUDAFieldSampler(cpu.a,cpu.origin,cpu.spacing);torch.cuda.synchronize();setup=time.perf_counter()-t
 gpu.sample(1000)
 stats=[]
 for seed in range(3):
  xyz,info=gpu.sample(180000,seed=20261107+seed);stats.append(info)
 xyz=xyz.cpu().numpy();np.save(root/f'{scene}.sample.npy',xyz)
 _,ref=cpu.sample(180000,seed=20261107)
 result=dict(scene=scene,cache_load_decode_seconds=load,gpu_setup_seconds=setup,gpu_runs=stats,cpu=ref,expected_acceptance=mass/gpu.envelope_mass,
  sample_mean_xyz=xyz.mean(0).tolist(),points=xyz.shape[0]);results.append(result)
 (root/'benchmark.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)

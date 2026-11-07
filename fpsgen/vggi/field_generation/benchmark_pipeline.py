import time,json,torch,numpy as np
from fpsgen.vggi.field_generation.codec import AmplitudeCodec
from fpsgen.vggi.field_generation.sample_cuda import CUDAFieldSampler,extension
from fpsgen.vggi.point_transport import match
torch.set_num_threads(2);extension()
d=np.load('fpsgen/vggi/runs/amplitude_gram_dataset_v1/cache/train/00_000500.npz');m=json.loads(str(d['metadata']));codec=AmplitudeCodec(m['shape'],m['origin'],m['spacing'],1.)
q=torch.tensor(d['q'],device='cuda');y=torch.tensor(np.load('fpsgen/vggi/runs/point_transport_poisson_gt_v1/00_000500/inputs.npz')['target'],device='cuda')
ref=codec.decode(d['q']);a=codec.decode_cuda(q);error=float(np.max(np.abs(a.cpu().numpy()-ref)))
rows=[]
for rep in range(4):
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();a=codec.decode_cuda(q);torch.cuda.synchronize();t1=time.perf_counter()
 s=CUDAFieldSampler(a,m['origin'],m['spacing']);torch.cuda.synchronize();t2=time.perf_counter();x,si=s.sample();t3=time.perf_counter();z,mi=match(x,y);torch.cuda.synchronize();t4=time.perf_counter()
 rows.append(dict(decode=t1-t,setup=t2-t1,sample=t3-t2,match=t4-t3,total=t4-t,peak_gib=torch.cuda.max_memory_allocated()/2**30))
print(json.dumps(dict(real_decode_max_abs_error=error,runs=rows)),flush=True)
open('fpsgen/vggi/runs/field_sampler_cuda_v1/pipeline.json','w').write(json.dumps(dict(real_decode_max_abs_error=error,runs=rows),indent=2))

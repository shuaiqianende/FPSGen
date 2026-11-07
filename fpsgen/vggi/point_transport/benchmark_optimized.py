"""Interleaved exact-graph speed and quality audit on three real scenes."""
import time,json,numpy as np,torch
from pathlib import Path
from fpsgen.vggi.point_transport.batch_soft import match
from fpsgen.vggi.point_transport.quality import set_metrics,collapse_metrics
root=Path('fpsgen/vggi/runs/point_transport_optimized_v1');root.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(2)
arr=[np.load(f'fpsgen/vggi/runs/point_transport_poisson_gt_v1/{s}/inputs.npz') for s in ['00_000500','00_001000','01_000500']]
results=[]
for b in [1,2,4]:
 x,y=[torch.tensor(np.stack([arr[i%3][k] for i in range(b)]),device='cuda') for k in ['source','target']]
 outputs={};times={k:[] for k in ['sort','linear']};infos={}
 for mode in times:match(x,y,graph_backend=mode,reuse_knn_bounds=mode=='linear')
 for rep in range(5):
  for mode in (['sort','linear'] if rep%2 else ['linear','sort']):
   torch.cuda.synchronize();t=time.perf_counter();z,info=match(x,y,graph_backend=mode,reuse_knn_bounds=mode=='linear');torch.cuda.synchronize()
   times[mode].append(time.perf_counter()-t);outputs[mode]=z.cpu().numpy();infos[mode]=info
 quality=[]
 for i in range(b):
  delta=np.linalg.norm(outputs['linear'][i]-outputs['sort'][i],axis=1);move=np.linalg.norm(outputs['linear'][i]-arr[i%3]['source'],axis=1)
  quality.append(dict(slot=i,delta_rms_m=float(np.sqrt(np.mean(delta**2))),delta_max_m=float(delta.max()),move_mean_m=float(move.mean()),move_rms_m=float(np.sqrt(np.mean(move**2))),set=set_metrics(outputs['linear'][i],arr[i%3]['target']),collapse=collapse_metrics(outputs['linear'][i])))
 result=dict(batch=b,times=times,median={k:float(np.median(v)) for k,v in times.items()},info=infos,quality=quality)
 results.append(result);(root/'benchmark.json').write_text(json.dumps(results,indent=2));print(json.dumps(dict(batch=b,median=result['median'],quality=quality)),flush=True)

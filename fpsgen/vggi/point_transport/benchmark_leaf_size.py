import time,json,torch,numpy as np
from fpsgen.vggi.point_transport.batch_soft import match
x=np.load('fpsgen/vggi/runs/point_transport_poisson_gt_v1/00_000500/inputs.npz');x,y=[torch.tensor(x[k][None],device='cuda') for k in ('source','target')]
torch.set_num_threads(2)
for leaf in [32,64,128,256,512,1024]:
 match(x,y,graph_backend='linear',leaf_size=leaf)
 times=[]
 for _ in range(3):
  torch.cuda.synchronize();t=time.perf_counter();z,info=match(x,y,graph_backend='linear',leaf_size=leaf);torch.cuda.synchronize();times.append(time.perf_counter()-t)
 print(json.dumps(dict(leaf=leaf,median=float(np.median(times)),graph=info['graph_seconds'],solve=info['solve_seconds'])),flush=True)

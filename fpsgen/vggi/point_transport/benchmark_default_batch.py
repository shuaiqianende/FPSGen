"""Separate algorithm improvement from the intrinsic benefit of batching."""
import time,json,numpy as np,torch
from pathlib import Path
from . import match
from .batch_soft import match as previous_batch
from .soft_flow import reference_match

def main():
    torch.set_num_threads(2)
    scenes=['00_000500','00_001000','01_000500','00_000500']
    arrays=[np.load(f'fpsgen/vggi/runs/point_transport_poisson_gt_v1/{s}/inputs.npz') for s in scenes]
    x,y=[torch.tensor(np.stack([a[k] for a in arrays]),device='cuda') for k in ['source','target']]
    config=dict(k=32,epsilons=(.02,.005,.002),penalty=50.,iterations=30,step_size=.9,debias=True)
    def call(mode):
        if mode=='old_serial':return [reference_match(a,b,**config) for a,b in zip(x,y)]
        if mode=='previous_batch':return previous_batch(x,y,**config)
        if mode=='new_serial':return [match(a,b,**config) for a,b in zip(x,y)]
        return match(x,y,**config)
    names=['old_serial','previous_batch','new_serial','new_batch'];times={n:[] for n in names};peaks={}
    for n in names:call(n)
    for repeat in range(5):
        for n in names[repeat%4:]+names[:repeat%4]:
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();out=call(n);torch.cuda.synchronize()
            times[n].append(time.perf_counter()-start);peaks[n]=max(peaks.get(n,0),torch.cuda.max_memory_allocated()/2**30);del out
    result=dict(batch=4,points_per_scene=180000,scene_order=scenes,config=config,times=times,median_seconds={n:float(np.median(t)) for n,t in times.items()},peak_allocated_gib=peaks)
    Path('fpsgen/vggi/runs/point_transport_optimized_v1/default_batch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

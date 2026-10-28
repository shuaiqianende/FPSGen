"""Real LiDAR inference timing; event spans and synchronized wall are distinct."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,statistics,time
import numpy as np,torch,yaml
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_network import OccupancyGridSpec
from fpsgen.hier_occ_flow.triplane_encoder import TriPlaneEncoder,voxel_statistics
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());gpu=c['training']['physical_gpus'][0]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('GPU mismatch')
    admission=gpu_preflight((gpu,),minimum_free_gib=22.,allow_shared_compute=False)
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    if a.output.exists():raise FileExistsError('Preserve benchmark')
    torch.set_num_threads(4);torch.manual_seed(c['training']['seed']);torch.cuda.manual_seed_all(c['training']['seed'])
    s=OccupancyGridSpec(x_min=-50,y_min=-50,z_min=-4,x_max=50,y_max=50,z_max=4.4,nx=384,ny=384,nz=32,xy_radius=50,sphere_radius=50)
    path=Path(c['data_root'])/'08/input_/000100.npy';xyz=torch.from_numpy(input_xyz(path)).cuda()
    model=TriPlaneEncoder(s,c['model']['encoder_variant'][-2:]).cuda().eval();events={};handles=[]
    def hook(name,module):
        def pre(mod,args):
            e=torch.cuda.Event(enable_timing=True);e.record();events.setdefault(name,[]).append([e,None])
        def post(mod,args,out):
            e=torch.cuda.Event(enable_timing=True);e.record();events[name][-1][1]=e
        handles.extend((module.register_forward_pre_hook(pre),module.register_forward_hook(post)))
    for name,module in model.named_modules():
        if name in ('planes.layers.0','planes.layers.1','planes.layers.2','planes.refine.0','planes.refine.1','planes.refine.2','bridge','sparse_conv'):
            hook(name,module)
    rows=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for i in range(15):
            events.clear();torch.cuda.synchronize();start=time.perf_counter();a0=torch.cuda.Event(enable_timing=True);a1=torch.cuda.Event(enable_timing=True)
            a0.record();result=model([xyz]);a1.record();torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
            if any(not bool(torch.isfinite(v).all()) for v in result):raise ValueError('Nonfinite output')
            if i>=5:
                components={k:sum(x.elapsed_time(y) for x,y in pairs) for k,pairs in events.items()}
                rows.append(dict(wall_ms=wall,event_ms=a0.elapsed_time(a1),module_event_ms=components))
    for h in handles:h.remove()
    # Independent voxel-statistics measurement includes any CUDA sync in it.
    stats=[]
    for _ in range(5):
        torch.cuda.synchronize();t=time.perf_counter();e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
        voxel_statistics([xyz],s);e1.record();torch.cuda.synchronize();stats.append(dict(wall_ms=(time.perf_counter()-t)*1000,event_ms=e0.elapsed_time(e1)))
    summary={k:statistics.mean(r[k] for r in rows) for k in ('wall_ms','event_ms')}
    summary['module_event_ms']={k:statistics.mean(r['module_event_ms'].get(k,0.) for r in rows) for k in set().union(*(r['module_event_ms'] for r in rows))}
    summary['statistics_wall_ms']=statistics.mean(v['wall_ms'] for v in stats);summary['statistics_event_ms']=statistics.mean(v['event_ms'] for v in stats)
    atomic_json(a.output,dict(status='COMPLETE',scope='random-initialized encoder, single real frame, AMP, warm5 timed10; not quality or full generation',gpu=admission,input_sha256=file_sha(path),rows=rows,statistics_rows=stats,summary=summary,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

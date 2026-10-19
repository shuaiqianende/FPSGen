"""Small-scale stress demonstration: tree pruning can fail on exact duplicates."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import torch
from export_stage2_snapshot import admit_gpu1
from fpsgen.ops.gpu_kdtree import load_extension,chamfer_kdtree
from fpsgen.ops.chamfer import Chamfer3DDist
from fpsgen.ops.chamfer_scene import SceneChamferDistance,load_scene_extension
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve audit')
    report=dict(status='RUNNING',admission=admit_gpu1(),cases=[],note='Synthetic small stress cases, not representative raw LiDAR quality metrics; synchronized build+query timing; first two runs warmup.')
    atomic_json(a.output,report)
    torch.set_num_threads(4)
    extension=load_extension('/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/gpu_kdtree_v1')
    load_scene_extension()
    scene_tiled=SceneChamferDistance('tiled');scene_bvh=SceneChamferDistance('bvh')
    native=Chamfer3DDist();gen=torch.Generator(device='cuda').manual_seed(294)
    def cuda_cd(p,q):
        dp,dq,_,_=native(p[None],q[None]);return float(.5*(dp.sqrt().double().mean()+dq.sqrt().double().mean()))
    cases=[('tiny_random',torch.randn(64,3,device='cuda',generator=gen),torch.randn(64,3,device='cuda',generator=gen)),
        ('small_random',torch.randn(1024,3,device='cuda',generator=gen),torch.randn(8192,3,device='cuda',generator=gen)),
        ('all_identical_points',torch.zeros(1024,3,device='cuda'),torch.zeros(8192,3,device='cuda')),
        ('identical_support_nonzero_distance',torch.ones(1024,3,device='cuda'),torch.zeros(8192,3,device='cuda'))]
    for name,p,q in cases:
        row=dict(name=name,prediction_points=len(p),target_points=len(q),backends={})
        def scene_cd(module):
            dp,dq,_,_=module(p,q);return float(.5*(dp.sqrt().double().mean()+dq.sqrt().double().mean()))
        for backend,fn in [('scene_tiled',lambda:scene_cd(scene_tiled)),('scene_bvh',lambda:scene_cd(scene_bvh)),('chamfer_cuda',lambda:cuda_cd(p,q)),('gpu_kdtree',lambda:chamfer_kdtree(p,q,extension=extension)['cd_m'])]:
            runs=[]
            for repeat in range(7):
                torch.cuda.synchronize();tick=time.perf_counter();value=fn();torch.cuda.synchronize()
                runs.append(dict(repeat=repeat,warmup=repeat<2,seconds=time.perf_counter()-tick,cd_m=value))
            row['backends'][backend]=dict(runs=runs,median_seconds=statistics.median(v['seconds'] for v in runs[2:]))
        row['tree_over_cuda_time_ratio']=row['backends']['gpu_kdtree']['median_seconds']/row['backends']['chamfer_cuda']['median_seconds']
        report['cases'].append(row);atomic_json(a.output,report);print(json.dumps(row),flush=True)
    report.update(status='COMPLETE',source_sha256=file_sha(Path(__file__)))
    atomic_json(a.output,report)


if __name__=='__main__':main()

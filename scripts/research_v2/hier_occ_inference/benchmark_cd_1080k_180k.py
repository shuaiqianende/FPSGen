"""Exactly1080000 target vs180000 prediction points: CD timing and full parity."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from export_stage2_snapshot import admit_gpu1
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.ops.gpu_kdtree import load_extension,ExactGPUKDTree,chamfer_kdtree
from fpsgen.ops.chamfer import Chamfer3DDist
from fpsgen.utils.gpu_kdtree_training import chamfer_sqrt_gpu_kdtree
from fpsgen.voxel_point.metrics import chamfer_sqrt


def timed(fn):
    torch.cuda.synchronize();tick=time.perf_counter();v=fn();torch.cuda.synchronize()
    return v,time.perf_counter()-tick


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audit')
    report=dict(status='PREPARING',admission=admit_gpu1(),seed=20261019,backends={},per_point_checks=[],
        protocol='SIZE BENCHMARK ONLY: fixed seeded subsets without replacement from real08/003900 prediction and raw map; does not replace full-point evaluation',
        timing='SingleGPU FP32 CD training forward and prediction-only gradient backward, no model; warmup excluded; tree build included',
        thresholds=dict(cd_absolute_m=1e-6,each_distance_m=1e-5,gradient_relative_l2=1e-4,gradient_max_absolute=1e-6))
    atomic_json(a.output,report);torch.set_num_threads(4)
    root=Path('/data-12/M2024-HWZ/FPSGen')
    source=root/'experiments/hs_stage2_ply/morph_step017000_seq08_003900_s1nfe4_s2nfe4/predicted/online/nfe4/fine_surface_hit_centers.ply'
    vertex=PlyData.read(str(source))['vertex'];full_pred=np.ascontiguousarray(np.column_stack([vertex[k] for k in ('x','y','z')]),dtype='f4')
    geometry=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    raw=RawMapSource('/data-12/M2024-HWZ/KITTI_Odometry',fast_viewpoint=True)
    full_gt=raw.load('08',3900);_,valid=geometry.voxelize_points(full_gt);full_gt=np.ascontiguousarray(full_gt[valid],dtype='f4')
    rng=np.random.default_rng(report['seed'])
    pi=np.sort(rng.choice(len(full_pred),180000,replace=False));gi=np.sort(rng.choice(len(full_gt),1080000,replace=False))
    pred,gt=full_pred[pi],full_gt[gi]
    report.update(prediction_source_sha256=file_sha(source),original_prediction_points=len(full_pred),original_gt_points=len(full_gt),
        predicted_points=len(pred),target_points=len(gt),pair_count=len(pred)*len(gt))
    atomic_json(a.output,report)
    p0,q=torch.from_numpy(pred).cuda(),torch.from_numpy(gt).cuda()
    extension=load_extension(root/'env/voxel_point_cache/gpu_kdtree_v1');native=Chamfer3DDist()
    cpu_refs=[];cpu_indices=[];cpu_times=[]
    for queries,support in [(pred,gt),(gt,pred)]:
        tick=time.perf_counter();d,idx=cKDTree(support,leafsize=32).query(queries,eps=0.,workers=4)
        cpu_times.append(time.perf_counter()-tick);cpu_refs.append(d);cpu_indices.append(idx)
    cpu_cd=.5*sum(float(d.mean()) for d in cpu_refs)
    # Oracle loss/gradient uses CPU exact indices, identical FP32 differentiable norm.
    p=p0.clone().requires_grad_();ip,iq=[torch.from_numpy(v).cuda() for v in cpu_indices]
    loss=.5*((p-q[ip]).norm(dim=-1).mean()+(q-p[iq]).norm(dim=-1).mean())
    oracle_gradient=torch.autograd.grad(loss,p)[0];oracle_loss=float(loss)
    report.update(cpu_cd_m=cpu_cd,cpu_build_query_seconds=sum(cpu_times),oracle_fp32_loss=oracle_loss)
    native_d=[]
    with torch.no_grad():
        dp,dq,jp,jq=native(p0[None],q[None]);torch.cuda.synchronize()
        native_d=[dp[0].sqrt().cpu().numpy(),dq[0].sqrt().cpu().numpy()]
        for i,(queries,support) in enumerate([(p0,q),(q,p0)]):
            tree=ExactGPUKDTree(support,extension=extension);d2,idx=tree.query(queries)
            distances=d2.sqrt().cpu().numpy();nn=idx.cpu().numpy()
            actual=np.abs(distances-cpu_refs[i]);legacy=np.abs(native_d[i]-cpu_refs[i])
            row=dict(direction=['prediction_to_gt','gt_to_prediction'][i],queries=len(queries),
                gpu_tree_max_distance_error_m=float(actual.max()),cuda_fp32_max_distance_error_m=float(legacy.max()),
                gpu_tree_nearest_index_mismatches=int(np.sum(nn!=cpu_indices[i])),
                pass_test=bool(actual.max()<=1e-9 and legacy.max()<=1e-5))
            report['per_point_checks'].append(row);print('Every query',row,flush=True)
            if not row['pass_test']:raise ValueError('Per-query distance tolerance failed')
        del dp,dq,jp,jq,tree,d2,idx
    def native_loss(p,g):
        d,e,_,_=native(p[None],g[None]);return .5*(d.sqrt().mean()+e.sqrt().mean())
    functions=dict(keops=lambda p,g:chamfer_sqrt(p,g,backend='keops'),chamfer_cuda=native_loss,
        gpu_kdtree=lambda p,g:chamfer_sqrt_gpu_kdtree(p,g,extension=extension))
    torch.cuda.reset_peak_memory_stats()
    for name,fn in functions.items():
        rows=[]
        for repeat in range(4):
            p=p0.clone().requires_grad_()
            loss,forward=timed(lambda:fn(p,q))
            gradient,backward=timed(lambda:torch.autograd.grad(loss,p)[0])
            finite=bool(torch.isfinite(gradient).all())
            row=dict(repeat=repeat,warmup=repeat==0,loss=float(loss),forward_seconds=forward,backward_seconds=backward,total_seconds=forward+backward,gradient_finite=finite)
            rows.append(row);print(name,row,flush=True)
        loss_error=abs(float(loss)-oracle_loss)
        relative=float((gradient-oracle_gradient).norm()/oracle_gradient.norm().clamp_min(1e-30))
        absolute=float((gradient-oracle_gradient).abs().max())
        report['backends'][name]=dict(runs=rows,median_forward_seconds=statistics.median(r['forward_seconds'] for r in rows[1:]),
            median_backward_seconds=statistics.median(r['backward_seconds'] for r in rows[1:]),median_total_seconds=statistics.median(r['total_seconds'] for r in rows[1:]),
            oracle_loss_absolute_error=loss_error,gradient_relative_l2=relative,gradient_max_absolute=absolute,
            pass_test=finite and loss_error<=1e-6 and relative<=1e-4 and absolute<=1e-6)
        atomic_json(a.output,report)
    with torch.no_grad():
        evaluation,seconds=timed(lambda:chamfer_kdtree(p0,q,extension=extension))
    report.update(status='PASS' if all(v['pass_test'] for v in report['backends'].values()) else 'GRADIENT_REVIEW_REQUIRED',
        gpu_tree_fp64_evaluation=evaluation,gpu_tree_fp64_evaluation_seconds=seconds,
        cpu_vs_gpu_evaluation_error_m=abs(cpu_cd-evaluation['cd_m']),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        code_sha256=file_sha(Path(__file__)))
    atomic_json(a.output,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

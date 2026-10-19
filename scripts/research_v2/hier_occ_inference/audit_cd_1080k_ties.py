"""Explain exact NN ties without weakening the original gradient gate."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
import torch
from export_stage2_snapshot import admit_gpu1
from fpsgen.ops.gpu_kdtree import load_extension,ExactGPUKDTree
from fpsgen.ops.endpoint_refinement.knn import knn
from fpsgen.ops.chamfer import Chamfer3DDist
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve previous tie audit')
    torch.set_num_threads(4)
    r=dict(status='RUNNING',admission=admit_gpu1(),note='Original gradient gate remains GRADIENT_REVIEW_REQUIRED; this audit identifies valid tie subgradients explicitly.')
    atomic_json(a.output,r)
    root=Path('/data-12/M2024-HWZ/FPSGen')
    source=root/'experiments/hs_stage2_ply/morph_step017000_seq08_003900_s1nfe4_s2nfe4/predicted/online/nfe4/fine_surface_hit_centers.ply'
    v=PlyData.read(str(source))['vertex'];fullp=np.ascontiguousarray(np.column_stack([v[k] for k in ('x','y','z')]),dtype='f4')
    g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    raw=RawMapSource('/data-12/M2024-HWZ/KITTI_Odometry',fast_viewpoint=True)
    fullq=raw.load('08',3900);_,valid=g.voxelize_points(fullq);fullq=np.ascontiguousarray(fullq[valid],dtype='f4')
    rng=np.random.default_rng(20261019)
    pi=np.sort(rng.choice(len(fullp),180000,replace=False));qi=np.sort(rng.choice(len(fullq),1080000,replace=False))
    pc,qc=fullp[pi],fullq[qi];p0,q=torch.from_numpy(pc).cuda(),torch.from_numpy(qc).cuda()
    extension=load_extension(root/'env/voxel_point_cache/gpu_kdtree_v1')
    indices={name:[] for name in ['cpu','gpu_kdtree','keops','chamfer_cuda']}
    details=[]
    with torch.no_grad():
        for direction,queries,support,queries_cpu,support_cpu in [('p_to_g',p0,q,pc,qc),('g_to_p',q,p0,qc,pc)]:
            ref,refidx=cKDTree(support_cpu,leafsize=32).query(queries_cpu,eps=0.,workers=4)
            indices['cpu'].append(torch.from_numpy(refidx).cuda())
            tree=ExactGPUKDTree(support,extension=extension);_,idx=tree.query(queries);indices['gpu_kdtree'].append(idx)
            _,idx=knn(queries,support,1,'keops');indices['keops'].append(idx[:,0].long())
            actual=indices['gpu_kdtree'][-1].cpu().numpy();bad=np.flatnonzero(actual!=refidx)
            ties=[]
            for j in bad:
                query=queries_cpu[j].astype('f8');c=support_cpu[refidx[j]].astype('f8');k=support_cpu[actual[j]].astype('f8')
                dc=float(np.sum((query-c)**2));dk=float(np.sum((query-k)**2))
                ties.append(dict(query_index=int(j),cpu_index=int(refidx[j]),gpu_index=int(actual[j]),query_xyz=query.tolist(),
                    cpu_neighbor_xyz=c.tolist(),gpu_neighbor_xyz=k.tolist(),cpu_squared_distance=dc,gpu_squared_distance=dk,
                    exact_equal_squared_distance=dc==dk))
            details.append(dict(direction=direction,queries=len(queries),cpu_vs_tree_index_differences=len(bad),
                keops_vs_tree_index_differences=int((indices['keops'][-1]!=indices['gpu_kdtree'][-1]).sum()),ties=ties))
        native=Chamfer3DDist();_,_,ip,iq=native(p0[None],q[None]);indices['chamfer_cuda']=[ip[0].long(),iq[0].long()]
    gradients={};losses={}
    for name,(pg,gp) in indices.items():
        p=p0.clone().requires_grad_();loss=.5*((p-q[pg]).norm(dim=-1).mean()+(q-p[gp]).norm(dim=-1).mean())
        gradients[name]=torch.autograd.grad(loss,p)[0];losses[name]=float(loss)
    compare={}
    for name in ['cpu','chamfer_cuda','gpu_kdtree']:
        d=gradients[name]-gradients['keops']
        compare[name]=dict(max_absolute=float(d.abs().max()),relative_l2=float(d.norm()/gradients['keops'].norm()),
            changed_prediction_points=int((d!=0).any(-1).sum()))
    r.update(status='PASS_EXACT_DISTANCE_WITH_TIE_SUBGRADIENT' if all(t['exact_equal_squared_distance'] for row in details for t in row['ties']) and compare['gpu_kdtree']['max_absolute']==0 else 'REVIEW_REQUIRED',
        per_direction=details,losses=losses,coordinate_gradients_vs_keops=compare,source_sha256=file_sha(Path(__file__)),
        conclusion='CPU and GPU may select different exact equidistant neighbors. CD values agree; tie-point subgradient need not be unique. Existing KeOps and GPU KD-tree gradients compared explicitly.')
    atomic_json(a.output,r);print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()

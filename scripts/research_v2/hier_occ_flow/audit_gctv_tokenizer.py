"""Compare original f64 voxel IDs with actual TriPlane f32 arithmetic."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json
import numpy as np,torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage1_network import OccupancyGridSpec,occupancy_lidar_features
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.dataset import input_xyz

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=64);a=p.parse_args()
    root=Path('/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/occ384_full_parallel/development')
    inputs=Path('/data-12/M2024-HWZ/KITTI_Odometry');g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2),origin=(-50.,-50.,-4.),maximum=(50.,50.,4.4),sphere_radius=50.)
    s=OccupancyGridSpec(x_min=-50,y_min=-50,z_min=-4,x_max=50,y_max=50,z_max=4.4,nx=384,ny=384,nz=32,xy_radius=50,sphere_radius=50);ds=SparseShardDataset(root,g,verify=False);torch.set_num_threads(4)
    rows=[]
    for i in np.linspace(0,len(ds)-1,a.frames,dtype=int):
        r=ds[int(i)];p=torch.from_numpy(input_xyz(inputs/r['sequence']/'input_'/f"{r['frame']:06d}.npy"));valid=p[s.valid_points(p)];_,inds=occupancy_lidar_features(p,s);reference=inds[0][0].long()
        q=torch.floor((valid.float()-valid.new_tensor(s.origin))/(valid.new_tensor(s.size)*2)).long();different=(q!=reference).any(1)
        unique,counts=torch.unique(reference,dim=0,return_counts=True)
        rows.append(dict(sequence=r['sequence'],frame=r['frame'],input_points=len(p),valid_points=len(valid),occupied_cells=len(unique),duplicate_points=int((counts-1).sum()),fp32_vs_original_f64_index_mismatches=int(different.sum())))
    atomic_json(a.output,dict(status='COMPLETE_DIAGNOSTIC',frames=len(rows),development_manifest_sha256=file_sha(root/'manifest.json'),rows=rows,total_index_mismatches=sum(r['fp32_vs_original_f64_index_mismatches'] for r in rows),actual_12d=['log_count','mean_local_xyz(3)','variance_local_xyz(3)','local_z_min','local_z_max','mean_normalized_range','variance_normalized_range','observed_mask'],original='6D pointwise local xyz, absolute XY, normalized distance -> trainable point MLP and voxel max aggregation',scope='Voxel ID/observed count diagnostic, not full D0 or evidence of architecture causality'))
    print('frames',len(rows),'index_mismatches',sum(r['fp32_vs_original_f64_index_mismatches'] for r in rows))

if __name__=='__main__':main()

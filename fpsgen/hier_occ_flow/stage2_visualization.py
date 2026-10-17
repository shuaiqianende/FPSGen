"""Sparse PLY diagnostics; evaluation GT stays outside deployment interfaces."""
from pathlib import Path
import numpy as np
from .storage import atomic_json
from .voxelization import pack_children,occupied_fine_coords


def write_xyz_ply(path,xyz,color=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    xyz=np.asarray(xyz,dtype='<f4')
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('Finite XYZ PLY required')
    fields=[('x','<f4'),('y','<f4'),('z','<f4')]
    if color is not None:fields += [('red','u1'),('green','u1'),('blue','u1')]
    points=np.empty(len(xyz),dtype=fields)
    for axis,name in enumerate(('x','y','z')):points[name]=xyz[:,axis]
    if color is not None:
        for value,name in zip(color,('red','green','blue')):points[name]=value
    properties='property float x\nproperty float y\nproperty float z\n'
    if color is not None:properties+='property uchar red\nproperty uchar green\nproperty uchar blue\n'
    with path.open('xb') as handle:
        handle.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n{properties}end_header\n'.encode())
        handle.write(points.tobytes())


def export_evaluation_diagnostics(directory,partial_xyz,candidate_ids,predicted,label_ids,label_masks,g):
    """All GT fine voxels, including missed parents, contribute to the FN PLY."""
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    predicted=np.asarray(predicted,bool)&g.valid_children(candidate_ids)
    coords=occupied_fine_coords(candidate_ids,pack_children(predicted,g),g)
    target=occupied_fine_coords(label_ids,label_masks,g)
    pids=np.unique(g.flatten(coords,fine=True));tids=np.unique(g.flatten(target,fine=True))
    fps=np.setdiff1d(pids,tids);fns=np.setdiff1d(tids,pids)
    write_xyz_ply(root/'partial_lidar.ply',partial_xyz,(160,160,160))
    write_xyz_ply(root/'candidate_parent_centers.ply',g.centers(g.unflatten(candidate_ids)),(230,180,20))
    write_xyz_ply(root/'gt_fine_occupancy.ply',g.centers(target,fine=True),(30,200,60))
    write_xyz_ply(root/'stage2_fine_occupancy.ply',g.centers(coords,fine=True),(30,100,230))
    write_xyz_ply(root/'false_positive.ply',g.centers(g.unflatten(fps,fine=True),fine=True),(230,40,40))
    write_xyz_ply(root/'false_negative.ply',g.centers(g.unflatten(fns,fine=True),fine=True),(200,40,210))
    atomic_json(root/'manifest.json',{'geometry_sha256':g.sha256,'scope':'evaluation only; includes GT',
        'predicted_fine_voxels':len(pids),'gt_fine_voxels':len(tids),'false_positives':len(fps),'false_negatives':len(fns),
        'all_missed_parent_gt_included':True,'points_are_quantized_occupancy_centers':True})

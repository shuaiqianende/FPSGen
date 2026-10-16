import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,unpack_children,pack_children,lookup_masks


@pytest.mark.parametrize('split',[(2,2,2),(3,3,2)])
def test_roundtrip_and_packed(split):
    g=SparseHierarchyGeometry(child_split=split)
    rng=np.random.default_rng(7); fine=rng.integers(0,g.fine_resolution,size=(10000,3))
    parent,child=g.split_fine(fine)
    assert np.array_equal(g.compose(parent,child),fine)
    assert np.array_equal(g.unflatten(g.flatten(parent)),parent)
    assert np.array_equal(g.unflatten(g.flatten(fine,fine=True),fine=True),fine)
    occ=rng.random((100,g.k))>.5
    assert np.array_equal(unpack_children(pack_children(occ,g),g),occ)
    xyz=g.centers(fine,fine=True); coords,valid=g.voxelize_points(xyz)
    assert np.array_equal(coords,fine[valid])
    assert np.array_equal(g.voxelize_points(torch.tensor(xyz))[0].numpy(),coords)


def test_boundary_sphere_intersection_and_points():
    g=SparseHierarchyGeometry(); xyz=np.array([[0,0,-4],[0,0,4.4],[50,0,0],[49.999,0,0],[0,0,0],[np.nan,0,0]])
    fine,valid=g.voxelize_points(xyz)
    assert valid.tolist()==[False,False,False,True,True,False]
    parent,child=g.split_fine(fine); ids=g.flatten(parent)
    assert g.valid_children(ids)[np.arange(len(ids)),child].all()
    ids,masks,stats=voxelize_surface(xyz,g)
    assert stats['valid_points']==2 and (unpack_children(masks,g).any(1)).all()
    assert np.array_equal(lookup_masks(ids,masks,ids),masks)
    assert (lookup_masks(ids,masks,np.array([0]))==0).all()


def test_dense_fine_not_needed_and_float32_edges():
    g=SparseHierarchyGeometry(parent_resolution=(352,352,32),child_split=(2,2,2))
    rng=np.random.default_rng(21); xyz=rng.uniform([-49,-49,-3.9],[49,49,4.3],size=(1000,3)).astype(np.float32)
    a,valid=g.voxelize_points(xyz); b,valid2=g.voxelize_points(torch.tensor(xyz))
    assert np.array_equal(a,b.numpy()) and np.array_equal(valid,valid2.numpy())
    with pytest.raises(ValueError): SparseHierarchyGeometry(child_split=(4,4,4))


def test_cell_with_center_outside_sphere_retains_inside_hits():
    g=SparseHierarchyGeometry(); parent=np.array([[255,140,15]],np.int64)
    assert np.linalg.norm(g.centers(parent)[0])>50
    point=np.asarray(g.origin)+parent[0]*g.parent_size+np.array([.01,.01,.01])
    fine,valid=g.voxelize_points(point[None]); assert valid.all()
    p,c=g.split_fine(fine); ids=g.flatten(p)
    assert g.valid_children(ids)[0,c[0]]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU not available/allocated for audit')
def test_cuda_cpu_voxelization_exact():
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('2','3','2,3')
    g=SparseHierarchyGeometry(); p=torch.randn(10000,3)*10
    a,m=g.voxelize_points(p); b,n=g.voxelize_points(p.cuda())
    assert torch.equal(a,b.cpu()) and torch.equal(m,n.cpu())

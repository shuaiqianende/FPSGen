import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.condition import sample_pyramid,observed_children
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.inference import sample,export_occupancy
from fpsgen.hier_occ_flow.candidate_selection import select_candidates,oracle_candidates
from fpsgen.hier_occ_flow.evaluation import fine_metrics
from fpsgen.hier_occ_flow.voxelization import voxelize_surface,unpack_children
from fpsgen.hier_occ_flow.dataset import input_xyz


def test_physical_xyz_to_zyx_sampling():
    shape=(8,12,4); lo=(-50.,-50.,-4.); hi=(50.,50.,4.4)
    axes=[torch.linspace(a+(b-a)/n/2,b-(b-a)/n/2,n) for a,b,n in zip(lo,hi,shape)]
    z,y,x=torch.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
    field=torch.stack((x,y,z))[None]
    centers=torch.tensor([[axes[0][2],axes[1][3],axes[2][1]],[0.,0.,.2]])
    result=sample_pyramid([field],centers,lo,hi)
    assert torch.allclose(result,centers,atol=2e-5)


@pytest.mark.parametrize('nfe',[1,2,4,8])
def test_constant_velocity_single_and_multistep(nfe):
    class Constant(torch.nn.Module):
        objective='fm'
        def forward(self,ids,*,state,time,**condition): return torch.ones_like(state)*.75
    model=Constant().eval(); ids=torch.arange(5)
    c={'valid_child_mask':torch.ones(5,8,dtype=torch.bool)}
    expected=torch.randn((5,8),generator=torch.Generator().manual_seed(7))+.75
    endpoint,occ,calls=sample(model,ids,c,nfe=nfe,generator=torch.Generator().manual_seed(7))
    assert torch.allclose(endpoint,expected,atol=1e-6) and calls==nfe


def test_missing_parent_counted_as_fn_and_oracle_scores_constant(tmp_path):
    g=SparseHierarchyGeometry(); xyz=np.array([[0,0,0],[4,4,0]],np.float32)
    ids,masks,_=voxelize_surface(xyz,g)
    one=ids[:1]; pred=unpack_children(masks[:1],g)
    metrics=fine_metrics(one,pred,ids,masks,g)
    assert metrics['fine']['fn']==1 and metrics['candidate']['fine_occupancy_recall_ceiling']==.5
    selected,scores,meta=oracle_candidates(ids,g,np.random.default_rng(7),negative_fraction=1.)
    assert (scores==1).all() and meta['mode']=='oracle_not_deployable'
    export_occupancy(tmp_path/'prediction',one,pred,g)
    assert (tmp_path/'prediction/fine_surface_hit_centers.ply').exists()
    full_observed=fine_metrics(one,pred,ids,masks,g,observed_parent_ids=ids,observed_masks=masks)
    assert full_observed['observed_hit']['fn']==1
    assert full_observed['not_directly_hit']['fn']==0


def test_input_fourth_column_never_used_and_observed_hits(tmp_path):
    xyz=np.array([[0,0,0],[1,1,1]],np.float32); path=tmp_path/'input.npy'
    np.save(path,np.column_stack((xyz,[999,111])))
    a=input_xyz(path); np.save(path,np.column_stack((xyz,[-1,222]))); b=input_xyz(path)
    assert np.array_equal(a,b)
    g=SparseHierarchyGeometry(); ids,masks,_=voxelize_surface(a,g)
    assert np.array_equal(observed_children(a,ids,g),unpack_children(masks,g))
    with pytest.raises(ValueError): select_candidates(np.ones((1,1,1)),g)


def test_boundary_contact_edges_do_not_reward_all_empty():
    g=SparseHierarchyGeometry(parent_resolution=(8,8,8),child_split=(2,2,2))
    coords=np.array([[7,8,8],[8,8,8]])
    xyz=g.centers(coords,fine=True); ids,masks,_=voxelize_surface(xyz,g)
    perfect=fine_metrics(ids,unpack_children(masks,g),ids,masks,g)
    empty=fine_metrics(ids,np.zeros((len(ids),g.k),bool),ids,masks,g)
    assert perfect['parent_boundary_contact_edges']['f1']==1
    assert empty['parent_boundary_contact_edges']['f1']==0

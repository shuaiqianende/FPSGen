import torch
from fpsgen.hier_occ_flow.gctv_encoder import Coupler,Compact3DEncoder,project,lift
from fpsgen.hier_occ_flow.triplane_encoder import voxel_statistics
class TinySpec:
    shape=(8,8,8);origin=(-4.,-4.,-4.);maximum=(4.,4.,4.);size=(1.,1.,1.);sphere_radius=4.
    def valid_points(self,p):return (p>p.new_tensor(self.origin)).all(-1)&(p<p.new_tensor(self.maximum)).all(-1)&(p.square().sum(-1)<16)


def test_non_square_xyz_projections_and_lift():
    x=torch.arange(2*3*5.).reshape(1,1,2,3,5);mask=torch.ones_like(x,dtype=torch.bool)
    xy,_,_=project(x,mask,2);xz,_,_=project(x,mask,3);yz,_,_=project(x,mask,4)
    assert xy.shape==(1,1,3,5) and xz.shape==(1,1,2,5) and yz.shape==(1,1,2,3)
    torch.testing.assert_close(lift((xy,xz,yz)),xy.unsqueeze(2)+xz.unsqueeze(3)+yz.unsqueeze(4))


def test_empty_and_single_voxel_inputs():
    x,m=voxel_statistics(torch.empty(0,3),TinySpec());assert not m.any() and torch.isfinite(x).all() and not x.any()
    x,m=voxel_statistics(torch.tensor([[.2,.1,.3]]),TinySpec());assert m.sum()==1 and torch.isfinite(x).all()


def test_f64_index_matches_original_boundary_example():
    from fpsgen.hier_occ_flow.stage1_network import OccupancyGridSpec,occupancy_lidar_features
    s=OccupancyGridSpec(x_min=-50,y_min=-50,z_min=-4,x_max=50,y_max=50,z_max=4.4,nx=384,ny=384,nz=32,xy_radius=50,sphere_radius=50)
    points=torch.tensor([[3.6270639896392822,2.604163408279419,-1.7075369358062744],[0.,0.,0.]])
    _,indices=occupancy_lidar_features(points,s);q=indices[0][0]
    _,mask=voxel_statistics(points,s,index_precision='float64')
    for x,y,z in q:assert mask[0,0,z,y,x]
    assert int(q[0,1])==100 and not mask[0,0,4,101,102]


def test_feedback_affects_output_and_gradients():
    torch.manual_seed(17);one=Coupler(8,False);two=Coupler(8,True);two.load_state_dict(one.state_dict(),strict=False)
    x=torch.randn(1,8,4,6,8,requires_grad=True);mask=torch.rand(1,1,4,6,8)>.5
    torch.testing.assert_close(one(x,mask),two(x,mask,enable_feedback=False),rtol=0,atol=0)
    with torch.no_grad():
        one.update.weight.fill_(.01);two.update.weight.copy_(one.update.weight);two.update2.weight.fill_(.01)
    on=two(x,mask);off=two(x,mask,enable_feedback=False)
    assert not torch.allclose(on,off)
    on.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in two.refine.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in two.planes.parameters())


def test_compact_all_variants_small_volume_backward():
    for variant in ('g0','g1','g2'):
        model=Compact3DEncoder(None,variant);x=torch.randn(1,12,4,8,12,requires_grad=True);mask=torch.rand(1,1,4,8,12)>.5
        outs=model.forward_statistics(x,mask);assert [o.shape for o in outs]==[(1,128,4,8,12),(1,128,2,4,6),(1,128,1,2,3)]
        sum(o.square().mean() for o in outs).backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_matched_random_seed_shared_initialization():
    torch.manual_seed(20261032);one=Compact3DEncoder(None,'g1')
    torch.manual_seed(20261032);two=Compact3DEncoder(None,'g2')
    a,b=one.state_dict(),two.state_dict()
    assert all(torch.equal(value,b[key]) for key,value in a.items())

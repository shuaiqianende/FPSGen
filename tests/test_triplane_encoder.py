import torch
import pytest
from fpsgen.hier_occ_flow.triplane_encoder import voxel_statistics,masked_projection,PlanePyramid


class TinySpec:
    shape=(8,8,8);origin=(-4.,-4.,-4.);maximum=(4.,4.,4.);size=(1.,1.,1.);sphere_radius=4.
    def valid_points(self,p):
        return (p>p.new_tensor(self.origin)).all(-1)&(p<p.new_tensor(self.maximum)).all(-1)&(p.square().sum(-1)<16)


def test_duplicate_points_empty_cells_and_axis_order():
    p=torch.tensor([[.2,.1,.3],[.2,.1,.3],[-2.1,.1,.3],[4.,0.,0.]])
    x,m=voxel_statistics(p,TinySpec());assert x.shape==(1,12,4,4,4)
    assert m.sum()==2 and x[0,0,2,2,2].item()==pytest.approx(torch.log(torch.tensor(3.)).item())
    assert m[0,0,2,2,0] and not m[0,0,0,0,0]
    assert torch.equal(x*(~m),torch.zeros_like(x))
    y,n=voxel_statistics(p.flip(0),TinySpec());torch.testing.assert_close(x,y);assert torch.equal(m,n)


def test_projection_ignores_unobserved_and_is_differentiable():
    x=torch.tensor([[[[[2.,7.]],[[4.,9.]]]]],requires_grad=True)
    mask=torch.tensor([[[[[True,False]],[[True,False]]]]])
    y=masked_projection(x,mask,2);torch.testing.assert_close(y,torch.tensor([[[[3.,0.]]]]))
    y.sum().backward();assert x.grad[0,0,0,0,0]==.5 and x.grad[0,0,0,0,1]==0


def test_plane_pyramid_backward():
    net=PlanePyramid();x=torch.randn(2,12,16,16,requires_grad=True)
    out=net.finish(net.features(x));assert [tuple(v.shape) for v in out]==[(2,64,16,16),(2,64,8,8),(2,64,4,4)]
    sum(v.square().mean() for v in out).backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

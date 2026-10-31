import torch
from fpsgen.hier_occ_flow.depth_moment_encoder import MomentResidual,DepthMomentEncoder
from fpsgen.hier_occ_flow.patch_unet_condition import PatchLidarCondition

def test_identity_gradient_and_depth():
    m=MomentResidual(4);x=torch.randn(2,4,4,6,8,requires_grad=True)
    y=m(x);assert torch.count_nonzero(y)==0
    (y*x).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(p.grad.abs().sum() for p in m.parameters())>0
    with torch.no_grad():
        m.mix[0].bias[:4]=1
    y=m(x);assert not torch.equal(y[:,:,0],y[:,:,-1])

def test_initialization():
    torch.manual_seed(20261032);a=PatchLidarCondition(None,'u1a').state_dict()
    torch.manual_seed(20261032);b=DepthMomentEncoder(None).state_dict()
    assert all(torch.equal(v,b[k]) for k,v in a.items())

import torch
from fpsgen.hier_occ_flow.patch_unet_condition import PatchLidarCondition, SpatialAttention

class TinySpec:
    def observed_mask(self,x):
        return x.reshape(1,1,16,16,16)

def test_shapes_and_shared_initialization():
    torch.set_num_threads(2)
    torch.manual_seed(123);p0=PatchLidarCondition(TinySpec(),'p0',False)
    torch.manual_seed(123);u1=PatchLidarCondition(TinySpec(),'u1',False)
    for name,v in p0.state_dict().items():assert torch.equal(v,u1.state_dict()[name]),name
    points=torch.rand(16**3)
    for m in (p0,u1):
        values=m(points)
        assert [tuple(v.shape) for v in values]==[(1,128,8,8,8),(1,128,4,4,4),(1,128,2,2,2)]
        sum(v.square().mean() for v in values).backward()
        assert torch.isfinite(m.input.weight.grad).all() and m.input.weight.grad.abs().sum()>0
    assert all(b.first[-1].weight.grad is not None for b in u1.up_blocks)

def test_attention_matches_manual_float32():
    torch.manual_seed(7);m=SpatialAttention(32).eval();torch.nn.init.normal_(m.output.weight,std=.01)
    x=torch.randn(1,32,2,2,2);y=m(x)
    z=torch.nn.functional.silu(m.outer(x)).reshape(1,32,-1)
    q,k,v=m.qkv(m.inner(z)).chunk(3,dim=1)
    w=torch.softmax(torch.einsum('bct,bcs->bts',q,k)/(32**.5),dim=-1)
    expected=(z+m.output(torch.einsum('bts,bcs->bct',w,v))).reshape_as(x)
    torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)

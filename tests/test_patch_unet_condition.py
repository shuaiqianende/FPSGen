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

def test_a_four_sites_and_zero_residual_identity():
    torch.set_num_threads(2);torch.manual_seed(123)
    base=PatchLidarCondition(TinySpec(),'u1',False).eval()
    torch.manual_seed(123);model=PatchLidarCondition(TinySpec(),'u1a',False).eval()
    for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name]),name
    calls=[];hooks=[]
    for name,m in list(model.plane_down.named_children())+list(model.plane_up.named_children()):
        hooks.append(m.register_forward_hook(lambda module,args,out:calls.append(tuple(out[0].shape))))
    x=torch.rand(16**3);before=base(x);after=model(x)
    assert len(calls)==4
    for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
    sum(y.square().mean() for y in after).backward()
    for m in list(model.plane_down)+list(model.plane_up):
        assert m.update.weight.grad is not None and torch.isfinite(m.update.weight.grad).all() and m.update.weight.grad.abs().sum()>0
    for h in hooks:h.remove()

def test_bc_paths_and_gradients():
    from fpsgen.hier_occ_flow.patch_unet_condition import PlaneMemory,DirectionalContext
    torch.set_num_threads(2);x=torch.randn(1,32,2,4,4,requires_grad=True)
    planes=(torch.randn(1,32,4,4,requires_grad=True),torch.randn(1,32,2,4,requires_grad=True),torch.randn(1,32,2,4,requires_grad=True))
    m=PlaneMemory(32);c=tuple(torch.ones(1,1,*p.shape[2:]) for p in planes)
    fused=m(planes,tuple(p*2 for p in planes),c);sum(y.square().mean() for y in fused).backward(retain_graph=True)
    assert all(g.weight.grad is not None and g.weight.grad.abs().sum()>0 for g in m.gates)
    for flag in (True,False):
        b=DirectionalContext(32,flag);y=b(x,planes);y.square().mean().backward(retain_graph=True);assert b.output.weight.grad.abs().sum()>0
    for variant in ('u1ab','u1ac','u1abc','u1abctl','u1acctl'):
        model=PatchLidarCondition(TinySpec(),variant,False);y=model(torch.rand(16**3));assert len(y)==3 and all(torch.isfinite(t).all() for t in y)

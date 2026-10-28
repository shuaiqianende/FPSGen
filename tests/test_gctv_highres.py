import torch
from fpsgen.hier_occ_flow.gctv_highres_encoder import AdaptiveProjection,HighResGCTVEncoder

def test_adaptive_projection_empty_and_zero_logits_equal_mean():
    torch.manual_seed(2);x=torch.randn(1,8,4,6,8,requires_grad=True);mask=torch.rand(1,1,4,6,8)>.5;mask[:,:,:,:,0]=False
    p=AdaptiveProjection(8);observed,context,coverage=p(x,mask,2)
    expected=(x*mask).sum(2)/mask.float().sum(2).clamp_min(1)
    torch.testing.assert_close(observed,expected);assert torch.isfinite(observed).all() and observed[:,:,:,0].abs().sum()==0
    observed.square().mean().backward();assert p.logits.weight.grad is not None and p.logits.weight.grad.abs().sum()>0

def test_h1_h4_same_initial_and_feedback_active():
    torch.manual_seed(20261032);a=HighResGCTVEncoder(None,'h1');torch.manual_seed(20261032);b=HighResGCTVEncoder(None,'h4')
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=torch.randn(1,12,4,8,12);mask=torch.rand(1,1,4,8,12)>.4
    for u,v in zip(a.forward_statistics(x,mask),b.forward_statistics(x,mask)):torch.testing.assert_close(u,v,rtol=0,atol=0)
    with torch.no_grad():b.f1.update2.weight.fill_(.01)
    assert not torch.allclose(b.forward_statistics(x,mask)[0],b.forward_statistics(x,mask,enable_feedback=False)[0])

def test_all_h_shapes_except_cuda_dcn():
    for name in ('h1','h2','h4','h5'):
        model=HighResGCTVEncoder(None,name);x=torch.randn(1,12,4,8,12,requires_grad=True);mask=torch.rand(1,1,4,8,12)>.5
        out=model.forward_statistics(x,mask);assert [tuple(v.shape) for v in out]==[(1,128,4,8,12),(1,128,2,4,6),(1,128,1,2,3)]
        sum(y.square().mean() for y in out).backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

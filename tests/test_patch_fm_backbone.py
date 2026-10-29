import pytest
import torch
from fpsgen.hier_occ_flow.patch_fm_backbone import PatchFlowUNet

def test_forward_backward_time_and_condition():
    torch.set_num_threads(2);torch.manual_seed(5)
    model=PatchFlowUNet(activation_checkpointing=False);x=torch.randn(2,1,8,16,16);observed=torch.zeros_like(x)
    cond=[torch.randn(2,128,4,8,8,requires_grad=True),torch.randn(2,128,2,4,4,requires_grad=True),torch.randn(2,128,1,2,2,requires_grad=True)]
    y=model(x,observed,cond,torch.tensor([0.,1.]));assert y.shape==x.shape and y.dtype==torch.float32 and torch.isfinite(y).all()
    y.square().mean().backward();assert all(c.grad is not None and torch.isfinite(c.grad).all() and c.grad.abs().sum()>0 for c in cond)
    # PatchScene zero-initializes each residual's output convolution; time
    # gradients become nonzero after that output receives its first update.
    opt=torch.optim.SGD(model.parameters(),lr=.1);opt.step();opt.zero_grad()
    model(x,observed,cond,torch.tensor([0.,1.])).square().mean().backward()
    assert model.embedding[0].weight.grad is not None and model.embedding[0].weight.grad.abs().sum()>0
    with pytest.raises(ValueError):model(x,observed,cond,torch.tensor([0.,1.01]))

def test_paired_initialization():
    torch.manual_seed(20261032+104729);a=PatchFlowUNet(activation_checkpointing=False)
    torch.manual_seed(20261032+104729);b=PatchFlowUNet(activation_checkpointing=False)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

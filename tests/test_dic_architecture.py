import torch
import torch.nn.functional as F

from fpsgen.models.bev_backbones.dic_core import FinalLayer, GroupNorm, LayerNorm2d


def test_official_final_layer_norm_contract():
    norm = FinalLayer(96, 3).norm_final
    assert isinstance(norm, LayerNorm2d)
    assert not norm.elementwise_affine
    assert norm.eps == 1e-6
    x = torch.randn(2, 96, 5, 7, requires_grad=True)
    expected = F.layer_norm(x.permute(0, 2, 3, 1), (96,), eps=1e-6).permute(0, 3, 1, 2)
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
    norm(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_historical_groupnorm_is_explicit_and_load_compatible():
    assert isinstance(FinalLayer(96, 3, final_norm="groupnorm").norm_final, GroupNorm)


def test_checkpointing_preserves_forward_and_gradients_without_changing_topology():
    import copy
    from fpsgen.models.bev_backbones.dic_core import DiCCore
    torch.manual_seed(42)
    reference=DiCCore(input_size=8,hidden_size=8)
    # Exercise the complete core, not its initially zero final-head output.
    torch.nn.init.normal_(reference.final_layer.out_proj.weight,std=.01)
    checkpointed=copy.deepcopy(reference)
    checkpointed.activation_checkpointing=True
    x=torch.randn(2,3,8,8);t=torch.rand(2)
    outputs=[]
    for core in (reference,checkpointed):
        out=core(x,t);out.square().mean().backward();outputs.append(out)
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    for (name,a),(other,b) in zip(reference.named_parameters(),checkpointed.named_parameters()):
        assert name==other
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

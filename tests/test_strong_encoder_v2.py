import torch
from fpsgen.hier_occ_flow.stage1_network import OccupancyUNet
from fpsgen.hier_occ_flow.strong_encoder_v2 import FullResolutionConditionFlow


def test_s5_zero_residual_and_gradients():
    torch.manual_seed(41)
    reference = OccupancyUNet('fm', (8,16,32,64), False).eval()
    variant = FullResolutionConditionFlow('fm', (8,16,32,64), False).eval()
    missing = variant.load_state_dict(reference.state_dict(), strict=False)
    assert set(missing.missing_keys) == {'fullres_projection.weight', 'fullres_projection.bias'}
    x = torch.randn(1,1,8,16,16)
    mask = torch.zeros_like(x)
    features = [torch.randn(1,128,4,8,8), torch.randn(1,128,2,4,4), torch.randn(1,128,1,2,2)]
    t = torch.tensor([0.25])
    a, b = reference(x,mask,features,t), variant(x,mask,features,t)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    b.square().mean().backward()
    assert torch.isfinite(variant.fullres_projection.weight.grad).all()
    assert variant.fullres_projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        cache = variant.prepare_inference_context(mask, features)
        torch.testing.assert_close(variant(x,mask,features,t), variant.forward_with_inference_context(x,cache,t),rtol=0,atol=0)

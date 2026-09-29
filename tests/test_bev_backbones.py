"""CPU-only shape and zero-condition tests for the dense BEV backbones."""

import pytest
import torch

from fpsgen.models.bev_backbones.condition import DiCConditionEncoder, PixelUConditionEncoder
from fpsgen.models.bev_backbones.dic_core import DiCCore


def test_dic_condition_is_exactly_zero_for_inactive_conditions():
    encoder = DiCConditionEncoder(hidden_size=96)
    features = encoder(torch.zeros(2, 32, 32, 32), torch.zeros(2, 2, 32, 32))
    assert [tuple(item.shape) for item in features] == [(2, 96, 32, 32), (2, 192, 16, 16), (2, 384, 8, 8)]
    assert all(torch.count_nonzero(item) == 0 for item in features)


def test_dic_tiny_spatial_forward_keeps_bchw_shape():
    core = DiCCore(input_size=32)
    output = core(torch.randn(1, 3, 32, 32), torch.tensor([500.0]))
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not hasattr(torch.nn.functional, "scaled_dot_product_attention"),
                    reason="PixelU requires the PyTorch-2 SDPA API")
def test_pixelu_condition_is_exactly_zero_for_inactive_conditions():
    encoder = PixelUConditionEncoder(patch_size=16, bottleneck_dim=16, hidden_size=96, context_tokens=32)
    patch, context = encoder(torch.zeros(2, 32, 64, 64), torch.zeros(2, 2, 64, 64))
    assert patch.shape == (2, 16, 96)
    assert context.shape == (2, 32, 96)
    assert torch.count_nonzero(patch) == 0
    assert torch.count_nonzero(context) == 0


@pytest.mark.skipif(not hasattr(torch.nn.functional, "scaled_dot_product_attention"),
                    reason="PixelU requires the PyTorch-2 SDPA API")
def test_pixelu_tiny_spatial_forward_keeps_bchw_shape():
    from fpsgen.models.bev_backbones.pixelu_core import UiTCore
    core = UiTCore(input_size=64, hidden_size=96, num_heads=6, bottleneck_dim=16)
    output = core(torch.randn(1, 3, 64, 64), torch.tensor([500.0]))
    assert output.shape == (1, 3, 64, 64)
    assert torch.isfinite(output).all()

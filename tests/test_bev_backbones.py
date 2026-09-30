"""CPU-only shape and zero-condition tests for the dense BEV backbones."""

import pytest
import torch

from fpsgen.models.bev_backbones.condition import DiCConditionEncoder, PixelUConditionEncoder
from fpsgen.models.bev_backbones.dic_core import DiCCore
from fpsgen.models.bev_backbones.hdit_core import HDiTConditionEncoder, HDiTCore
from fpsgen.models.bev_backbones.dip_bev import DiPConditionEncoder
from fpsgen.models.bev_backbones.dip_core import DiPCore
from fpsgen.models.bev_backbones.ncsnpp_core import NCSNConditionEncoder, NCSNppCore


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


def test_hdit_condition_is_exactly_zero_for_inactive_conditions():
    encoder = HDiTConditionEncoder(widths=(16, 32, 64), patch_size=4, mapping_width=32)
    features = encoder(torch.zeros(2, 32, 64, 64), torch.zeros(2, 2, 64, 64))
    assert [tuple(item.shape) for item in features[:3]] == [(2, 16, 16, 16), (2, 8, 8, 32), (2, 4, 4, 64)]
    assert all(torch.count_nonzero(item) == 0 for item in features)


def test_hdit_tiny_spatial_forward_keeps_bchw_shape_at_all_times():
    core = HDiTCore(input_size=64, patch_size=4, widths=(16, 32, 64), depths=(2, 2, 4),
                    d_ffs=(48, 96, 192), d_head=16, mapping_width=32, mapping_d_ff=64)
    for time in (0.0, 0.5, 1.0):
        output = core(torch.randn(1, 3, 64, 64), torch.tensor([time]))
        assert output.shape == (1, 3, 64, 64)
        assert torch.isfinite(output).all()


@pytest.mark.skipif(not hasattr(torch.nn.functional, "scaled_dot_product_attention"),
                    reason="DiP requires the PyTorch-2 SDPA API")
def test_dip_condition_is_exactly_zero_for_inactive_conditions():
    encoder = DiPConditionEncoder(patch_size=16, hidden_size=96)
    patch, global_condition = encoder(torch.zeros(2, 32, 64, 64), torch.zeros(2, 2, 64, 64))
    assert patch.shape == (2, 16, 96) and global_condition.shape == (2, 96)
    assert torch.count_nonzero(patch) == 0 and torch.count_nonzero(global_condition) == 0


@pytest.mark.skipif(not hasattr(torch.nn.functional, "scaled_dot_product_attention"),
                    reason="DiP requires the PyTorch-2 SDPA API")
def test_dip_tiny_spatial_forward_keeps_bchw_shape_at_all_times():
    core = DiPCore(input_size=64, patch_size=16, hidden_size=192, num_groups=3, num_cond_blocks=2)
    for time in (0.0, 0.5, 1.0):
        output = core(torch.randn(1, 3, 64, 64), torch.tensor([time]))
        assert output.shape == (1, 3, 64, 64)
        assert torch.isfinite(output).all()


def test_ncsnpp_condition_is_exactly_zero_for_inactive_conditions():
    encoder = NCSNConditionEncoder(nf=16)
    maps, global_condition = encoder(torch.zeros(2, 32, 64, 64), torch.zeros(2, 2, 64, 64))
    assert [tuple(item.shape) for item in maps] == [(2, 16, 64, 64), (2, 16, 32, 32), (2, 32, 16, 16), (2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2), (2, 32, 1, 1)]
    assert all(torch.count_nonzero(item) == 0 for item in maps)
    assert torch.count_nonzero(global_condition) == 0


def test_ncsnpp_tiny_spatial_forward_keeps_bchw_shape_at_all_times():
    core = NCSNppCore(input_size=64, nf=16)
    for time in (0.0, 500.0, 1000.0):
        output = core(torch.randn(1, 3, 64, 64), torch.tensor([time]))
        assert output.shape == (1, 3, 64, 64)
        assert torch.isfinite(output).all()

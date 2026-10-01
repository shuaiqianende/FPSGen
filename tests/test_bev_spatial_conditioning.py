"""Contract tests for the concat34 spatial-control study."""
import torch

from fpsgen.models.bev_conditioning.condition_packet import make_condition_packet
from fpsgen.models.bev_conditioning.pixelcontrol_adapter import PixelControlAdapter, PixelControlEncoder
from fpsgen.models.bev_conditioning.spatial_ops import layout_boundary
from fpsgen.models.bev_backbones.synflow_core import SynConditionEncoder, SynFlowCore


def _inputs(size=64):
    return torch.randn(2, 32, size, size), torch.randn(2, 2, size, size)


def test_concat34_zero_state_is_exactly_input_invariant():
    raw, layout = _inputs()
    keep = torch.zeros(2, 3, dtype=torch.bool)
    first = make_condition_packet(raw, layout, keep).map
    second = make_condition_packet(torch.randn_like(raw) * 100, torch.randn_like(layout) * 100, keep).map
    assert first.shape == (2, 34, 64, 64)
    assert torch.count_nonzero(first) == 0
    assert torch.equal(first, second)


def test_concat34_single_condition_isolation():
    raw, layout = _inputs()
    keep = torch.tensor([[True, False, False], [False, True, False]])
    first = make_condition_packet(raw, layout, keep).map
    changed = make_condition_packet(raw, torch.randn_like(layout), keep).map
    assert torch.equal(first[0], changed[0])
    changed_raw = make_condition_packet(torch.randn_like(raw), layout, keep).map
    assert torch.equal(first[1], changed_raw[1])


def test_spatial_pyramid_preserves_left_right_alignment():
    condition = torch.zeros(1, 34, 64, 64)
    condition[:, 32, 12:20, 4:12] = 1
    encoder = SynConditionEncoder((8, 8, 16, 16, 32, 32))
    with torch.no_grad():
        encoder.stem[0].weight.zero_()
        encoder.stem[0].weight[0, 32, 1, 1] = 1
    feature = encoder(condition)[0][0, 0]
    assert feature[:, :32].abs().sum() > feature[:, 32:].abs().sum()


def test_boundary_operator_respects_vehicle_road_keep_mask():
    layout = torch.full((1, 2, 16, 16), -1.)
    layout[:, 0, 4:12, 4:12] = 1
    off = layout_boundary(layout, torch.tensor([[True, False, False]]))
    on = layout_boundary(layout, torch.tensor([[True, True, False]]))
    assert torch.count_nonzero(off) == 0 and torch.count_nonzero(on) > 0


def test_synflow_tiny_is_finite_for_generic_and_spade_paths():
    pyramid_encoder = SynConditionEncoder((8, 8, 16, 16, 32, 32))
    condition = pyramid_encoder(torch.randn(1, 34, 64, 64))
    for injection in ("generic_add", "spade"):
        core = SynFlowCore(input_size=64, model_channels=8, channel_mult=(1, 1, 2, 2, 4, 4),
                           num_res_blocks=1, attention_spatial_resolutions=(8, 4, 2), num_head_channels=8,
                           injection=injection)
        result = core(torch.randn(1, 3, 64, 64), torch.tensor([.25]), condition)
        assert result.shape == (1, 3, 64, 64) and torch.isfinite(result).all()


def test_pixelcontrol_tokens_and_zero_adapter_contract():
    encoder = PixelControlEncoder(32, base_channels=4, max_channels=16)
    zero_tokens = encoder(torch.zeros(2, 34, 256, 256))
    active_tokens = encoder(torch.ones(1, 34, 256, 256))
    assert zero_tokens.shape == (2, 256, 32) and torch.count_nonzero(zero_tokens) == 0
    assert active_tokens.norm() > 0
    adapter = PixelControlAdapter(32, depth=2, gate_init=1.)
    feature = torch.randn(1, 256, 32)
    residual = adapter.residual(0, active_tokens, feature)
    assert torch.count_nonzero(residual) == 0
    assert adapter.gates[0].item() == 1.

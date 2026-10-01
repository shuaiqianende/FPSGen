"""Neutrality and condition contracts for NCSN++ spatial-adaptive injection."""
import torch

from fpsgen.models.bev_backbones.ncsnpp_core import BigGANResBlock, NCSNConditionEncoder, NCSNppCore
from fpsgen.models.bev_backbones.ncsnpp_bev import BEVNCSNppS
from fpsgen.models.bev_conditioning.ncsnpp_spade import NCSNppSPADEPair


def test_zero_initialized_spade_pair_is_exactly_neutral_for_resblock():
    torch.manual_seed(7)
    block = BigGANResBlock(16, 16, 32)
    pair = NCSNppSPADEPair(16, 16, 16, cond_dim=4)
    x, temb, condition = torch.randn(2, 16, 8, 8), torch.randn(2, 32), torch.randn(2, 16, 8, 8)
    baseline = block(x, temb)
    spade = block(x, temb, condition, pair)
    assert torch.equal(baseline, spade)
    assert pair.norm1.last_gamma_rms == 0 and pair.norm2.last_beta_rms == 0


def test_neutral_spade_core_matches_same_core_with_spatial_route_disabled():
    torch.manual_seed(8)
    baseline = NCSNppCore(input_size=64, nf=16)
    for head in baseline.output_heads:
        torch.nn.init.normal_(head[-1].weight, std=.01)
    adapted = NCSNppCore(input_size=64, nf=16)
    adapted.load_state_dict(baseline.state_dict())
    adapted.enable_spade_conditioning(cond_dim=4)
    maps = tuple(torch.randn(1, channel, 64 // (2 ** level), 64 // (2 ** level))
                 for level, channel in enumerate(baseline.channels))
    xt, t, global_value = torch.randn(1, 3, 64, 64), torch.tensor([500.]), torch.randn(1, 64)
    assert torch.equal(baseline(xt, t, None, global_value), adapted(xt, t, None, global_value, None, maps))


def test_ncsnpp_spade_tiny_core_is_finite_and_gradients_open_after_update():
    torch.manual_seed(9)
    core = NCSNppCore(input_size=64, nf=16)
    core.enable_spade_conditioning(cond_dim=4)
    encoder = NCSNConditionEncoder(nf=16)
    raw = torch.randn(1, 32, 64, 64)
    layout = torch.randn(1, 2, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    optimizer = torch.optim.AdamW(list(core.parameters()) + list(encoder.parameters()), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        maps, global_value = encoder(raw, layout)
        output = core(torch.randn(1, 3, 64, 64), torch.tensor([500.]), None, global_value, None, maps)
        (output - target).square().mean().backward(); optimizer.step()
    assert output.shape == (1, 3, 64, 64) and torch.isfinite(output).all()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in encoder.parameters())
    diagnostics = core.spade_diagnostics()
    assert any(key.startswith("gamma_") for key in diagnostics)


def test_ncsnpp_spade_condition_off_is_exactly_invariant_to_source_contents():
    """An explicit 000 state must never leak source values into N1."""
    torch.manual_seed(10)
    model = BEVNCSNppS({
        "condition": {"fusion": "shared", "spatial": True, "global": True,
                      "gate_init": 0.1, "lidar_channels": 32, "layout_channels": 2,
                      "spade": {"enabled": True, "cond_dim": 4}},
        "ncsnpp": {"input_size": 64, "nf": 16, "ch_mult": [1, 1, 2, 2, 2, 2, 2],
                    "num_res_blocks": 2, "attn_resolutions": [16]},
    }).eval()
    xt, t = torch.randn(1, 3, 64, 64), torch.tensor([0.4])
    keep = torch.zeros(1, 3, dtype=torch.bool)
    with torch.no_grad():
        first = model(xt, t, torch.randn(1, 32, 64, 64), torch.randn(1, 2, 64, 64), keep)
        second = model(xt, t, torch.randn(1, 32, 64, 64) * 100, torch.randn(1, 2, 64, 64) * 100, keep)
    assert torch.equal(first, second)

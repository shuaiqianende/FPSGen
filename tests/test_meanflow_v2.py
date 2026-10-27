import copy
import pytest
import torch
from torch import nn
from fpsgen.hier_occ_flow.meanflow_v2 import (
    adaptive_scene_objective, meanflow_v2_loss, preserved_target_buffers, scene_diagnostics)
from fpsgen.hier_occ_flow.meanflow_math import meanflow_loss
from fpsgen.hier_occ_flow.meanflow_jvp import endpoint_jvp


def example():
    x = torch.tensor([[.3, -.2], [.6, -.4]], dtype=torch.double)
    t = torch.tensor([.2, .3], dtype=torch.double)
    s = torch.tensor([.7, .8], dtype=torch.double)
    v = torch.tensor([[1., -2.], [2., -1.]], dtype=torch.double)
    return x, t, s, v, torch.ones_like(x, dtype=torch.bool)


def test_adaptive_weight_is_detached_and_scene_mse_not_sum():
    raw = torch.tensor([1., 9.], requires_grad=True)
    loss, weight = adaptive_scene_objective(raw)
    assert not weight.requires_grad
    assert torch.equal(weight, (raw.detach() + .01).reciprocal())
    loss.backward()
    assert torch.equal(raw.grad, weight / 2)


def test_a_preserves_original_jvp_and_exact_adaptive_objective():
    x, t, s, v, mask = example()
    fn = lambda z, a, b: .3 * z.square() + 2 * a[:, None] + 7 * b[:, None]
    expected, old = meanflow_loss(fn, x, t, s, v, mask, adaptive_power=1., adaptive_epsilon=.01)
    actual, new = meanflow_v2_loss(fn, x, t, s, v, mask, arm='MFV2-A')
    assert torch.equal(actual, expected)
    assert torch.equal(old['derivative'], new['derivative'])
    assert torch.equal(old['target'], new['target'])


def test_forward_sign_in_analytic_exponential_field():
    x, t, s, _, _ = example()
    a = .7
    fn = lambda z, t, s: z * torch.expm1(a * (s - t)[:, None]) / (s - t)[:, None]
    primal, jvp = endpoint_jvp(fn, x, t, s, a * x)
    assert torch.allclose(primal - (s - t)[:, None] * jvp, a * x, atol=1e-12)
    assert not torch.allclose(primal + (s - t)[:, None] * jvp, a * x)


def test_b_jvp_uses_boundary_and_holds_endpoint_fixed():
    x, t, s, v, mask = example()
    def fn(z, a, b):
        return .3 * z.square() + 2 * a[:, None] + 7 * b[:, None]
    _, d = meanflow_v2_loss(fn, x, t, s, v, mask, arm='MFV2-B')
    boundary = fn(x, t, t)
    assert torch.equal(d['boundary_velocity'], boundary)
    assert torch.allclose(d['derivative'], .6 * x * boundary + 2)
    assert not torch.allclose(d['derivative'], .6 * x * v + 2)
    assert torch.allclose(d['prediction'], fn(x, t, s).float() - (s-t)[:, None] * d['derivative'].float())


def test_b_diagonal_reduces_to_fm_raw_and_adaptive_gradient():
    x, t, _, v, mask = example()
    w = nn.Parameter(torch.tensor(.4, dtype=torch.double))
    fn = lambda z, a, b: w * z + a[:, None]
    loss, d = meanflow_v2_loss(fn, x, t, t, v, mask, arm='MFV2-B')
    raw = (fn(x, t, t).float() - v.float()).square().mean(1)
    expected, _ = adaptive_scene_objective(raw)
    assert torch.equal(d['per_scene'], raw) and torch.equal(loss, expected)
    assert d['jvp_calls'] == d['boundary_forwards'] == 0
    assert torch.equal(torch.autograd.grad(loss, w, retain_graph=True)[0], torch.autograd.grad(expected, w)[0])


def test_conditional_tangent_substitution_v_space_equals_u_space():
    x, t, s, v, mask = example()
    # Boundary output is constructed to be exactly the conditional velocity.
    fn = lambda z, a, b: v + (b-a)[:, None] * .3 * z
    a, da = meanflow_v2_loss(fn, x, t, s, v, mask, arm='MFV2-A')
    b, db = meanflow_v2_loss(fn, x, t, s, v, mask, arm='MFV2-B')
    assert torch.equal(da['derivative'], db['derivative'])
    assert torch.allclose(da['per_scene'], db['per_scene'], atol=1e-7)
    assert torch.allclose(a, b, atol=1e-7)


@pytest.mark.parametrize('arm', ['MFV2-A', 'MFV2-B'])
def test_encoder_gradient_survives_and_targets_have_no_reverse_graph(arm):
    x, t, s, v, mask = example()
    encoder = nn.Linear(2, 2).double()
    flow = nn.Linear(2, 2).double()
    features = encoder(x)
    fn = lambda z, a, b: flow(z + features) + a[:, None] + b[:, None]
    loss, d = meanflow_v2_loss(fn, x, t, s, v, mask, arm=arm, model=flow)
    assert not d['derivative'].requires_grad and not d['target'].requires_grad
    if d['boundary_velocity'] is not None:
        assert not d['boundary_velocity'].requires_grad
    loss.backward()
    for m in (encoder, flow):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert sum(p.grad.abs().sum() for p in m.parameters()) > 0


@pytest.mark.parametrize('arm', ['MFV2-A', 'MFV2-B'])
def test_invalid_voxels_do_not_enter_function_or_loss(arm):
    x, t, s, v, mask = example()
    mask[:, 1] = False
    x[:, 1] = float('nan')
    v[:, 1] = float('inf')
    seen = []
    def fn(z, a, b):
        seen.append(z.detach().clone())
        return z + a[:, None] + b[:, None]
    loss, d = meanflow_v2_loss(fn, x, t, s, v, mask, arm=arm)
    assert torch.isfinite(loss)
    assert all(torch.equal(z[:, 1], torch.zeros(2, dtype=z.dtype)) for z in seen)
    assert torch.equal(d['prediction'][:, 1], torch.zeros(2, dtype=d['prediction'].dtype))


def test_norm_buffers_restored_on_success_and_exception():
    m = nn.BatchNorm1d(2).train()
    before = copy.deepcopy(m.state_dict())
    with preserved_target_buffers(m), torch.no_grad():
        m(torch.randn(8, 2))
    assert all(torch.equal(before[k], v) for k,v in m.state_dict().items())
    with pytest.raises(RuntimeError, match='intentional'):
        with preserved_target_buffers(m), torch.no_grad():
            m(torch.randn(8, 2))
            raise RuntimeError('intentional')
    assert all(torch.equal(before[k], v) for k,v in m.state_dict().items())


def test_scene_diagnostics_exclude_invalid_and_split_binary_labels():
    x, t, s, v, mask = example()
    _, detail = meanflow_v2_loss(lambda z,a,b: z+a[:,None], x,t,s,v,mask,arm='MFV2-B')
    rows = scene_diagnostics(detail, torch.tensor([[1.,-1.],[1.,-1.]]), mask)
    assert len(rows) == 2 and all(r['positive_count'] == r['negative_count'] == 1 for r in rows)
    assert all(r['v_est_rms'] is not None and r['jvp_rms'] >= 0 for r in rows)

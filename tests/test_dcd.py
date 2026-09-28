"""CUDA-unit tests for the training-safe DCD implementation."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DCD uses FPSGen's CUDA Chamfer extension"
)


def _dcd():
    from fpsgen.ops.dcd import density_aware_chamfer
    return density_aware_chamfer


def test_dcd_identical_cloud_is_zero():
    points = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], device="cuda"
    )
    loss, _, _ = _dcd()(points, points.clone(), alpha=1.0, n_lambda=1.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_dcd_one_to_one_two_cm_offset_has_expected_scale():
    target = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], device="cuda"
    )
    pred = target + torch.tensor([0.02, 0., 0.], device="cuda")
    loss, _, _ = _dcd()(pred, target, alpha=1.0, n_lambda=1.0)
    expected = 1.0 - torch.exp(torch.tensor(-0.02 ** 2, device="cuda"))
    assert torch.allclose(loss, expected.unsqueeze(0), atol=2e-5)


def test_dcd_many_to_one_assignment_is_penalized():
    target = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]], device="cuda")
    one_to_one = target.clone()
    many_to_one = torch.tensor(
        [[[0., 0., 0.], [0.001, 0., 0.], [2., 0., 0.]]], device="cuda"
    )
    dcd = _dcd()
    clean_loss, _, _ = dcd(one_to_one, target, alpha=1.0, n_lambda=1.0)
    collapsed_loss, _, _ = dcd(many_to_one, target, alpha=1.0, n_lambda=1.0)
    assert collapsed_loss.item() > clean_loss.item()


def test_dcd_gradient_is_finite():
    target = torch.rand((1, 32, 3), device="cuda")
    pred = (target + 0.02 * torch.randn_like(target)).requires_grad_(True)
    loss, _, _ = _dcd()(pred, target, alpha=1.0, n_lambda=1.0)
    loss.mean().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

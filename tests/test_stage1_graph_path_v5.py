import pytest
import torch

from fpsgen.hier_occ_flow.stage1_network import OccupancyUNet


def test_validated_inference_forward_matches_default_forward():
    torch.manual_seed(17)
    model = OccupancyUNet("fm", widths=(8, 8, 8, 8), activation_checkpointing=False,
                          embed_width=16).eval()
    state = torch.randn(1, 1, 16, 16, 16)
    observed = torch.randint(0, 2, state.shape).float()
    condition = (torch.randn(1, 128, 8, 8, 8),
                 torch.randn(1, 128, 4, 4, 4),
                 torch.randn(1, 128, 2, 2, 2))
    time = torch.tensor(0.5)
    with torch.no_grad():
        reference = model(state, observed, condition, time)
        validated = model.forward_inference_validated(state, observed, condition, time)
    assert torch.equal(reference, validated)


def test_validated_inference_forward_rejects_training_or_grad_mode():
    model = OccupancyUNet("fm", widths=(8, 8, 8, 8), activation_checkpointing=False,
                          embed_width=16).eval()
    state = torch.zeros(1, 1, 16, 16, 16)
    observed = torch.zeros_like(state)
    condition = (torch.zeros(1, 128, 8, 8, 8),
                 torch.zeros(1, 128, 4, 4, 4),
                 torch.zeros(1, 128, 2, 2, 2))
    with pytest.raises(RuntimeError, match="eval and no_grad"):
        model.forward_inference_validated(state, observed, condition, torch.tensor(0.5))
    model.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval and no_grad"):
        model.forward_inference_validated(state, observed, condition, torch.tensor(0.5))

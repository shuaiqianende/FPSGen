import torch

from fpsgen.hier_occ_flow.stage1_network import OccupancyUNet


def _inputs(device="cpu"):
    state = torch.randn((1, 1, 16, 16, 16), device=device)
    observed = torch.zeros_like(state)
    condition = (
        torch.randn((1, 128, 8, 8, 8), device=device),
        torch.randn((1, 128, 4, 4, 4), device=device),
        torch.randn((1, 128, 2, 2, 2), device=device),
    )
    return state, observed, condition


def test_cached_condition_projection_matches_uncached_cpu():
    torch.manual_seed(21)
    model = OccupancyUNet("fm", widths=(8, 16, 32, 64), activation_checkpointing=False).eval()
    state, observed, condition = _inputs()
    with torch.no_grad():
        context = model.prepare_inference_context(observed, condition)
        assert len(context["condition_projected"]) == 3
        current_a, current_b = state.clone(), state.clone()
        for step in range(2):
            t = state.new_tensor(step / 2)
            uncached = model(current_a, observed, condition, t)
            cached = model(current_b, observed, None, t, inference_context=context)
            assert torch.equal(uncached, cached)
            current_a = current_a + uncached / 2
            current_b = current_b + cached / 2
        assert torch.equal(current_a, current_b)


def test_condition_cache_is_inference_only():
    model = OccupancyUNet("fm", activation_checkpointing=False).eval()
    _, observed, condition = _inputs()
    try:
        model.prepare_inference_context(observed, condition)
    except RuntimeError as exc:
        assert "inference-only" in str(exc)
    else:
        raise AssertionError("cache preparation with gradients enabled must fail")


def test_cached_context_rejects_another_observed_mask():
    model = OccupancyUNet("fm", activation_checkpointing=False).eval()
    state, observed, condition = _inputs()
    with torch.no_grad():
        context = model.prepare_inference_context(observed, condition)
        try:
            model(state, torch.zeros_like(observed), None, state.new_zeros(()), inference_context=context)
        except ValueError as exc:
            assert "different observed mask" in str(exc)
        else:
            raise AssertionError("context reuse with a different mask must fail")


def test_cached_condition_projection_matches_cuda_autocast():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(33)
    model = OccupancyUNet("fm", widths=(8, 16, 32, 64), activation_checkpointing=False).cuda().eval()
    state, observed, condition = (x.cuda() for x in _inputs())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        context = model.prepare_inference_context(observed, condition)
        current_a, current_b = state.clone(), state.clone()
        for step in range(2):
            t = state.new_tensor(step / 2)
            uncached = model(current_a, observed, condition, t)
            cached = model(current_b, observed, None, t, inference_context=context)
            assert torch.equal(uncached, cached)
            current_a = current_a + uncached / 2
            current_b = current_b + cached / 2
        assert torch.equal(current_a, current_b)

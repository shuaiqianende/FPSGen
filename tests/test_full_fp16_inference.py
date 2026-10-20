import pytest
import torch
from torch import nn

from fpsgen.hier_occ_flow.full_fp16_inference import (
    explicit_fp16_copy,
    parameter_dtype_audit,
)


def test_explicit_fp16_copy_converts_parameters_but_preserves_frequency_buffer():
    module = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    module.register_buffer("frequencies", torch.arange(4, dtype=torch.float32))

    audit, hooks = explicit_fp16_copy(module)

    assert hooks == []
    assert all(parameter.dtype == torch.float16 for parameter in module.parameters())
    assert module.frequencies.dtype == torch.float32
    assert audit["parameter_bytes"] == sum(p.numel() * 2 for p in module.parameters())
    assert audit["fp32_frequency_buffers"] == ["frequencies"]
    assert parameter_dtype_audit(module)["parameter_bytes_by_dtype"] == {
        "torch.float16": audit["parameter_bytes"]
    }


def test_selective_norm_runs_fp32_and_returns_activation_dtype():
    class ToHalf(nn.Module):
        def forward(self, value):
            return value.half()

    module = nn.Sequential(ToHalf(), nn.GroupNorm(2, 4)).eval()
    audit, hooks = explicit_fp16_copy(module, selective_norm=True)
    norm = module[1]

    assert len(hooks) == 2
    assert norm.weight.dtype == torch.float32
    assert norm.bias.dtype == torch.float32
    assert audit["norm_modules_fp32"] == ["1"]

    output = module(torch.randn(2, 4, dtype=torch.float32))
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_explicit_fp16_requires_eval_mode():
    with pytest.raises(ValueError, match="call eval"):
        explicit_fp16_copy(nn.Linear(2, 2).train())

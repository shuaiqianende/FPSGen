"""Opt-in explicit-FP16 copies for frozen HS-OccFlow inference only.

Geometry, integer indices, RNG/source state, time-frequency buffers and Euler
state are deliberately kept in their original precision. The caller must load
the original FP32 checkpoint before invoking these helpers.
"""
from __future__ import annotations

import torch
from torch import nn


def _norm_modules(module):
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm,
                  nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    return [(name, child) for name, child in module.named_modules()
            if isinstance(child, norm_types)]


def explicit_fp16_copy(module: nn.Module, *, selective_norm: bool = False):
    """Convert an already-loaded inference module and return an audit record.

    Selective mode leaves norm parameters/statistics FP32 and places explicit
    FP32 input/output precision islands around normalization. The normalized
    result is cast back to the incoming activation dtype.
    """
    if module.training:
        raise ValueError("Explicit FP16 conversion is inference-only; call eval() first")
    norms = _norm_modules(module)
    module.half()
    hooks = []
    if selective_norm:
        for name, norm in norms:
            norm.float()
            state = {"input_dtype": None}

            def pre_hook(_mod, args, _state=state):
                if args and torch.is_tensor(args[0]):
                    _state["input_dtype"] = args[0].dtype
                    if args[0].is_floating_point() and args[0].dtype != torch.float32:
                        return (args[0].float(), *args[1:])
                return None

            def output_hook(_mod, _args, output, _state=state):
                dtype = _state["input_dtype"]
                if dtype is None:
                    return output
                if torch.is_tensor(output) and output.is_floating_point():
                    return output.to(dtype)
                return output

            hooks.append(norm.register_forward_pre_hook(pre_hook))
            hooks.append(norm.register_forward_hook(output_hook))

    # Preserve time/frequency math and any non-parameter floating buffers.
    # Occupancy validity buffers are Boolean and remain unchanged.
    for name, buffer in module.named_buffers():
        if name.endswith("frequencies") and buffer.is_floating_point():
            buffer.data = buffer.data.float()

    params = list(module.parameters())
    audit = {
        "mode": "fp16_selective_norm" if selective_norm else "fp16_full",
        "parameter_count": sum(p.numel() for p in params),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
        "parameter_dtypes": dict(sorted({str(p.dtype): sum(q.numel() for q in params if q.dtype == p.dtype)
                                          for p in params}.items())),
        "norm_modules_fp32": [name for name, _ in norms] if selective_norm else [],
        "norm_precision_hooks": len(hooks),
        "fp32_frequency_buffers": [name for name, b in module.named_buffers()
                                   if name.endswith("frequencies") and b.dtype == torch.float32],
    }
    return audit, hooks


def parameter_dtype_audit(module: nn.Module):
    totals = {}
    module_rows = []
    for name, child in module.named_modules():
        direct = list(child.parameters(recurse=False))
        if not direct:
            continue
        dtypes = sorted({str(p.dtype) for p in direct})
        nbytes = sum(p.numel() * p.element_size() for p in direct)
        for p in direct:
            totals[str(p.dtype)] = totals.get(str(p.dtype), 0) + p.numel() * p.element_size()
        module_rows.append({"module": name or "<root>", "type": type(child).__name__,
                            "parameter_dtypes": dtypes, "parameter_bytes": nbytes})
    return {"parameter_bytes_by_dtype": totals, "modules": module_rows}

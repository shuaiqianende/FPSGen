"""Opt-in Stage1 3D channels-last memory-format experiment."""
from __future__ import annotations

import torch
from torch import nn


def enable_channels_last_3d(module: nn.Module):
    """Convert Conv3d weights/5D buffers and inputs without touching 1D params.

    ``Module.to(memory_format=...)`` applies the format to every tensor and
    fails on the encoder's vector parameters. This helper only changes 5D
    tensors and inserts layout conversions at Conv3d boundaries. It returns
    hook handles so callers can scope the experiment to one inference model.
    """
    handles = []
    for layer in module.modules():
        if not isinstance(layer, nn.Conv3d):
            continue
        with torch.no_grad():
            layer.weight.data = layer.weight.data.contiguous(
                memory_format=torch.channels_last_3d
            )
            if layer.weight.grad is not None:
                layer.weight.grad = layer.weight.grad.contiguous(
                    memory_format=torch.channels_last_3d
                )
            for name, value in tuple(layer._buffers.items()):
                if isinstance(value, torch.Tensor) and value.ndim == 5:
                    layer._buffers[name] = value.contiguous(
                        memory_format=torch.channels_last_3d
                    )

        def layout_input(_layer, args):
            if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 5:
                value = args[0]
                if not value.is_contiguous(memory_format=torch.channels_last_3d):
                    value = value.contiguous(memory_format=torch.channels_last_3d)
                return (value,) + args[1:]
            return args

        handles.append(layer.register_forward_pre_hook(layout_input))
    return handles


def remove_layout_hooks(handles):
    for handle in handles:
        handle.remove()

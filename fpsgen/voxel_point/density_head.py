"""FP32 nonnegative conversion; semantic logits are never exposed."""

import torch
import torch.nn.functional as F


def nonnegative_density(raw, valid):
    with torch.autocast(device_type=raw.device.type, enabled=False):
        density = F.softplus(raw.float()) * valid
    if not bool(torch.isfinite(density).all()):
        raise ValueError("Nonfinite network density")
    return density

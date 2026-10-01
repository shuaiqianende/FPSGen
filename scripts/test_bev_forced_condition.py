#!/usr/bin/env python3
"""Synthetic same-noise/different-layout condition-path diagnostic.

This is intentionally not a quality benchmark.  It gives a selected new
wrapper a fixed ``t=0`` pair: identical x0, left/right vehicle rectangles, and
left/right occupancy targets.  A control path that cannot lower paired loss
within 300 updates should not enter a long experimental queue.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from fpsgen.models.bev_backbones import build_bev_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.steps <= 300: raise ValueError("steps must be in [1,300]")
    torch.manual_seed(20261001)
    model = build_bev_backbone(yaml.safe_load(args.config.read_text())).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x0 = torch.randn(2, 3, 256, 256, device=args.device)
    raw = torch.zeros(2, 32, 256, 256, device=args.device)
    layout = torch.full((2, 2, 256, 256), -1., device=args.device)
    layout[0, 0, 96:160, 24:96] = 1.
    layout[1, 0, 96:160, 160:232] = 1.
    target = torch.full_like(x0, -1.)
    target[0, 2, 96:160, 24:96] = 1.
    target[1, 2, 96:160, 160:232] = 1.
    velocity_target = target - x0
    t = torch.zeros(2, device=args.device)
    keep = torch.tensor([[False, True, False], [False, True, False]], device=args.device)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x0, t, raw, layout, keep)
        loss = (prediction - velocity_target).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError("non-finite forced-condition loss")
        loss.backward(); optimizer.step()
    with torch.no_grad():
        correct = (model(x0, t, raw, layout, keep) - velocity_target).square().mean()
        swapped = (model(x0, t, raw, layout.flip(0), keep) - velocity_target).square().mean()
    print({"steps": args.steps, "correct_mse": float(correct), "swapped_mse": float(swapped),
           "condition_effect_directional": bool(correct < swapped)})


if __name__ == "__main__": main()

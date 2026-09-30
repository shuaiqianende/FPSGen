#!/usr/bin/env python3
"""One-real-batch gradient evidence for a prepared dense BEV backbone.

This is a smoke-only probe.  It deliberately does not construct a Trainer or
save a checkpoint.  A caller may request a tiny in-memory warmup when an
official zero-scale output head blocks upstream gradients on its first update.
The normal eight optimizer-step smoke remains the training-path validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from fpsgen.datasets import datasets
from fpsgen.models.gen_img import FlowIMG


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def grad_norm(module: torch.nn.Module) -> float:
    squared = torch.zeros((), device="cuda")
    found = False
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.float().square().sum()
            found = True
    return float(squared.sqrt().item()) if found else 0.0


def named_grad_norm(module: torch.nn.Module, terms) -> float:
    squared = torch.zeros((), device="cuda")
    found = False
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and any(term in name for term in terms):
            squared = squared + parameter.grad.float().square().sum()
            found = True
    return float(squared.sqrt().item()) if found else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-steps", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke gradient probe requires one CUDA-visible GPU")
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["data"]["data_dir"] = args.data_root
    device = torch.device("cuda")
    torch.manual_seed(20261001)
    model = FlowIMG(cfg).to(device).train()
    loader = datasets.dataloaders[cfg["data"]["dataloader"]](cfg).train_dataloader()
    batch = move_to_device(next(iter(loader)), device)
    gt = batch["pcd_full"]
    # Keep every condition active in this dedicated derivative check.  The
    # normal smoke already validates the unchanged uniform 8-state sampler.
    layout = model.processor.get_layout_bev(batch["pcd_full"], batch["full_label"]) * 2.0 - 1.0
    target = model.processor.points_to_bev_target(gt)
    xt = torch.randn_like(target)
    t = torch.full((target.shape[0],), 0.5, device=device)
    weight = torch.tensor([1.0, 2.0, 1.0], device=device).view(1, 3, 1, 1)

    def compute_loss():
        # PointPillar features carry an autograd graph and must be rebuilt for
        # each optional warmup update rather than reused after backward.
        raw_pc = model.model.get_raw_pc_bev(batch["pcd_part"])
        output = model.model(xt, t, raw_pc, layout)
        return ((output - target).square() * weight).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        compute_loss().backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss()
    loss.backward()
    torch.cuda.synchronize()
    result = {
        "backbone": cfg["model"]["backbone"],
        "warmup_steps": args.warmup_steps,
        "loss": float(loss.item()),
        "condition_grad_norm": grad_norm(model.model.condition_encoder),
        "core_grad_norm": grad_norm(model.model.core),
        "output_head_grad_norm": named_grad_norm(model.model.core, ("out", "final", "unpatch")),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_condition_grads": all(torch.isfinite(p.grad).all().item() for p in model.model.condition_encoder.parameters() if p.grad is not None),
        "finite_core_grads": all(torch.isfinite(p.grad).all().item() for p in model.model.core.parameters() if p.grad is not None),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    if result["condition_grad_norm"] <= 0:
        raise RuntimeError(f"Condition path has no gradient: {result}")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

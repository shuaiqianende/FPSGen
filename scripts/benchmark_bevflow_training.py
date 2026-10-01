#!/usr/bin/env python3
"""Reproducible Legacy BEVFlow optimizer-step benchmark.

The script deliberately keeps KNN, rasterization, and loss in FP32.  AMP and
``torch.compile`` are scoped to the dense BEVFlow forward only.  It has no
Lightning logging or visualization side effects and is intended for GPU2/3
performance studies, not quality training.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from fpsgen.datasets import datasets
from fpsgen.models.gen_img import FlowIMG
from fpsgen.utils.training_runtime import apply_training_environment, load_training_config, seed_training


def move(value, device, non_blocking=False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move(item, device, non_blocking) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [move(item, device, non_blocking) for item in value]
    return value


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


class CudaTimer:
    def __call__(self, fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); result = fn(); end.record(); end.synchronize()
        return result, start.elapsed_time(end)


def make_optimizer(model, args, cfg):
    kwargs = dict(lr=float(cfg["train"]["lr"]), betas=(.9, .999), weight_decay=1e-4)
    if args.optimizer == "fused":
        kwargs["fused"] = True
    elif args.optimizer == "foreach":
        kwargs["foreach"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def one_step(flow, batch, optimizer, timer, precision, scaler, breakdown):
    """One exact-contract manual optimizer step, returning optional timings."""
    B = batch["pcd_full"].shape[0]
    device = flow.device
    timings = {}

    def measured(name, fn):
        if not breakdown:
            return fn()
        value, elapsed = timer(fn); timings[name] = elapsed; return value

    state = torch.multinomial(flow.condition_mode_probabilities, B, replacement=True,
                              generator=flow._flow_generator("condition", 1003))
    drop_lidar, drop_vehicle, drop_road = ~((state & 4) > 0), ~((state & 2) > 0), ~((state & 1) > 0)
    lidar_mask, vehicle_mask, road_mask = (drop_lidar.view(B, 1, 1, 1),
                                            drop_vehicle.view(B, 1, 1, 1),
                                            drop_road.view(B, 1, 1, 1))
    layout = measured("layout_ms", lambda: flow.processor.get_layout_bev(batch["pcd_full"], batch["full_label"]) * 2.0 - 1.0)
    layout[:, :1] = torch.where(vehicle_mask, torch.zeros_like(layout[:, :1]), layout[:, :1])
    layout[:, 1:2] = torch.where(road_mask, torch.zeros_like(layout[:, 1:2]), layout[:, 1:2])
    raw = measured("knn_pillar_ms", lambda: flow.model.get_raw_pc_bev(batch["pcd_part"]))
    raw = torch.where(lidar_mask, torch.zeros_like(raw), raw)
    x1 = measured("target_ms", lambda: flow.processor.points_to_bev_target(batch["pcd_full"]))
    t = torch.rand((B,), device=device, generator=flow._flow_generator("time", 2003))
    expand = t.view(B, 1, 1, 1)
    x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=flow._flow_generator("noise", 3003))
    xt, target = (1 - expand) * x0 + expand * x1, x1 - x0

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    def dense():
        if amp_dtype is None:
            return flow._dense_forward(xt, t, raw, layout)
        with torch.autocast("cuda", dtype=amp_dtype):
            return flow._dense_forward(xt, t, raw, layout)
    prediction = measured("dense_forward_ms", dense)
    def loss_fn():
        return (torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none") *
                flow.fm_channel_weights.view(1, -1, 1, 1)).mean()
    loss = measured("loss_ms", loss_fn)
    optimizer.zero_grad(set_to_none=True)
    def backward():
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
    measured("backward_ms", backward)
    def step_optimizer():
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer); scaler.update()
    measured("optimizer_ms", step_optimizer)
    return float(loss.detach()), timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("e2e", "fixed"), default="fixed")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--compile-dense", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--optimizer", choices=("default", "fused", "foreach"), default="default")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose one GPU")
    cfg = apply_training_environment(load_training_config(args.config))
    if cfg.get("model", {}).get("backbone", "legacy") != "legacy":
        raise ValueError("This harness is intentionally limited to model.backbone=legacy")
    cfg["train"]["seed"] = args.seed
    if args.workers is not None: cfg["train"]["num_workers"] = args.workers
    if args.pin_memory: cfg["train"]["pin_memory"] = True
    if args.prefetch_factor is not None: cfg["train"]["prefetch_factor"] = args.prefetch_factor
    cfg.setdefault("runtime", {}).update({"compile_dense": args.compile_dense, "visualization_interval": 0})
    seed_training(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if args.tf32: torch.set_float32_matmul_precision("high")
    flow = FlowIMG(cfg).cuda().train()
    optimizer = make_optimizer(flow, args, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")
    data = datasets.dataloaders[cfg["data"]["dataloader"]](cfg).train_dataloader()
    iterator = iter(data)
    timer = CudaTimer()
    fixed_batch = None
    all_ms, losses, waits, h2d, stage_values = [], [], [], [], {}

    def next_batch():
        nonlocal iterator
        started = time.perf_counter()
        try: cpu = next(iterator)
        except StopIteration:
            iterator = iter(data); cpu = next(iterator)
        waits.append((time.perf_counter() - started) * 1e3)
        batch, elapsed = timer(lambda: move(cpu, flow.device, bool(cfg["train"].get("pin_memory", False))))
        h2d.append(elapsed); return batch

    if args.mode == "fixed": fixed_batch = next_batch()
    total = args.warmup + args.steps
    for index in range(total):
        batch = fixed_batch if fixed_batch is not None else next_batch()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); loss, stages = one_step(flow, batch, optimizer, timer, args.precision, scaler, args.breakdown); end.record(); end.synchronize()
        if index >= args.warmup:
            all_ms.append(start.elapsed_time(end)); losses.append(loss)
            for name, value in stages.items(): stage_values.setdefault(name, []).append(value)
    step_mean = float(statistics.mean(all_ms))
    wait_mean = float(statistics.mean(waits)) if waits and args.mode == "e2e" else 0.0
    h2d_mean = float(statistics.mean(h2d)) if h2d and args.mode == "e2e" else 0.0
    e2e_mean = step_mean + wait_mean + h2d_mean
    result = {
        "mode": args.mode, "precision": args.precision, "compile_dense": args.compile_dense,
        "tf32": args.tf32, "cudnn_benchmark": args.cudnn_benchmark, "optimizer": args.optimizer,
        "batch_size": cfg["train"]["batch_size"], "warmup_steps": args.warmup, "timed_steps": args.steps,
        "mean_ms": step_mean, "median_ms": float(statistics.median(all_ms)),
        "p90_ms": percentile(all_ms, 90), "p99_ms": percentile(all_ms, 99),
        "step_per_sec": 1000.0 / step_mean,
        "samples_per_sec": cfg["train"]["batch_size"] * 1000.0 / step_mean,
        "end_to_end_mean_ms": e2e_mean,
        "end_to_end_samples_per_sec": cfg["train"]["batch_size"] * 1000.0 / e2e_mean,
        "data_wait_mean_ms": wait_mean, "h2d_mean_ms": h2d_mean,
        "loss_mean": float(statistics.mean(losses)), "loss_last": float(losses[-1]),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "finite": bool(np.isfinite(losses).all()),
        "stage_mean_ms": {name: float(statistics.mean(values)) for name, values in stage_values.items()},
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

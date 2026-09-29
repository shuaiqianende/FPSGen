#!/usr/bin/env python3
"""Benchmark the Stage-1 BEVFlow training step without changing production code.

The benchmark deliberately keeps point coordinates, KeOps KNN, PointPillar,
BEV target rasterization and the loss in FP32.  AMP and optional Inductor are
restricted to ``BEVFlowTransNet.forward`` after its FP32 ``raw_pc`` condition
has been constructed.  It writes no checkpoint and is intended to be run with
one visible GPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

# Set before importing PyKeOps through FPSGen.  This keeps JIT artifacts out of
# the legacy environment/cache.
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("KEOPS_CACHE_FOLDER", str(ROOT / "env" / "keops_cache"))
Path(os.environ["KEOPS_CACHE_FOLDER"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from fpsgen.datasets.datasets import dataloaders
from fpsgen.models.gen_img import FlowIMG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), required=True)
    parser.add_argument("--compile-dense", action="store_true")
    parser.add_argument("--config", default="configs/train_bev.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--fp16-init-scale", type=float, default=65536.0)
    parser.add_argument(
        "--end-to-end", action="store_true",
        help="Fetch a fresh DataLoader batch each step and report input wait/H2D time.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override config DataLoader worker count for an end-to-end probe.",
    )
    parser.add_argument(
        "--pin-memory", action="store_true",
        help="Enable pinned DataLoader tensors and non-blocking H2D copies.",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def to_device(batch: dict, non_blocking: bool = False) -> dict:
    return {key: value.cuda(non_blocking=non_blocking) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient in {name}")


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=<one physical GPU>.")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the selected GPU.")
    if args.batch_size < 1 or args.warmup < 1 or args.steps < 1:
        raise ValueError("batch-size, warmup and steps must be positive.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["data"]["data_dir"] = args.dataset_root
    cfg["train"]["n_gpus"] = 1
    cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("num-workers must be non-negative")
        cfg["train"]["num_workers"] = args.num_workers
    elif not args.end_to_end:
        # Compute-only mode intentionally excludes input-pipeline work.
        cfg["train"]["num_workers"] = 0
    if args.pin_memory:
        cfg["train"]["pin_memory"] = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    flow = FlowIMG(cfg).cuda().train()
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg["train"]["lr"],
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    loader = dataloaders[cfg["data"]["dataloader"]](cfg).train_dataloader()
    loader_iter = iter(loader)

    def fetch_batch():
        nonlocal loader_iter
        wait_start = time.perf_counter()
        try:
            host_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            host_batch = next(loader_iter)
        input_wait_ms = (time.perf_counter() - wait_start) * 1e3
        torch.cuda.synchronize()
        h2d_start = time.perf_counter()
        device_batch = to_device(host_batch, non_blocking=args.pin_memory)
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - h2d_start) * 1e3
        if device_batch["pcd_full"].shape[:2] != (args.batch_size, cfg["data"]["num_points"]):
            raise RuntimeError(
                f"unexpected full-cloud batch shape: {tuple(device_batch['pcd_full'].shape)}"
            )
        return device_batch, input_wait_ms, h2d_ms

    batch, _, _ = fetch_batch()

    eager_forward: Callable = flow.model.forward
    compile_elapsed_s = 0.0
    if args.compile_dense:
        # ``torch._dynamo`` is not populated as an attribute until explicitly
        # imported in the PyTorch 2.0 package used by this isolated probe.
        import torch._dynamo as dynamo
        dynamo.reset()
        compile_start = time.perf_counter()
        dense_forward: Callable = torch.compile(eager_forward, backend="inductor")
        compile_elapsed_s = time.perf_counter() - compile_start
    else:
        dense_forward = eager_forward

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.precision == "fp16", init_scale=args.fp16_init_scale
    )
    overflow_steps = 0
    timed_overflow_steps = 0

    def one_step(step_index: int):
        nonlocal overflow_steps, timed_overflow_steps
        nonlocal batch
        input_wait_ms = 0.0
        h2d_ms = 0.0
        if args.end_to_end:
            batch, input_wait_ms, h2d_ms = fetch_batch()
        # Keep all random flow-path and condition-mask choices identical across
        # precision modes. Geometry processing stays outside autocast.
        random.seed(args.seed + step_index)
        np.random.seed(args.seed + step_index)
        torch.manual_seed(args.seed + step_index)
        torch.cuda.manual_seed_all(args.seed + step_index)
        optimizer.zero_grad(set_to_none=True)

        gt_points = batch["pcd_full"].float()
        B = gt_points.shape[0]
        state_indices = torch.randint(0, 8, (B,), device="cuda")
        drop_lidar = ~((state_indices & 4) > 0)
        drop_vehicle = ~((state_indices & 2) > 0)
        drop_road = ~((state_indices & 1) > 0)
        layout = flow.processor.get_layout_bev(gt_points, batch["full_label"])
        layout = layout * 2.0 - 1.0
        layout[:, 0] = torch.where(drop_vehicle[:, None, None], torch.zeros_like(layout[:, 0]), layout[:, 0])
        layout[:, 1] = torch.where(drop_road[:, None, None], torch.zeros_like(layout[:, 1]), layout[:, 1])

        # FP32 point geometry and PyKeOps/Pillar path.
        raw_pc = flow.model.get_raw_pc_bev(batch["pcd_part"].float())
        raw_pc = torch.where(drop_lidar[:, None, None, None], torch.zeros_like(raw_pc), raw_pc)
        x1 = flow.processor.points_to_bev_target(gt_points)
        t = torch.rand((B,), device="cuda")
        t_expand = t[:, None, None, None]
        x0 = torch.randn_like(x1)
        xt = (1.0 - t_expand) * x0 + t_expand * x1
        ut = x1 - x0

        if amp_dtype is None:
            out = dense_forward(xt, t, raw_pc, layout)
        else:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = dense_forward(xt, t, raw_pc, layout)
        # The velocity target and MSE reduction remain FP32 for the numerical
        # comparison. This is the exact boundary proposed for Stage-1 AMP.
        out = out.float()
        channel_weight = torch.tensor([1.0, 2.0, 1.0], device="cuda")[None, :, None, None]
        loss = (F.mse_loss(out, ut, reduction="none") * channel_weight).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step_index}: {loss}")
        if scaler.is_enabled():
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # A transient overflow is expected while GradScaler calibrates its
            # initial scale.  Let the scaler skip that update and lower its
            # scale; reject it only if it persists into timed measurements.
            gradients_finite = True
            try:
                assert_finite_gradients(flow)
            except FloatingPointError:
                gradients_finite = False
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < scale_before
            if overflow:
                overflow_steps += 1
                if step_index >= args.warmup:
                    timed_overflow_steps += 1
            elif not gradients_finite:
                raise FloatingPointError(
                    "non-finite FP16 gradient was not handled by GradScaler"
                )
        else:
            loss.backward()
            assert_finite_gradients(flow)
            optimizer.step()
        return float(loss.detach()), input_wait_ms, h2d_ms

    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup):
        value, _, _ = one_step(index)
        print(f"warmup {index + 1}/{args.warmup}: loss={value:.8f}", flush=True)
    torch.cuda.synchronize()

    timings, losses, input_waits, h2d_times = [], [], [], []
    for index in range(args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, input_wait_ms, h2d_ms = one_step(args.warmup + index)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        timings.append(elapsed_ms)
        losses.append(loss)
        input_waits.append(input_wait_ms)
        h2d_times.append(h2d_ms)
        print(f"timed {index + 1}/{args.steps}: loss={loss:.8f} step_ms={elapsed_ms:.1f}", flush=True)

    run_name = args.run_name or f"{args.precision}_{'compile' if args.compile_dense else 'eager'}_b{args.batch_size}"
    result = {
        "run_name": run_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "precision": args.precision,
        "compile_dense": args.compile_dense,
        "compile_wrapper_creation_s": compile_elapsed_s,
        "batch_size": args.batch_size,
        "full_points": int(batch["pcd_full"].shape[1]),
        "partial_points": int(batch["pcd_part"].shape[1]),
        "end_to_end": args.end_to_end,
        "num_workers": cfg["train"]["num_workers"],
        "pin_memory": cfg["train"].get("pin_memory", False),
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.quantile(timings, .95)),
        "input_wait_mean_ms": float(np.mean(input_waits)),
        "input_wait_p95_ms": float(np.quantile(input_waits, .95)),
        "h2d_mean_ms": float(np.mean(h2d_times)),
        "h2d_p95_ms": float(np.quantile(h2d_times, .95)),
        "samples_per_sec": float(args.batch_size / (np.mean(timings) / 1e3)),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 2**20),
        "loss_mean": float(np.mean(losses)),
        "loss_last": float(losses[-1]),
        "finite": timed_overflow_steps == 0,
        "fp16_overflow_steps": overflow_steps if scaler.is_enabled() else 0,
        "fp16_timed_overflow_steps": timed_overflow_steps if scaler.is_enabled() else 0,
        "fp16_final_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
        "amp_scope": "dense BEVFlow forward only; KNN/pillar/targets/loss FP32",
    }
    result_path = ROOT / "env" / "results" / f"bevflow_speed_{run_name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

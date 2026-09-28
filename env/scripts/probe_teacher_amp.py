"""Real FPSGen Teacher AMP compatibility and timing probe.

This script intentionally lives outside the training implementation. It keeps
P0 generation, TensorField/coordinate construction, endpoints and DCD in
FP32, and restricts autocast to the ME backbone. It never writes checkpoints.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml

from fpsgen.datasets.datasets import dataloaders
from fpsgen.models.gen_teacher import DiffusionPoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), required=True)
    parser.add_argument("--config", default="configs/research_v2/train_teacher_dcd_a1_l1_5ep.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--run-name", default=None, help="Result/log suffix; defaults to precision.")
    parser.add_argument(
        "--empty-cache-mode",
        choices=("current", "disabled"),
        default="current",
        help="Benchmark only: retain or locally neutralize hot-path empty_cache calls.",
    )
    return parser.parse_args()


def to_device(batch: dict) -> dict:
    return {key: value.cuda(non_blocking=False) if torch.is_tensor(value) else value for key, value in batch.items()}


def install_forward_precision_boundary(model: DiffusionPoints, precision: str) -> None:
    """Replace only the probe's forward method; production code is untouched."""
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)

    def probe_forward(self: DiffusionPoints, x_full, x_full_sparse, x_part, t):
        if amp_dtype is None:
            part_feat = self.partial_enc(x_part)
            out = self.model(x_full, x_full_sparse, part_feat)
        else:
            # x_full/x_part were already built as FP32 TensorFields. The
            # residual returns to FP32 before endpoint and DCD arithmetic.
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                part_feat = self.partial_enc(x_part)
                out = self.model(x_full, x_full_sparse, part_feat)
        return out.float().reshape(t.shape[0], -1, 3)

    model.forward = types.MethodType(probe_forward, model)


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run this probe with CUDA_VISIBLE_DEVICES=1; Python must see exactly one GPU.")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on the selected GPU.")
    if args.empty_cache_mode == "disabled":
        # The formal model remains unchanged. This local probe measures the
        # isolated effect of its repeated cache-clearing calls.
        torch.cuda.empty_cache = lambda: None

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["data"]["data_dir"] = args.dataset_root
    cfg["train"]["num_workers"] = 0  # Probe compute, not worker throughput.
    if cfg["train"]["batch_size"] != 2 or cfg["data"]["num_points"] != 180000:
        raise ValueError("Teacher speed parity probe requires batch_size=2 and num_points=180000.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = DiffusionPoints(cfg).cuda().train()
    install_forward_precision_boundary(model, args.precision)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loader = dataloaders[cfg["data"]["dataloader"]](cfg).train_dataloader()
    batch = to_device(next(iter(loader)))

    def one_step(step: int) -> float:
        # Make P0 sampling identical across precision runs. Coordinates and
        # DCD remain FP32; autocast is limited to the sparse backbone above.
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        optimizer.zero_grad(set_to_none=True)
        loss = model._shared_step(batch, metric_prefix="probe")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        return float(loss.detach())

    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup):
        warmup_loss = one_step(index)
        print(f"warmup {index + 1}/{args.warmup}: loss={warmup_loss:.8f}", flush=True)
    torch.cuda.synchronize()
    times_ms, losses = [], []
    for index in range(args.steps):
        start = time.perf_counter()
        losses.append(one_step(args.warmup + index))
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)
        print(
            f"timed {index + 1}/{args.steps}: loss={losses[-1]:.8f} "
            f"step_ms={times_ms[-1]:.1f}",
            flush=True,
        )

    result = {
        "run_name": args.run_name or args.precision,
        "empty_cache_mode": args.empty_cache_mode,
        "precision": args.precision,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": cfg["train"]["batch_size"],
        "points": cfg["data"]["num_points"],
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "mean_ms": sum(times_ms) / len(times_ms),
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
        "p95_ms": sorted(times_ms)[max(0, int(0.95 * len(times_ms)) - 1)],
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "finite": True,
    }
    Path("env/results").mkdir(parents=True, exist_ok=True)
    result_path = Path("env/results") / f"teacher_amp_{args.run_name or args.precision}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure Legacy BEVFlow Dataset+collate latency without model execution."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from fpsgen.datasets import datasets
from fpsgen.utils.training_runtime import apply_training_environment, load_training_config, seed_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    cfg = apply_training_environment(load_training_config(args.config))
    cfg["train"].update({"num_workers": args.workers, "prefetch_factor": args.prefetch_factor})
    seed_training(int(cfg["train"].get("seed", 42)))
    loader = datasets.dataloaders[cfg["data"]["dataloader"]](cfg).train_dataloader()
    iterator = iter(loader); values = []
    for index in range(args.warmup + args.batches):
        start = time.perf_counter()
        try: next(iterator)
        except StopIteration: iterator = iter(loader); next(iterator)
        elapsed = (time.perf_counter() - start) * 1e3
        if index >= args.warmup: values.append(elapsed)
    result = {"workers": args.workers, "prefetch_factor": args.prefetch_factor,
              "batch_size": cfg["train"]["batch_size"], "batches": args.batches,
              "mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)),
              "p90_ms": float(np.percentile(values, 90)), "p99_ms": float(np.percentile(values, 99)),
              "samples_per_sec": cfg["train"]["batch_size"] * 1000.0 / float(np.mean(values))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result); writer.writeheader(); writer.writerow(result)
    print(result)


if __name__ == "__main__": main()

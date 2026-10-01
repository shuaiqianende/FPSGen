#!/usr/bin/env python3
"""Profile a real Legacy BEVFlow optimizer step without compiling geometry.

This is a deliberately thin preset around :mod:`benchmark_bevflow_training`:
it leaves the Dataset, KeOps KNN and BEV rasterization eager, measures their
CUDA-event time separately, and runs the dense ``BEVFlowTransNet`` unchanged.
It is intended for deciding whether a cache is justified, not for quality
training.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    command = [
        sys.executable, "scripts/benchmark_bevflow_training.py",
        "--config", str(args.config), "--output", str(args.output),
        "--mode", "fixed", "--precision", "fp32", "--warmup", str(args.warmup),
        "--steps", str(args.steps), "--breakdown",
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

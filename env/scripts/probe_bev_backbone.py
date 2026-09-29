#!/usr/bin/env python3
"""Prepared-only dense BEV backbone probe; it never calls torch.compile itself."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("dic_s", "pixelu_s"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--compile", action="store_true",
                        help="Reserved for the later Inductor phase; deliberately fails now.")
    args = parser.parse_args()
    if args.compile:
        raise RuntimeError("Inductor is intentionally disabled during backbone preparation.")
    config = Path("configs/research_v2") / f"train_bev_{args.backbone.replace('_', '_')}_gt_possion.yaml"
    command = ["python", "scripts/inspect_bev_backbone.py", "--config", str(config)]
    print("Prepared probe only; no CUDA or compile execution.")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

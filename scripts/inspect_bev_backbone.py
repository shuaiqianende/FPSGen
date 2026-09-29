#!/usr/bin/env python3
"""CPU-only topology and parameter inspection for Stage-1 BEV backbones."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from fpsgen.models.bev_backbones import build_bev_backbone


def count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    model = build_bev_backbone(cfg)
    name = cfg.get("model", {}).get("backbone", "legacy")
    result = {
        "backbone": name,
        "generator_core_params": count(getattr(model, "core", model)),
        "condition_adapter_params": count(getattr(model, "condition_encoder", None)) if hasattr(model, "condition_encoder") else 0,
        "shared_lidar_encoder_params": count(getattr(model, "pc_encoder", None)) if hasattr(model, "pc_encoder") else 0,
        "total_params": count(model),
    }
    if name == "dic_s":
        result.update({"hidden": model.core.hidden_size, "depth": list(model.core.depth),
                       "mult_channels": list(model.core.mult_channels), "skip_stride": model.core.skip_stride})
    if name == "pixelu_s":
        result.update({"patch": model.core.patch_size, "depth": list(model.core.depth),
                       "hidden": model.core.hidden_size, "heads": model.core.num_heads,
                       "bottleneck": model.core.bottleneck_dim})
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

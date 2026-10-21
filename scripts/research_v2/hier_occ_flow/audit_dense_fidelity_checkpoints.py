#!/usr/bin/env python3
"""CPU-only static audit of OMD V1 checkpoints and registered PR/fixed-K curves."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_stats(value):
    x = value.detach().float().cpu()
    return {"shape": list(x.shape), "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std(unbiased=False))}


def checkpoint_summary(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["ema"]
    if state is None:
        raise ValueError(f"EMA is missing in {path}")
    context_scales = {}
    gates = {}
    for name, value in state.items():
        if name.endswith("context_scale"):
            context_scales[name] = tensor_stats(value)
        elif name.endswith("gate.bias"):
            gates[name] = tensor_stats(value)
    return {"path": str(path), "sha256": sha256(path),
            "successful_updates": int(saved["successful_updates"]),
            "successful_frame_exposures": int(saved["sampler"]["successful_frame_exposures"]),
            "context_scale": context_scales, "gate_bias": gates}


def curve_summary(path):
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE" or value.get("frames") != 900:
        raise ValueError(f"Expected completed 900-frame curve artifact: {path}")
    summaries = value["threshold_summaries"]
    key = min(summaries, key=lambda x: abs(float(x) - 0.5))
    threshold = summaries[key]
    fixed = value["fixed_budget_summaries"]
    return {"path": str(path), "sha256": sha256(path),
            "development_manifest_sha256": value["development_manifest_sha256"],
            "frames": value["frames"], "threshold": float(key),
            "threshold_macro": threshold["macro"], "threshold_micro": threshold["micro"],
            "fixed_budget": fixed}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--r0", type=Path, required=True)
    p.add_argument("--r1", type=Path, required=True)
    p.add_argument("--r2", type=Path, required=True)
    p.add_argument("--curve-r0", type=Path, required=True)
    p.add_argument("--curve-r1", type=Path, required=True)
    p.add_argument("--curve-r2", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = {
        "scope": "static_checkpoint_and_saved_metric_audit",
        "runtime_forward_activation_analysis": "NOT_RUN_NO_CUDA_DEVICE",
        "checkpoints": {"R0": checkpoint_summary(a.r0),
                        "R1": checkpoint_summary(a.r1),
                        "R2": checkpoint_summary(a.r2)},
        "curves": {"R0": curve_summary(a.curve_r0),
                   "R1": curve_summary(a.curve_r1),
                   "R2": curve_summary(a.curve_r2)},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired B20 bootstrap and B100 condition-use statistics for Phase A."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ("u0_generic", "s0_synflow", "c0_cracksegflow", "p0_pixeldit_generic", "p1_pixelcontrol")
PAIRS = (("s0_synflow", "u0_generic"), ("s0_synflow", "c0_cracksegflow"),
         ("c0_cracksegflow", "u0_generic"), ("p1_pixelcontrol", "p0_pixeldit_generic"))
METRICS = ("occupancy_iou", "occupancy_f1", "completion_f1", "height_mae_gtocc_m", "density_mass_tv")


def bootstrap_delta(a, b, rng, count):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    indices = rng.integers(0, len(delta), size=(count, len(delta)))
    means = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95": [float(x) for x in np.quantile(means, (.025, .975))],
            "fraction_a_gt_b": float((means > 0).mean()), "n_frames": int(len(delta))}


def rows(path):
    return {row["frame"]: row for row in csv.DictReader(path.open())}


def prefix(metric): return f"bevflow_{metric}_mean"


def condition_records(path, mode, max_time):
    values = {}
    for row in csv.DictReader(path.open()):
        if row["mode"] == mode and float(row["time"]) < max_time:
            values.setdefault(int(row["frame_index"]), []).append(row)
    return {key: {field: float(np.mean([float(row[field]) for row in group])) for field in ("delta_shuffle", "g_shuffle")}
            for key, group in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/research_v2/spatial_control"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research_v2/spatial_control/spatial_control_paired_bootstrap.json"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(); rng = np.random.default_rng(args.seed)
    result = {"bootstrap_resamples": args.resamples, "bootstrap_seed": args.seed, "generation": {}, "cfg_amplification": {}, "condition_usage": {}}
    per_frame = {}
    for model in MODELS:
        per_frame[model] = {}
        for cfg in (1, 2):
            candidates = list(args.root.glob(f"fpsgen_spatial_{model}*_lidar_b20_cfg{cfg}/per_frame.csv"))
            if candidates: per_frame[model][cfg] = rows(candidates[0])
    for cfg in (1, 2):
        section = result["generation"][f"cfg{cfg}"] = {}
        for a, b in PAIRS:
            if cfg not in per_frame[a] or cfg not in per_frame[b]: continue
            frames = sorted(set(per_frame[a][cfg]) & set(per_frame[b][cfg]))
            section[f"{a}-{b}"] = {metric: bootstrap_delta(
                [float(per_frame[a][cfg][frame][prefix(metric)]) for frame in frames],
                [float(per_frame[b][cfg][frame][prefix(metric)]) for frame in frames], rng, args.resamples)
                for metric in METRICS}
        for model, by_cfg in per_frame.items():
            if 1 in by_cfg and 2 in by_cfg:
                frames = sorted(set(by_cfg[1]) & set(by_cfg[2]))
                result["cfg_amplification"][model] = {metric: bootstrap_delta(
                    [float(by_cfg[2][frame][prefix(metric)]) for frame in frames],
                    [float(by_cfg[1][frame][prefix(metric)]) for frame in frames], rng, args.resamples)
                    for metric in METRICS}
    records = {model: next(iter(args.root.glob(f"fpsgen_spatial_{model}*_condition_usage_b100_records.csv")), None) for model in MODELS}
    if all(records.values()):
        for mode in ("100", "010", "001", "111"):
            for cutoff, label in ((.2, "t_lt_0_2"), (.4, "t_lt_0_4")):
                by_model = {model: condition_records(path, mode, cutoff) for model, path in records.items()}
                result["condition_usage"].setdefault(mode, {})[label] = {}
                for a, b in PAIRS:
                    frames = sorted(set(by_model[a]) & set(by_model[b]))
                    result["condition_usage"][mode][label][f"{a}-{b}"] = {
                        metric: bootstrap_delta([by_model[a][frame][metric] for frame in frames],
                                                [by_model[b][frame][metric] for frame in frames], rng, args.resamples)
                        for metric in ("delta_shuffle", "g_shuffle")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

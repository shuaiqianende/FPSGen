#!/usr/bin/env python3
"""Create two-seed N0/N1 short-run summary once queues have completed."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path("outputs/research_v2/ncsnpp_spade")
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_RESAMPLES = 10_000

def summary_losses(path):
    rows = list(csv.DictReader(path.open()))
    values = {}
    for row in rows:
        if row["kind"] in {"trailing_200", "trailing_500"}:
            values[(row["kind"], row["global_step"])] = float(row["mean_loss"])
    return values


def paired_bootstrap(values_a, values_b, generator):
    """Frame-paired bootstrap for N1-N0; no false independent-frame pool."""
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    if not len(diff):
        return None
    ids = generator.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    means = diff[ids].mean(axis=1)
    return {"mean": float(diff.mean()), "ci95": [float(np.quantile(means, .025)),
            float(np.quantile(means, .975))], "fraction_n1_gt_n0": float((means > 0).mean()),
            "n_frames": int(len(diff))}


def frame_metrics(path, names):
    rows = {row["frame"]: row for row in csv.DictReader(path.open())}
    return {name: np.asarray([float(row[f"bevflow_{name}_mean"])
                              for _, row in sorted(rows.items())]) for name in names}


def usage_per_frame(path, cutoff):
    values = {}
    for row in csv.DictReader(path.open()):
        if row["mode"] != "100" or float(row["time"]) >= cutoff:
            continue
        values.setdefault(int(row["frame_index"]), []).append(float(row["g_shuffle"]))
    return np.asarray([np.mean(item) for _, item in sorted(values.items())])


def paired_statistics():
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    result = {"bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
              "generation": {}, "condition_usage": {}}
    metrics = ("occupancy_iou", "occupancy_f1", "completion_f1", "height_mae_gtocc_m", "density_mass_tv")
    for seed in (42, 123):
        for cfg in (1, 2):
            n0 = next(iter(ROOT.glob(f"fpsgen_n0_*seed{seed}_*_lidar_b20_cfg{cfg}/per_frame.csv")), None)
            n1 = next(iter(ROOT.glob(f"fpsgen_n1_*seed{seed}_*_lidar_b20_cfg{cfg}/per_frame.csv")), None)
            if n0 and n1:
                result["generation"].setdefault(f"seed{seed}_cfg{cfg}", {})
                a, b = frame_metrics(n1, metrics), frame_metrics(n0, metrics)
                result["generation"][f"seed{seed}_cfg{cfg}"] = {
                    metric: paired_bootstrap(a[metric], b[metric], generator) for metric in metrics}
        n0 = next(iter(ROOT.glob(f"fpsgen_n0_*seed{seed}_*_condition_usage_b100_records.csv")), None)
        n1 = next(iter(ROOT.glob(f"fpsgen_n1_*seed{seed}_*_condition_usage_b100_records.csv")), None)
        if n0 and n1:
            result["condition_usage"][f"seed{seed}"] = {
                label: paired_bootstrap(usage_per_frame(n1, cutoff), usage_per_frame(n0, cutoff), generator)
                for label, cutoff in (("gshuffle100_t_lt_0_2", .2), ("gshuffle100_t_lt_0_4", .4))}
    return result

def main():
    summary = []; by_model = {"n0": [], "n1": []}
    for model in ("n0", "n1"):
        for seed in (42, 123):
            run = next(iter(ROOT.glob(f"fpsgen_{model}_*seed{seed}_*_throughput_loss_summary.csv")), None)
            usage = next(iter(ROOT.glob(f"fpsgen_{model}_*seed{seed}_*_condition_usage_b100.json")), None)
            if not run or not usage: continue
            losses = summary_losses(run); condition = json.loads(usage.read_text())["modes"]
            row = {"model": model, "seed": seed,
                   # At step 1500 this is exactly optimizer steps 1000--1499.
                   "loss_1000_1499": losses.get(("trailing_500", "1500")),
                   "loss_final200": losses.get(("trailing_200", "1500")),
                   "gshuffle100_t_lt_0_2": condition["100"]["t_lt_0_2"]["full"]["g_shuffle"],
                   "gshuffle100_t_lt_0_4": condition["100"]["t_lt_0_4"]["full"]["g_shuffle"]}
            for cfg in (1, 2):
                generation = next(iter(ROOT.glob(f"fpsgen_{model}_*seed{seed}_*_lidar_b20_cfg{cfg}/summary.json")), None)
                if generation:
                    metrics = json.loads(generation.read_text())["bevflow_lidar_only"]
                    row[f"iou_cfg{cfg}"] = metrics["occupancy_iou"]["mean"]
                    row[f"completion_cfg{cfg}"] = metrics["completion_f1"]["mean"]
            summary.append(row); by_model[model].append(row)
    ROOT.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in summary for key in row})
    with (ROOT / "final_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(summary)
    aggregate = {model: {key: {"mean": float(np.mean([row[key] for row in rows if key in row])), "std": float(np.std([row[key] for row in rows if key in row]))}
                          for key in keys if key not in {"model", "seed"} and any(key in row for row in rows)}
                 for model, rows in by_model.items() if rows}
    (ROOT / "seed_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    paired = paired_statistics()
    (ROOT / "paired_bootstrap.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(json.dumps({"aggregate": aggregate, "paired": paired}, indent=2))

if __name__ == "__main__": main()

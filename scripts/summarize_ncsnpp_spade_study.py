#!/usr/bin/env python3
"""Create two-seed N0/N1 short-run summary once queues have completed."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path("outputs/research_v2/ncsnpp_spade")

def summary_losses(path):
    rows = list(csv.DictReader(path.open()))
    values = {}
    for row in rows:
        if row["kind"] in {"trailing_200", "trailing_500"}:
            values[(row["kind"], row["global_step"])] = float(row["mean_loss"])
    return values

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
    print(json.dumps(aggregate, indent=2))

if __name__ == "__main__": main()

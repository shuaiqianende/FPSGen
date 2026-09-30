#!/usr/bin/env python3
"""Create the strict within-backbone Phase-1 condition-injection ranking.

This consumes only completed checkpoints plus their condition-usage and B20
screen-evaluation artifacts.  It refuses to rank incomplete data by default:
that guard prevents a good training-loss curve from being mistaken for a
condition-aware winner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "condition_campaign"
BACKBONES = ("hdit", "dip", "ncsnpp")
VARIANTS = ("hybrid_shared", "spatial", "global", "separate", "native")
INJECTION = {
    "hybrid_shared": "Hybrid-Shared",
    "spatial": "Spatial-Only",
    "global": "Global-Only",
    "separate": "Hybrid-Separate",
    "native": "Native-Hybrid",
}
C0_RUN_IDS = {
    "hdit": "fpsgen_bev_hdit_s_5ep_b8_gpu2",
    "dip": "fpsgen_bev_dip_s_5ep_b8_gpu3",
    "ncsnpp": "fpsgen_bev_ncsnpp_s_5ep_b8_gpu1",
}
LOWER_IS_BETTER = ("final_500_sampled_loss", "density_mass_tv", "height_mae_gtocc_m")
HIGHER_IS_BETTER = ("gshuffle_100", "gshuffle_layout_mean", "occupancy_iou", "completion_f1")
WEIGHTS = {
    "final_500_sampled_loss": 1,
    "gshuffle_100": 2,
    "gshuffle_layout_mean": 1,
    "density_mass_tv": 1,
    "height_mae_gtocc_m": 1,
    "occupancy_iou": 2,
    "completion_f1": 2,
}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def experiment_for(backbone: str, variant: str) -> Path:
    run_id = C0_RUN_IDS[backbone] if variant == "hybrid_shared" else f"bev_cond_{backbone}_{variant}_s42_5ep"
    return ROOT / "experiments" / run_id


def profile_for(backbone: str, variant: str) -> Path:
    if variant == "hybrid_shared":
        return ROOT / "outputs" / "research_v2" / "bev_train" / "profiles" / f"{C0_RUN_IDS[backbone]}_throughput.csv"
    return OUTPUT / "profiles" / f"bev_cond_{backbone}_{variant}_s42_5ep.csv"


def final_500_sampled_loss(experiment: Path) -> float | None:
    """Use the last full five-scalar 500-step bin, never a partial tail."""
    values: dict[int, float] = {}
    for event in experiment.glob("lightning_logs/version_*/events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
        accumulator.Reload()
        for scalar in accumulator.Scalars("train/loss_mse"):
            values[scalar.step] = scalar.value
    bins: dict[int, list[float]] = defaultdict(list)
    for step, value in values.items():
        bins[(step // 500) * 500].append(value)
    complete = [(start, samples) for start, samples in bins.items() if len(samples) >= 5]
    if not complete:
        return None
    _, samples = max(complete, key=lambda item: item[0])
    return sum(samples) / len(samples)


def profile_status(path: Path) -> dict[str, float | bool | None]:
    if not path.exists():
        return {"finite_profile": None, "last_step": None}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"finite_profile": False, "last_step": None}
    finite = all(math.isfinite(float(row["loss"])) for row in rows if row.get("loss"))
    return {"finite_profile": finite, "last_step": int(rows[-1]["global_step"])}


def screen_metrics(path: Path) -> dict[str, float | None]:
    summary = load_json(path / "summary.json")
    values = summary.get("bevflow_lidar_only", {}) if summary else {}
    return {
        "density_mass_tv": values.get("density_mass_tv", {}).get("mean"),
        "height_mae_gtocc_m": values.get("height_mae_gtocc_m", {}).get("mean"),
        "occupancy_iou": values.get("occupancy_iou", {}).get("mean"),
        "completion_f1": values.get("completion_f1", {}).get("mean"),
    }


def usage_metrics(path: Path) -> dict[str, float | None]:
    usage = load_json(path)
    modes = usage.get("modes", {}) if usage else {}
    def gain(mode: str) -> float | None:
        return modes.get(mode, {}).get("g_shuffle")
    layouts = [gain(mode) for mode in ("010", "001", "111")]
    return {
        "gshuffle_100": gain("100"),
        "gshuffle_010": gain("010"),
        "gshuffle_001": gain("001"),
        "gshuffle_111": gain("111"),
        "gshuffle_layout_mean": sum(value for value in layouts if value is not None) / len(layouts) if all(value is not None for value in layouts) else None,
    }


def record(backbone: str, variant: str) -> dict[str, Any]:
    experiment = experiment_for(backbone, variant)
    run_id = experiment.name
    complete = len(list(experiment.glob("lightning_logs/version_*/checkpoints/*epoch=04.ckpt"))) == 1
    record = {
        "backbone": backbone,
        "variant": variant,
        "injection": INJECTION[variant],
        "run_id": run_id,
        "complete": complete,
        "final_500_sampled_loss": final_500_sampled_loss(experiment) if complete else None,
        **profile_status(profile_for(backbone, variant)),
        **usage_metrics(OUTPUT / "condition_usage" / f"{backbone}_{variant}.json"),
        **screen_metrics(OUTPUT / "screen_eval" / f"{backbone}_{variant}"),
    }
    record["condition_insensitive"] = bool(
        record["gshuffle_100"] is not None
        and record["gshuffle_111"] is not None
        and abs(record["gshuffle_100"]) <= 0.01
        and abs(record["gshuffle_111"]) <= 0.01
    )
    return record


def rank_group(records: list[dict[str, Any]]) -> None:
    for metric in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER):
        reverse = metric in HIGHER_IS_BETTER
        ordered = sorted(records, key=lambda item: item[metric], reverse=reverse)
        for position, item in enumerate(ordered, start=1):
            item.setdefault("metric_ranks", {})[metric] = position
    for item in records:
        item["weighted_rank_score"] = sum(
            WEIGHTS[metric] * item["metric_ranks"][metric]
            for metric in WEIGHTS
        )
    records.sort(key=lambda item: (item["weighted_rank_score"], item["variant"]))
    for position, item in enumerate(records, start=1):
        item["overall_rank"] = position


def require_complete(records: list[dict[str, Any]]) -> None:
    required = (*LOWER_IS_BETTER, *HIGHER_IS_BETTER)
    missing = [
        f"{item['backbone']}/{item['variant']}: " + ", ".join(
            key for key in required if item.get(key) is None
        )
        for item in records
        if not item["complete"] or any(item.get(key) is None for key in required)
    ]
    if missing:
        raise RuntimeError("Phase-1 ranking requires every C0–C4 checkpoint and metric:\n" + "\n".join(missing))


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = ["backbone", "variant", "injection", "complete", "final_500_sampled_loss", "gshuffle_100", "gshuffle_010", "gshuffle_001", "gshuffle_111", "gshuffle_layout_mean", "density_mass_tv", "height_mae_gtocc_m", "occupancy_iou", "completion_f1", "condition_insensitive", "weighted_rank_score", "overall_rank"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: item.get(key) for key in columns} for item in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="write a live artifact without ranking")
    args = parser.parse_args()
    records = [record(backbone, variant) for backbone in BACKBONES for variant in VARIANTS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.allow_incomplete:
        require_complete(records)
        by_backbone = {backbone: [item for item in records if item["backbone"] == backbone] for backbone in BACKBONES}
        for group in by_backbone.values():
            rank_group(group)
        ranked = [item for backbone in BACKBONES for item in by_backbone[backbone]]
        write_csv(OUTPUT / "phase1_summary.csv", ranked)
        result = {
            "ranking_contract": {"weights": WEIGHTS, "condition_insensitive_threshold": 0.01},
            "records": ranked,
            "top2": {backbone: [item["variant"] for item in by_backbone[backbone][:2]] for backbone in BACKBONES},
        }
        (OUTPUT / "phase1_ranking.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["top2"], sort_keys=True))
        return
    (OUTPUT / "phase1_live.json").write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print("Wrote live incomplete Phase-1 artifact")


if __name__ == "__main__":
    main()

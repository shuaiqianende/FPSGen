#!/usr/bin/env python3
"""Summarize the reproducible Stage-1 BEV condition-injection campaign.

The campaign writes TensorBoard scalars, a compact synchronized-throughput
CSV, gradient probes, condition-use JSON and screen-evaluation directories.
This tool joins those immutable artifacts into both machine-readable JSON and
an auditable Markdown table.  It intentionally reports missing artifacts as
``—``; an incomplete run must never look like a successful result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "condition_campaign"
BACKBONES = ("hdit", "dip", "ncsnpp")
VARIANTS = ("spatial", "global", "separate", "native")


def percentile(values: Iterable[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile, or ``None`` when absent."""
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def load_scalars(experiment: Path) -> tuple[list[dict[str, float]], str | None]:
    """Read loss/epoch scalar samples from all event files in one experiment."""
    events = sorted(experiment.glob("lightning_logs/version_*/events.out.tfevents.*"))
    samples: dict[int, dict[str, float]] = defaultdict(dict)
    try:
        for event in events:
            accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
            accumulator.Reload()
            for scalar in accumulator.Scalars("train/loss_mse"):
                samples[scalar.step]["loss"] = scalar.value
            for scalar in accumulator.Scalars("epoch"):
                samples[scalar.step]["epoch"] = scalar.value
    except Exception as error:  # pragma: no cover - protects live/truncated files.
        return [], f"TensorBoard parse failure: {error}"
    return [
        {"step": float(step), **value}
        for step, value in sorted(samples.items())
        if "loss" in value
    ], None


def loss_summary(samples: list[dict[str, float]]) -> dict[str, Any]:
    """Compute sampled 500-step and per-epoch loss means without interpolation."""
    windows: dict[str, list[float]] = defaultdict(list)
    epochs: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        windows[f"{int(sample['step']) // 500 * 500:05d}"].append(sample["loss"])
        if "epoch" in sample:
            epochs[str(int(sample["epoch"]))].append(sample["loss"])
    return {
        "scalar_samples": len(samples),
        "last_step": int(samples[-1]["step"]) if samples else None,
        "loss_500_step_mean": {
            key: sum(values) / len(values) for key, values in sorted(windows.items())
        },
        "sampled_epoch_mean": {
            key: sum(values) / len(values) for key, values in sorted(epochs.items(), key=lambda item: int(item[0]))
        },
    }


def load_profile(run_id: str) -> dict[str, float | None]:
    path = OUTPUT / "profiles" / f"{run_id}.csv"
    if not path.exists():
        return {key: None for key in ("median_step_seconds", "p95_step_seconds", "median_data_gap_seconds", "p95_data_gap_seconds", "max_gpu_allocated_mb", "max_gpu_reserved_mb")}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    step, gap = values("optimizer_step_seconds"), values("data_ready_gap_seconds")
    return {
        "median_step_seconds": percentile(step, 0.5),
        "p95_step_seconds": percentile(step, 0.95),
        "median_data_gap_seconds": percentile(gap, 0.5),
        "p95_data_gap_seconds": percentile(gap, 0.95),
        "max_gpu_allocated_mb": max(values("gpu_allocated_mb"), default=None),
        "max_gpu_reserved_mb": max(values("gpu_reserved_mb"), default=None),
    }


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except json.JSONDecodeError:
        return None


def run_record(backbone: str, variant: str) -> dict[str, Any]:
    run_id = f"bev_cond_{backbone}_{variant}_s42_5ep"
    experiment = ROOT / "experiments" / run_id
    scalars, parse_error = load_scalars(experiment)
    checkpoints = sorted(experiment.glob("lightning_logs/version_*/checkpoints/*epoch=04.ckpt"))
    gradient = load_json(OUTPUT / "smoke" / f"{backbone}_{variant}_gradients.json")
    usage = load_json(OUTPUT / "condition_usage" / f"{backbone}_{variant}.json")
    parameter_count = load_json(
        OUTPUT / "parameter_counts" / f"train_bev_{backbone}_cond_{variant}_s42_5ep.json"
    )
    return {
        "run_id": run_id,
        "backbone": backbone,
        "variant": variant,
        "complete": len(checkpoints) == 1,
        "checkpoint": str(checkpoints[0].relative_to(ROOT)) if len(checkpoints) == 1 else None,
        "parse_error": parse_error,
        "loss": loss_summary(scalars),
        "throughput": load_profile(run_id),
        "gradient_probe": gradient,
        "adapter_parameter_ratio": parameter_count.get("ratio") if parameter_count else None,
        "condition_usage_full": usage.get("modes", {}).get("111") if usage else None,
        "condition_usage": usage,
        "screen_evaluation_present": (OUTPUT / "screen_eval" / f"{backbone}_{variant}").exists(),
    }


def markdown(records: list[dict[str, Any]]) -> str:
    """Render the compact comparison table used by the study document."""
    lines = [
        "# Stage-1 BEV condition-injection campaign results",
        "",
        "Generated by `scripts/summarize_bev_condition_campaign.py`. Loss means use the",
        "logged 100-step TensorBoard samples; throughput statistics use synchronized",
        "50-step profiler samples. `—` means the run or its post-run evaluator is incomplete.",
        "",
        "| Backbone | Variant | Complete | Last step | Epoch means (sampled) | p50/p95 step (s) | p50/p95 gap (s) | Peak alloc/reserved (MiB) | Adapter/core | Smoke finite | full-mode Gzero/Gshuffle/Δv | Checkpoint |",
        "| --- | --- | :---: | ---: | --- | --- | --- | --- | ---: | :---: | --- | --- |",
    ]
    for record in records:
        loss, throughput = record["loss"], record["throughput"]
        epoch_means = ", ".join(f"e{epoch}:{value:.4f}" for epoch, value in loss["sampled_epoch_mean"].items()) or "—"
        usage = record["condition_usage_full"] or {}
        condition_use = "/".join(number(usage.get(key)) for key in ("g_zero", "g_shuffle", "delta_v"))
        probe = record["gradient_probe"] or {}
        finite = all(probe.get(key) is True for key in ("finite_loss", "finite_condition_grads", "finite_core_grads"))
        lines.append(
            "| {backbone} | {variant} | {complete} | {last_step} | {epoch_means} | {step} | {gap} | {memory} | {adapter} | {finite} | {condition_use} | {checkpoint} |".format(
                backbone=record["backbone"], variant=record["variant"],
                complete="yes" if record["complete"] else "no",
                last_step=loss["last_step"] if loss["last_step"] is not None else "—",
                epoch_means=epoch_means,
                step=f"{number(throughput['median_step_seconds'])}/{number(throughput['p95_step_seconds'])}",
                gap=f"{number(throughput['median_data_gap_seconds'])}/{number(throughput['p95_data_gap_seconds'])}",
                memory=f"{number(throughput['max_gpu_allocated_mb'], 1)}/{number(throughput['max_gpu_reserved_mb'], 1)}",
                adapter=number(record["adapter_parameter_ratio"], 4),
                finite="yes" if finite else "—",
                condition_use=condition_use, checkpoint=record["checkpoint"] or "—",
            )
        )
    lines.extend(["", "## 500-step sampled loss means", ""])
    for record in records:
        values = record["loss"]["loss_500_step_mean"]
        text = ", ".join(f"{int(key)}–{int(key) + 499}: {value:.4f}" for key, value in values.items()) or "—"
        lines.append(f"- `{record['backbone']}/{record['variant']}`: {text}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "phase1_summary.json")
    parser.add_argument("--markdown", type=Path, default=OUTPUT / "phase1_summary.md")
    args = parser.parse_args()
    records = [run_record(backbone, variant) for backbone in BACKBONES for variant in VARIANTS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(records), encoding="utf-8")
    print(f"Wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()

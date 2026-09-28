"""Aggregate verified speed-environment probes into compact CSV artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "env" / "results"
LOGS = ROOT / "env" / "logs"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def main() -> None:
    experiments = [
        ("legacy_torch113_fp32", "teacher_amp_legacy_torch113_fp32.json"),
        ("torch2_fp32", "teacher_amp_fp32.json"),
        ("torch2_fp16", "teacher_amp_fp16.json"),
        ("torch2_bf16", "teacher_amp_bf16.json"),
        ("torch2_fp32_no_empty_cache", "teacher_amp_torch2_fp32_no_empty_cache.json"),
        ("torch2_bf16_no_empty_cache", "teacher_amp_torch2_bf16_no_empty_cache.json"),
    ]
    rows = []
    for experiment, filename in experiments:
        record = load(filename)
        record["experiment"] = experiment
        record.setdefault("empty_cache_mode", "current")
        rows.append(record)
    baseline = rows[0]
    for row in rows:
        row["speedup_vs_legacy"] = baseline["mean_ms"] / row["mean_ms"]
        row["memory_saving_vs_legacy"] = 1 - row["peak_allocated_mb"] / baseline["peak_allocated_mb"]
        row["loss_rel_error_vs_legacy"] = abs(row["loss_mean"] - baseline["loss_mean"]) / baseline["loss_mean"]

    fields = [
        "experiment", "torch", "cuda", "precision", "empty_cache_mode", "batch_size", "points",
        "warmup_steps", "timed_steps", "mean_ms", "median_ms", "p95_ms", "speedup_vs_legacy",
        "peak_allocated_mb", "peak_reserved_mb", "memory_saving_vs_legacy", "loss_mean",
        "loss_rel_error_vs_legacy", "finite",
    ]
    with (RESULTS / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    amp_rows = []
    for scope, precision, log_name in [
        ("MinkowskiEngine synthetic", "fp32", "probe_me_fp32.log"),
        ("MinkowskiEngine synthetic", "fp16", "probe_me_fp16.log"),
        ("MinkowskiEngine synthetic", "bf16", "probe_me_bf16.log"),
    ]:
        text = (LOGS / log_name).read_text()
        amp_rows.append({
            "scope": scope,
            "precision": precision,
            "steps": 100,
            "status": "pass" if f"{precision} ME PASS" in text or (precision == "fp32" and "FP32 ME PASS" in text) else "fail",
            "coordinates": "FP32 before ME internal integer conversion",
            "evidence": log_name,
        })
    for experiment, filename in experiments[1:]:
        record = load(filename)
        amp_rows.append({
            "scope": "real Teacher DCD-only",
            "precision": record["precision"],
            "steps": record["warmup_steps"] + record["timed_steps"],
            "status": "pass" if record["finite"] else "fail",
            "coordinates": "FP32; AMP restricted to ME backbone",
            "evidence": filename,
        })
    with (RESULTS / "amp_probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(amp_rows[0]))
        writer.writeheader()
        writer.writerows(amp_rows)

    compile_result = load("compile_probe.json")
    compile_rows = [
        {"scope": "dense MLP", "inductor_callable": compile_result["dense_compile"], "captured_ops": "n/a", "effective": True, "notes": "Inductor execution passed."},
        {
            "scope": "ME sparse Conv/BN/ReLU",
            "inductor_callable": compile_result["me_compile_forward"],
            "captured_ops": compile_result["me_captured_op_count"],
            "effective": compile_result["me_inductor_effective"],
            "notes": compile_result["me_dynamo_summary"],
        },
    ]
    with (RESULTS / "compile_probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compile_rows[0]))
        writer.writeheader()
        writer.writerows(compile_rows)


if __name__ == "__main__":
    main()

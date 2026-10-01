#!/usr/bin/env python3
"""Create a compact, reproducible table from Legacy BEVFlow speed results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULTS = (
    ("B0 FP32 eager e2e", "b0_e2e_w4.json"),
    ("B2 fast DataLoader e2e", "b2_e2e_w4.json"),
    ("C0 FP32 TF32", "c0_fp32_tf32.json"),
    ("C1 BF16", "c1_bf16.json"),
    ("C2 FP16 eager", "c2_fp16.json"),
    ("C3 FP32 Inductor", "c3_fp32_inductor.json"),
    ("C4 FP16 Inductor", "c4_fp16_inductor.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/research_v2/bevflow_speed"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for name, filename in DEFAULTS:
        path = args.results / filename
        if not path.exists():
            rows.append({"name": name, "status": "not_run_or_unsupported"})
            continue
        value = json.loads(path.read_text())
        end_to_end_sps = value.get("end_to_end_samples_per_sec")
        if end_to_end_sps is None and value.get("mode") == "e2e":
            effective_ms = (
                float(value.get("mean_ms", 0.0))
                + float(value.get("data_wait_mean_ms", 0.0))
                + float(value.get("h2d_mean_ms", 0.0))
            )
            end_to_end_sps = 1000.0 * float(value.get("batch_size", 1)) / effective_ms
        rows.append({
            "name": name, "status": "pass" if value.get("finite") else "nonfinite",
            "precision": value.get("precision"), "compile_dense": value.get("compile_dense"),
            "mean_ms": value.get("mean_ms"), "samples_per_sec": value.get("samples_per_sec"),
            "end_to_end_samples_per_sec": end_to_end_sps,
            "p99_ms": value.get("p99_ms"), "peak_allocated_mb": value.get("peak_allocated_mb"),
        })
    output = args.output or args.results / "single_gpu_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    # Include the end-to-end paired short-run result when both final
    # checkpoints have produced their loss and quality summaries.
    training_rows = []
    for label, stem, method in (
        ("B0 FP32 eager", "final_fp32", "FP32"),
        ("F0 FP16 dense Inductor", "final_fp16", "FP16+Inductor"),
    ):
        loss_path = args.results / f"{stem}_throughput_loss_summary.csv"
        usage_path = args.results / f"{stem}_usage_b100.json"
        quality_path = args.results / f"{stem}_lidar_b20_cfg2" / "summary.json"
        if not loss_path.exists():
            continue
        summaries = {}
        with loss_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                summaries[(row["kind"], row["global_step"])] = row.get("mean_loss")
        row = {
            "model": label,
            "precision_compile": method,
            "loss_0_499": summaries.get(("trailing_500", "500")),
            "loss_500_999": summaries.get(("trailing_500", "1000")),
            "loss_1000_1499": summaries.get(("trailing_500", "1500")),
            "loss_final_200": summaries.get(("trailing_200", "1500")),
            "loss_all_1500": summaries.get(("epoch", "1500")),
        }
        if usage_path.exists():
            usage = json.loads(usage_path.read_text())
            row["gshuffle_100_t_lt_0_2"] = usage["headlines"].get("gshuffle_100_t_lt_0_2")
            row["gshuffle_100_t_lt_0_4"] = usage["headlines"].get("gshuffle_100_t_lt_0_4")
            row["gshuffle_111_t_lt_0_2"] = usage["headlines"].get("gshuffle_111_t_lt_0_2")
        if quality_path.exists():
            quality = json.loads(quality_path.read_text())["bevflow_lidar_only"]
            for metric in (
                "occupancy_iou", "occupancy_f1", "completion_f1",
                "height_mae_gtocc_m", "density_mass_tv",
            ):
                row[metric] = quality[metric]["mean"]
        training_rows.append(row)
    if training_rows:
        training_output = args.results / "final_comparison.csv"
        training_keys = sorted({key for row in training_rows for key in row})
        with training_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=training_keys)
            writer.writeheader(); writer.writerows(training_rows)
        print(training_output)
    print(output)


if __name__ == "__main__":
    main()

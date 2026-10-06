#!/usr/bin/env python3
"""Final seq08 B200x3 comparison; preserve historical execution provenance."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/research_v2/formal_bev_compare"
OLD = BASE / "sid2_dip_warnonly_500ep_20261003"
DIP = BASE / "dip_single_gpu3_resume50_b32_20261005"
LEGACY = BASE / "legacy_final_gpu2_20261004"
FRAME_MANIFEST = ROOT / "configs/research_v2/formal_bev_compare/seq08_b200.txt"
MODELS = {"FPSGen": LEGACY, "SiD2-style": OLD / "sid2_s_formal", "DiP-S/16": DIP}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, values):
    if not values:
        return
    keys = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(values)


def gpu_free():
    gpu = subprocess.check_output(["nvidia-smi", "-i", "3", "--query-gpu=uuid,memory.used",
                                   "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, used = gpu[0].strip(), int(gpu[1])
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader,nounits"], text=True)
    if used > 500 or any(line.split(",")[0].strip() == uuid for line in apps.splitlines()):
        raise RuntimeError("GPU3 occupied; evaluation stopped without interrupting another job")


def final_losses(path):
    data = rows(path)
    result = {}
    for kind, key in [("trailing_500", "train_loss_final500"), ("trailing_200", "train_loss_final200")]:
        selected = [r for r in data if r["kind"] == kind and int(r["global_step"]) == 299000]
        result[key] = float(selected[-1]["mean_loss"]) if selected else None
    return result


def summarize(root):
    from scripts.summarize_formal_bev_compare import METRICS, frame_means, paired_bootstrap, validate_final_summary
    frames = FRAME_MANIFEST.read_text().splitlines()
    quality, usage, by_time, efficiency, training, epoch_history, frame_data = [], [], [], [], [], [], [], {}
    for model, directory in MODELS.items():
        result_dir = root / "dip" if model == "DiP-S/16" else directory
        hashes = []
        for cfg in (1, 2):
            summary = read(result_dir / f"cfg{cfg}/summary.json")
            validate_final_summary(summary, cfg)
            if summary["checkpoint_global_step"] != 299000:
                raise ValueError("Final checkpoint must complete299000 optimizer steps")
            if Path(summary["manifest"]).read_bytes() != FRAME_MANIFEST.read_bytes():
                raise ValueError("Frozen frame manifest mismatch")
            hashes.append(summary["checkpoint_sha256"])
            samples = frame_means(result_dir / f"cfg{cfg}/samples.csv", expected_frames=frames)
            frame_data[(model, cfg)] = samples
            quality.append(dict(method=model, epoch=500, cfg=cfg, frames=200, generation_seeds_per_frame=3,
                                **{m: float(np.mean([samples[f][m] for f in frames])) for m in METRICS}))
        if len(set(hashes)) != 1:
            raise ValueError("CFG1 and CFG2 use different checkpoints")
        condition = read(result_dir / "condition_usage.json")
        if condition["frames"] != 200 or condition["checkpoint_global_step"] != 299000:
            raise ValueError("Condition evaluation is not final B200")
        for mode in ["100", "010", "001", "111"]:
            for cutoff in ["0.2", "0.4", "1.0"]:
                values = condition["modes"][mode][cutoff]
                usage.append(dict(method=model, mode=mode, time_cutoff=float(cutoff), **values))
            for time_bin, values in condition["modes"][mode]["by_time_bin"].items():
                by_time.append(dict(method=model, mode=mode, time_bin=time_bin, **values))
        infer = read(root / f"efficiency/{model}/inference.json")
        if model == "FPSGen":
            reuse = read(OLD / "legacy/reuse_manifest.json")
            params = reuse["params"]
            train_dir = ROOT / "outputs/research_v2/bevflow_formal_b16_gpu23_fp16_inductor"
            losses = final_losses(train_dir / "throughput_loss_summary.csv")
            provenance = dict(execution="2GPU x16 throughout", train_wall_seconds=None,
                              fp16_skipped_updates=None, train_peak_allocated_mib=None,
                              historical_provenance="Incomplete: no original rank manifests/scaler skip counter")
        elif model == "SiD2-style":
            manifests = [read(directory / f"run_manifest_rank{i}.json") for i in [0, 1]]
            if any(d["status"] != "COMPLETE" or d["completed_epochs"] != 500 for d in manifests):
                raise ValueError("SiD2 training incomplete")
            params = manifests[0]["params"]; train_dir = directory
            losses = final_losses(train_dir / "throughput_loss_summary.csv")
            provenance = dict(execution="2GPU x16 throughout", train_wall_seconds=max(d["wall_seconds"] for d in manifests),
                              fp16_skipped_updates_by_rank=[d["skipped_optimizer_steps"] for d in manifests],
                              train_peak_allocated_mib=max(d["peak_allocated_mb"] for d in manifests))
        else:
            manifest = read(DIP / "run_manifest_rank0.json")
            if manifest["status"] != "COMPLETE" or manifest["completed_epochs"] != 500:
                raise ValueError("DiP continuation incomplete")
            handoff = read(DIP / "handoff_manifest.json")
            params = manifest["params"]; train_dir = DIP
            losses = final_losses(train_dir / "throughput_loss_summary.csv")
            provenance = dict(execution="epoch1-50:2GPU x16; epoch51-500:GPU3 x32",
                              continuation_wall_seconds=manifest["wall_seconds"],
                              fp16_skipped_updates_continuation=manifest["skipped_optimizer_steps"],
                              fp16_skipped_updates_first50=None,
                              discarded_steps_at_handoff=handoff["discarded_after_checkpoint_steps"],
                              train_peak_allocated_mib=manifest["peak_allocated_mb"],
                              first50_peak_allocated_mib=handoff["old_manifest"]["peak_allocated_mb"],
                              protocol_limitation="Global batch32 unchanged; world-size, sample order and private RNG streams changed at resume; not a full-DDP matched run")
        throughput = rows(train_dir / "throughput.csv")
        steady = [r for r in throughput if 100 < int(r["global_step"]) <= 400] if model != "DiP-S/16" else [r for r in throughput if 30000 < int(r["global_step"]) <= 30300]
        train_speed = 9600 / sum(float(r["interval_wall_ms"]) / 1000 for r in steady) if len(steady) == 3 else None
        item = dict(method=model, params_core=params["core"], params_condition=params["condition"],
                    params_pointpillar=params["pointpillar"], params_total=params["total"],
                    train_early_300step_samples_sec=train_speed,
                    train_early_speed_scope="After compile startup; DiP continuation not comparable to twoGPU speed",
                    inference_gpu=infer["gpu"], inference_precision=infer["precision"], inference_backend=infer["backend"],
                    inference_b1_ms=infer["batches"]["1"]["mean_ms_per_sample"],
                    inference_b8_samples_sec=infer["batches"]["8"]["samples_per_second"],
                    inference_b1_peak_mib=infer["batches"]["1"]["peak_allocated_mb"],
                    inference_b8_peak_mib=infer["batches"]["8"]["peak_allocated_mb"])
        efficiency.append(item)
        training.append(dict(method=model, epoch=500, global_step=299000, **losses, **provenance))
        if model == "FPSGen":
            epoch_rows = [dict(method=model, epoch=int(r['epoch'])+1, global_step=int(r['global_step']),
                               mean_loss=float(r['mean_loss']), scope='historical rank0 logger epoch mean')
                          for r in rows(train_dir/'throughput_loss_summary.csv') if r['kind']=='epoch']
        else:
            epoch_rows = [dict(method=model, epoch=int(r['epoch'])+1, global_step=int(r['global_step']),
                               mean_loss=float(r['mean_loss']), wall_seconds=float(r['wall_seconds']))
                          for r in rows(train_dir/'epoch_loss.csv')]
            if model == 'DiP-S/16':
                epoch_rows = [dict(method=model, epoch=int(r['epoch'])+1, global_step=int(r['global_step']),
                                   mean_loss=float(r['mean_loss']), wall_seconds=float(r['wall_seconds']))
                              for r in rows(OLD/'dip_s16_formal/epoch_loss.csv') if int(r['epoch'])<50] + epoch_rows
        if [r['epoch'] for r in epoch_rows] != list(range(1,501)):
            raise ValueError(f'Incomplete/nonunique epoch loss history: {model}')
        epoch_history.extend(epoch_rows)
    comparisons = {}
    for model in ["SiD2-style", "DiP-S/16"]:
        comparisons["FPSGen-" + model] = {}
        for cfg in (1, 2):
            a, b = frame_data[("FPSGen", cfg)], frame_data[(model, cfg)]
            comparisons["FPSGen-" + model][f"cfg{cfg}"] = {
                m: paired_bootstrap([a[f][m] for f in frames], [b[f][m] for f in frames],
                                    lower_is_better=m in METRICS[-2:]) for m in METRICS}
    for name, values in [("final_quality.csv", quality), ("final_condition_usage.csv", usage),
                         ("final_condition_usage_by_tbin.csv", by_time),
                         ("final_efficiency.csv", efficiency), ("training_summary.csv", training),
                         ("epoch_loss_history.csv", epoch_history)]:
        write_csv(root / name, values)
    save(root / "paired_bootstrap.json", dict(comparisons=comparisons,
        policy="Average3 generation seeds within frame; paired10000-resample bootstrap over200 frames; seed20261001; one training seed, not architecture-wide uncertainty"))
    save(root / "final_report.json", dict(quality=quality, condition_usage=usage, efficiency=efficiency,
                                         training=training, paired_comparisons=comparisons))
    print("FINAL QUALITY", json.dumps(quality), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise ValueError("Only physicalGPU3 authorized; GPU2 remains released")
    root = args.output.resolve()
    if not root.is_relative_to(ROOT / "outputs"):
        raise ValueError("Use fresh repository outputs")
    gpu_free()
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = {"FPSGen": Path(read(OLD / "legacy/reuse_manifest.json")["checkpoint"]),
                   "SiD2-style": OLD / "sid2_s_formal/final.ckpt", "DiP-S/16": DIP / "final.ckpt"}
    if not all(p.is_file() for p in checkpoints.values()):
        raise FileNotFoundError("Final checkpoints missing")
    save(root / "campaign_manifest.json", dict(
        checkpoints={m: str(p) for m, p in checkpoints.items()},
        completed_epochs=500, optimizer_steps=299000, physical_gpu=3,
        frame_manifest=str(FRAME_MANIFEST), frame_manifest_sha256=hashlib.sha256(FRAME_MANIFEST.read_bytes()).hexdigest(),
        frames=200, sequence="08", condition="100", sampler="Euler10", cfg=[1, 2],
        generation_seeds=[20261001, 20261002, 20261003],
        reuse_evaluations={"FPSGen": str(LEGACY), "SiD2-style": str(OLD / "sid2_s_formal")},
        scope="Stage1BEV only, not PointFlow or full point-cloud completion; B200 subset not every seq08 frame",
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    stages = []
    for cfg in (1, 2):
        stages.append((f"dip_cfg{cfg}", ["scripts/eval_bevflow_lidar_only.py", "--bev-ckpt", str(checkpoints["DiP-S/16"]),
            "--dataset-root", "/data-12/M2024-HWZ/KITTI_Odometry", "--manifest", str(FRAME_MANIFEST),
            "--output", str(root / f"dip/cfg{cfg}"), "--gt-dir", "gt_possion",
            "--generation-seeds", "20261001", "20261002", "20261003", "--guidance-scale", str(cfg),
            "--bev-steps", "10", "--runtime-precision", "fp16", "--dense-backend", "inductor",
            "--save-visuals", "0", "--progress-interval", "10"]))
    stages.append(("dip_condition", ["scripts/eval_formal_bev_condition_usage.py", "--checkpoint", str(checkpoints["DiP-S/16"]),
        "--data-root", "/data-12/M2024-HWZ/KITTI_Odometry", "--manifest", str(FRAME_MANIFEST),
        "--output", str(root / "dip/condition_usage.json")]))
    for model, checkpoint in checkpoints.items():
        stages.append((model + "_inference", ["scripts/benchmark_formal_bev_inference.py", "--checkpoint", str(checkpoint),
            "--data-root", "/data-12/M2024-HWZ/KITTI_Odometry", "--manifest", str(FRAME_MANIFEST),
            "--dense-backend", "inductor", "--output", str(root / f"efficiency/{model}/inference.json")]))
    completed = []
    for stage, arguments in stages:
        gpu_free()
        command = [sys.executable, "-u", *arguments]
        save(root / "queue_status.json", dict(status="RUNNING", stage=stage, completed=completed, command=command))
        print("START", stage, flush=True)
        with (root / f"{stage.replace('/', '_')}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end="", flush=True)
            code = process.wait()
        if code:
            save(root / "queue_status.json", dict(status="FAILED", stage=stage, returncode=code, completed=completed))
            raise RuntimeError(f"Evaluation stage failed: {stage}; no protocol/backend change")
        completed.append(stage)
    summarize(root)
    save(root / "queue_status.json", dict(status="COMPLETE", completed=completed,
         end=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print("COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()

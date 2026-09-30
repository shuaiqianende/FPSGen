#!/usr/bin/env python3
"""Barrier, ranking, and seed-123 confirmation for the BEV condition study.

This process intentionally owns no CUDA device.  It waits for the two
Phase-1 workers to finish their fixed queues, reruns every B20 screen with the
required campaign seed, ranks C0–C4 *within* each backbone, and then uses one
serial subprocess per physical GPU for the six Phase-2 confirmations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

try:  # ``python scripts/...`` and ``import scripts...`` are both supported.
    from rank_bev_condition_campaign import (
        BACKBONES, C0_RUN_IDS, INJECTION, OUTPUT, ROOT, final_500_sampled_loss,
        screen_metrics, usage_metrics,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by module consumers.
    from scripts.rank_bev_condition_campaign import (
        BACKBONES, C0_RUN_IDS, INJECTION, OUTPUT, ROOT, final_500_sampled_loss,
        screen_metrics, usage_metrics,
    )


BASE = {
    "hdit": ROOT / "configs/research_v2/train_bev_hdit_s_gt_possion_5ep_b8_gpu2.yaml",
    "dip": ROOT / "configs/research_v2/train_bev_dip_s_gt_possion_5ep_b8_gpu3.yaml",
    "ncsnpp": ROOT / "configs/research_v2/train_bev_ncsnpp_s_gt_possion_5ep_b8_gpu1.yaml",
}
PHASE1_VARIANTS = ("spatial", "global", "separate", "native")
GPU_QUEUES = {2: ("ncsnpp", "hdit", "dip"), 3: ("ncsnpp", "hdit", "dip")}


def command_env(gpu: int) -> dict[str, str]:
    if gpu not in {2, 3}:
        raise ValueError(f"Only physical GPU2/GPU3 may run campaign work, got GPU{gpu}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TRAIN_DATABASE", "/data-12/M2024-HWZ/KITTI_Odometry")
    env["TMPDIR"] = str(ROOT / "env/tmp")
    env["MPLCONFIGDIR"] = str(ROOT / "env/mpl_cache")
    env["XDG_CACHE_HOME"] = str(ROOT / "env/xdg_cache")
    return env


def run(command: list[str], gpu: int, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=command_env(gpu), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def verify_gpu_contract(gpu: int) -> None:
    code = (
        "import torch; "
        "assert torch.cuda.device_count() == 1, torch.cuda.device_count(); "
        "assert torch.cuda.current_device() == 0; "
        "print(torch.cuda.get_device_name(0))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=command_env(gpu), check=True)


def verify_adapter_budget(config: Path, gpu: int) -> None:
    """Recheck the <5% adapter/core contract for every fresh Phase-2 run."""
    code = """
import json, sys, yaml
from fpsgen.models.gen_img import FlowIMG
cfg=yaml.safe_load(open(sys.argv[1]))
model=FlowIMG(cfg).model
core=sum(p.numel() for p in model.core.parameters())
adapter=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('core.') and not n.startswith('pc_encoder.'))
print(json.dumps({'core': core, 'adapter': adapter, 'ratio': adapter / core}))
assert adapter < .05 * core, (adapter, core)
"""
    result = subprocess.check_output([sys.executable, "-c", code, str(config)], cwd=ROOT,
                                     env=command_env(gpu), text=True)
    path = OUTPUT / "parameter_counts" / f"{config.stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result, encoding="utf-8")


def checkpoint(backbone: str, variant: str, *, phase2: bool = False) -> Path | None:
    if phase2:
        run_id, epoch = f"bev_cond_{backbone}_{variant}_s123_8ep", "07"
    elif variant == "hybrid_shared":
        run_id, epoch = C0_RUN_IDS[backbone], "04"
    else:
        run_id, epoch = f"bev_cond_{backbone}_{variant}_s42_5ep", "04"
    candidates = sorted((ROOT / "experiments" / run_id).glob(
        f"lightning_logs/version_*/checkpoints/*epoch={epoch}.ckpt"
    ))
    return candidates[0] if len(candidates) == 1 else None


def phase1_training_ready() -> bool:
    return all(
        checkpoint(backbone, variant) is not None
        for backbone in BACKBONES for variant in PHASE1_VARIANTS
    )


def phase1_postprocess_ready() -> bool:
    return all(
        (OUTPUT / "condition_usage" / f"{backbone}_{variant}.json").exists()
        and (OUTPUT / "screen_eval" / f"{backbone}_{variant}" / "summary.json").exists()
        for backbone in BACKBONES for variant in PHASE1_VARIANTS
    )


def wait_for_phase1() -> None:
    while not (phase1_training_ready() and phase1_postprocess_ready()):
        print("Phase-1 workers are still active; waiting for all C1–C4 checkpoints and evaluators.", flush=True)
        time.sleep(60)


def evaluate(backbone: str, variant: str, gpu: int, *, frames: int, samples: int) -> None:
    """Run fixed-seed usage plus LiDAR-only generation evaluation for one checkpoint."""
    ckpt = checkpoint(backbone, variant)
    if ckpt is None:
        raise FileNotFoundError(f"Missing Phase-1 checkpoint for {backbone}/{variant}")
    data_root = command_env(gpu)["TRAIN_DATABASE"]
    run([
        sys.executable, "scripts/eval_bev_condition_usage.py", "--checkpoint", str(ckpt),
        "--data-root", data_root, "--frames", "100", "--seed", "20261001",
        "--output", str(OUTPUT / "condition_usage" / f"{backbone}_{variant}.json"),
    ], gpu, OUTPUT / "logs" / f"{backbone}_{variant}_usage_seed20261001.log")
    manifest = "configs/research_v2/gate_seq08_100.txt" if frames == 100 else "configs/research_v2/gate_seq08_20.txt"
    run([
        sys.executable, "scripts/eval_bevflow_lidar_only.py", "--bev-ckpt", str(ckpt),
        "--dataset-root", data_root, "--manifest", manifest, "--samples-per-frame", str(samples),
        "--save-visuals", "0", "--base-seed", "20261001",
        "--output", str(OUTPUT / "screen_eval" / f"{backbone}_{variant}"),
    ], gpu, OUTPUT / "logs" / f"{backbone}_{variant}_screen_seed20261001.log")


def rescreen_phase1() -> None:
    """Overwrite provisional screens so C0–C4 all use the same B20/seed contract."""
    queues = {
        2: [("hdit", variant) for variant in ("hybrid_shared", *PHASE1_VARIANTS)]
        + [("ncsnpp", variant) for variant in ("hybrid_shared", "spatial", "global")],
        3: [("dip", variant) for variant in ("hybrid_shared", *PHASE1_VARIANTS)]
        + [("ncsnpp", variant) for variant in ("separate", "native")],
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_serial_evaluate, queue, gpu) for gpu, queue in queues.items()]
        for future in futures:
            future.result()


def _serial_evaluate(queue: list[tuple[str, str]], gpu: int) -> None:
    for backbone, variant in queue:
        evaluate(backbone, variant, gpu, frames=20, samples=1)


def materialize_phase2(backbone: str, variant: str) -> Path:
    """Create a fresh seed-123, eight-epoch config without changing C0's topology."""
    cfg = yaml.safe_load(BASE[backbone].read_text(encoding="utf-8"))
    run_id = f"bev_cond_{backbone}_{variant}_s123_8ep"
    cfg["experiment"]["id"] = run_id
    cfg["train"].update({"n_gpus": 1, "batch_size": 8, "max_epoch": 8,
                         "checkpoint_every_n_epochs": 1, "seed": 123})
    cfg["train"].pop("limit_train_batches", None)
    cfg["runtime"].update({
        "compile_dense": False,
        "visualization_interval": 1000,
        "throughput_csv": str(Path("outputs/condition_campaign/profiles") / f"{run_id}.csv"),
    })
    # The C0 policy remains shared spatial+global fusion.  Phase-2 applies the
    # v2 compact adapter contract to it as well, so a historical full-width C0
    # adapter cannot evade the same <5% fairness audit as C1–C4.
    settings = (
        {"spatial": True, "global": True, "fusion": "shared", "native": False}
        if variant == "hybrid_shared"
        else yaml.safe_load(
            (ROOT / "configs/research_v2/condition_ablation/phase1.yaml").read_text(encoding="utf-8")
        )["campaign"]["variants"][variant]
    )
    cfg["model"]["condition"].update({**settings, "gate_init": 0.1, "bias": False, "api_version": 2})
    path = OUTPUT / "configs" / f"train_bev_{backbone}_cond_{variant}_s123_8ep.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def phase2_run(backbone: str, variant: str, gpu: int) -> None:
    config = materialize_phase2(backbone, variant)
    verify_adapter_budget(config, gpu)
    run([sys.executable, "fpsgen/train_bev.py", "--config", str(config)], gpu,
        OUTPUT / "logs" / f"{config.stem}.log")
    ckpt = checkpoint(backbone, variant, phase2=True)
    if ckpt is None:
        raise FileNotFoundError(f"Expected epoch=07 checkpoint for {backbone}/{variant}")
    data_root = command_env(gpu)["TRAIN_DATABASE"]
    phase2 = OUTPUT / "phase2"
    run([
        sys.executable, "scripts/eval_bev_condition_usage.py", "--checkpoint", str(ckpt),
        "--data-root", data_root, "--frames", "100", "--seed", "20261001",
        "--output", str(phase2 / "condition_usage" / f"{backbone}_{variant}.json"),
    ], gpu, OUTPUT / "logs" / f"{backbone}_{variant}_s123_usage.log")
    run([
        sys.executable, "scripts/eval_bevflow_lidar_only.py", "--bev-ckpt", str(ckpt),
        "--dataset-root", data_root, "--manifest", "configs/research_v2/gate_seq08_100.txt",
        "--samples-per-frame", "3", "--save-visuals", "0", "--base-seed", "20261001",
        "--output", str(phase2 / "screen_eval" / f"{backbone}_{variant}"),
    ], gpu, OUTPUT / "logs" / f"{backbone}_{variant}_s123_screen.log")


def phase2_summary(top2: dict[str, list[str]]) -> None:
    rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for variant in top2[backbone]:
            ckpt = checkpoint(backbone, variant, phase2=True)
            screen = screen_metrics(OUTPUT / "phase2" / "screen_eval" / f"{backbone}_{variant}")
            usage = usage_metrics(OUTPUT / "phase2" / "condition_usage" / f"{backbone}_{variant}.json")
            rows.append({
                "backbone": backbone, "variant": variant, "injection": INJECTION[variant],
                "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt else None,
                "seed": 123,
                "final_500_sampled_loss": final_500_sampled_loss(
                    ROOT / "experiments" / f"bev_cond_{backbone}_{variant}_s123_8ep"
                ),
                **usage, **screen,
            })
    path = OUTPUT / "phase2_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    ranking = json.loads((OUTPUT / "phase1_ranking.json").read_text(encoding="utf-8"))
    seed42 = {(item["backbone"], item["variant"]): item for item in ranking["records"]}
    seed123 = {(item["backbone"], item["variant"]): item for item in rows}
    stability_rows = []
    for key, second in seed123.items():
        first = seed42[key]
        row = {"backbone": key[0], "variant": key[1], "injection": INJECTION[key[1]]}
        for metric in ("completion_f1", "occupancy_iou", "density_mass_tv", "gshuffle_100"):
            row[f"seed42_{metric}"] = first.get(metric)
            row[f"seed123_{metric}"] = second.get(metric)
            if first.get(metric) is not None and second.get(metric) is not None:
                row[f"mean_{metric}"] = (first[metric] + second[metric]) / 2
                row[f"difference_{metric}"] = second[metric] - first[metric]
        stability_rows.append(row)
    stability_path = OUTPUT / "seed_stability.csv"
    stability_columns = sorted({key for row in stability_rows for key in row})
    with stability_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=stability_columns)
        writer.writeheader()
        writer.writerows(stability_rows)
    comparison_columns = [
        "backbone", "injection", "final_500_sampled_loss", "gshuffle_100",
        "gshuffle_layout_mean", "density_mass_tv", "height_mae_gtocc_m",
        "occupancy_iou", "completion_f1", "condition_insensitive",
    ]
    comparison_path = OUTPUT / "final_condition_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_columns)
        writer.writeheader()
        writer.writerows({
            "backbone": item["backbone"], "injection": item["injection"],
            **{key: item.get(key) for key in comparison_columns[2:]},
        } for item in ranking["records"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-only", action="store_true", help="wait for the Phase-1 barrier without launching work")
    args = parser.parse_args()
    wait_for_phase1()
    if args.wait_only:
        return
    for gpu in (2, 3):
        verify_gpu_contract(gpu)
    rescreen_phase1()
    subprocess.run([sys.executable, "scripts/rank_bev_condition_campaign.py"], cwd=ROOT, check=True)
    ranking = json.loads((OUTPUT / "phase1_ranking.json").read_text(encoding="utf-8"))
    top2 = ranking["top2"]
    queues = {
        2: [("ncsnpp", top2["ncsnpp"][0]), ("hdit", top2["hdit"][0]), ("dip", top2["dip"][0])],
        3: [("ncsnpp", top2["ncsnpp"][1]), ("hdit", top2["hdit"][1]), ("dip", top2["dip"][1])],
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_serial_phase2, queue, gpu) for gpu, queue in queues.items()]
        for future in futures:
            future.result()
    phase2_summary(top2)


def _serial_phase2(queue: list[tuple[str, str]], gpu: int) -> None:
    for backbone, variant in queue:
        phase2_run(backbone, variant, gpu)


if __name__ == "__main__":
    main()

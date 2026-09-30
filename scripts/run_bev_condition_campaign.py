#!/usr/bin/env python3
"""Run the reproducible Stage-1 condition-injection campaign on GPU2/GPU3.

The script is deliberately a coordinator, not a distributed trainer: each
worker owns exactly one physical GPU and launches one subprocess at a time.
It materializes configs from the established C0 backbone configs, runs a
three-step B1 smoke, then a 5-epoch B8 run and its screen evaluators.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "condition_campaign"
BASE = {
    "hdit": ROOT / "configs/research_v2/train_bev_hdit_s_gt_possion_5ep_b8_gpu2.yaml",
    "dip": ROOT / "configs/research_v2/train_bev_dip_s_gt_possion_5ep_b8_gpu3.yaml",
    "ncsnpp": ROOT / "configs/research_v2/train_bev_ncsnpp_s_gt_possion_5ep_b8_gpu1.yaml",
}
CHECKPOINT_PREFIX = {
    "hdit": "fpsgen_bev_hdit_s_5ep_b8_gpu2",
    "dip": "fpsgen_bev_dip_s_5ep_b8_gpu3",
    "ncsnpp": "fpsgen_bev_ncsnpp_s_5ep_b8_gpu1",
}


def command_env(gpu: int) -> dict[str, str]:
    if gpu not in {2, 3}:
        raise ValueError(f"Campaign permits only physical GPU2/GPU3, got GPU{gpu}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TRAIN_DATABASE", "/data-12/M2024-HWZ/KITTI_Odometry")
    env["TMPDIR"] = str(ROOT / "env/tmp")
    env["MPLCONFIGDIR"] = str(ROOT / "env/mpl_cache")
    env["XDG_CACHE_HOME"] = str(ROOT / "env/xdg_cache")
    return env


def run(command, gpu: int, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=command_env(gpu), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def verify_gpu_contract(gpu: int) -> None:
    code = (
        "import torch; "
        "assert torch.cuda.device_count() == 1, torch.cuda.device_count(); "
        "assert torch.device('cuda').index is None; "
        "print(torch.cuda.get_device_name(0))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=command_env(gpu), check=True)


def verify_adapter_budget(config: Path, gpu: int) -> None:
    code = """
import json, sys, yaml
from fpsgen.models.gen_img import FlowIMG
cfg=yaml.safe_load(open(sys.argv[1]))
model=FlowIMG(cfg).model
core=sum(p.numel() for p in model.core.parameters())
# The wrapper has exactly three parameter owners: frozen-by-definition core,
# PointPillar frontend, and condition adapters/gates.  Count every parameter
# outside the first two so a newly named gate cannot escape this safety audit.
adapter=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('core.') and not n.startswith('pc_encoder.'))
print(json.dumps({'core':core,'adapter':adapter,'ratio':adapter/core}))
assert adapter < .05 * core, (adapter, core)
"""
    result = subprocess.check_output([sys.executable, "-c", code, str(config)], cwd=ROOT,
                                     env=command_env(gpu), text=True)
    (OUTPUT / "parameter_counts").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "parameter_counts" / f"{config.stem}.json").write_text(result, encoding="utf-8")


def load_manifest() -> dict:
    with (ROOT / "configs/research_v2/condition_ablation/phase1.yaml").open() as handle:
        return yaml.safe_load(handle)["campaign"]


def materialize(backbone: str, variant: str, settings: dict, *, smoke: bool) -> Path:
    with BASE[backbone].open() as handle:
        cfg = yaml.safe_load(handle)
    suffix = "smoke3" if smoke else "s42_5ep"
    run_id = f"bev_cond_{backbone}_{variant}_{suffix}"
    cfg["experiment"]["id"] = run_id
    cfg["train"].update({"n_gpus": 1, "batch_size": 1 if smoke else 8,
                         "max_epoch": 1 if smoke else 5,
                         "limit_train_batches": 3 if smoke else 1.0,
                         "checkpoint_every_n_epochs": 999 if smoke else 1,
                         "seed": 42})
    cfg["runtime"].update({"compile_dense": False, "visualization_interval": 0 if smoke else 1000})
    if smoke:
        cfg["runtime"].pop("throughput_csv", None)
    else:
        cfg["runtime"]["throughput_csv"] = str(
            Path("outputs/condition_campaign/profiles") / f"{run_id}.csv"
        )
    cfg["model"]["condition"].update({**settings, "gate_init": 0.1, "bias": False,
                                         "api_version": 2})
    path = OUTPUT / "configs" / f"train_bev_{backbone}_cond_{variant}_{suffix}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def final_checkpoint(run_id: str) -> Path:
    candidates = sorted((ROOT / "experiments" / run_id).glob("lightning_logs/version_*/checkpoints/*epoch=04.ckpt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one epoch=04 checkpoint for {run_id}, found {candidates}")
    return candidates[0]


def ncsn_c0_complete() -> bool:
    root = ROOT / "experiments" / CHECKPOINT_PREFIX["ncsnpp"]
    return any(root.glob("lightning_logs/version_*/checkpoints/*epoch=04.ckpt"))


def wait_for_ncsn_c0() -> None:
    """Keep the worker recoverable until the protected GPU1 C0 run finishes."""
    while not ncsn_c0_complete():
        print("Waiting for the protected NCSNpp C0 epoch=04 checkpoint before C1–C4.", flush=True)
        time.sleep(60)


def run_one(backbone: str, variant: str, settings: dict, gpu: int) -> None:
    (OUTPUT / "smoke").mkdir(parents=True, exist_ok=True)
    smoke = materialize(backbone, variant, settings, smoke=True)
    formal = materialize(backbone, variant, settings, smoke=False)
    verify_adapter_budget(formal, gpu)
    run([sys.executable, "fpsgen/train_bev.py", "--config", str(smoke)], gpu,
        OUTPUT / "logs" / f"{smoke.stem}.log")
    # A full-condition derivative check proves condition encoder connectivity.
    run([sys.executable, "scripts/probe_bev_backbone_gradients.py", "--config", str(formal),
         "--data-root", command_env(gpu)["TRAIN_DATABASE"], "--output",
         str(OUTPUT / "smoke" / f"{backbone}_{variant}_gradients.json")], gpu,
        OUTPUT / "logs" / f"{backbone}_{variant}_gradients.log")
    run([sys.executable, "fpsgen/train_bev.py", "--config", str(formal)], gpu,
        OUTPUT / "logs" / f"{formal.stem}.log")
    checkpoint = final_checkpoint(f"bev_cond_{backbone}_{variant}_s42_5ep")
    run([sys.executable, "scripts/eval_bev_condition_usage.py", "--checkpoint", str(checkpoint),
         "--data-root", command_env(gpu)["TRAIN_DATABASE"], "--frames", "100", "--output",
         str(OUTPUT / "condition_usage" / f"{backbone}_{variant}.json")], gpu,
        OUTPUT / "logs" / f"{backbone}_{variant}_usage.log")
    run([sys.executable, "scripts/eval_bevflow_lidar_only.py", "--bev-ckpt", str(checkpoint),
         "--dataset-root", command_env(gpu)["TRAIN_DATABASE"], "--manifest",
         "configs/research_v2/gate_seq08_20.txt", "--samples-per-frame", "1", "--save-visuals", "0",
         "--base-seed", "20261001",
         "--output", str(OUTPUT / "screen_eval" / f"{backbone}_{variant}")], gpu,
        OUTPUT / "logs" / f"{backbone}_{variant}_screen.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int, choices=(2, 3))
    parser.add_argument("--backbone", action="append", choices=tuple(BASE), default=[])
    args = parser.parse_args()
    manifest = load_manifest()
    if set(manifest["allowed_physical_gpus"]) != {2, 3}:
        raise RuntimeError("Campaign manifest must hard-restrict physical GPUs to {2, 3}")
    verify_gpu_contract(args.gpu)
    backbones = args.backbone or (["hdit", "ncsnpp"] if args.gpu == 2 else ["dip", "ncsnpp"])
    for backbone in backbones:
        if backbone == "ncsnpp" and not ncsn_c0_complete():
            wait_for_ncsn_c0()
        variants = ("spatial", "global") if backbone == "ncsnpp" and args.gpu == 2 else (
            ("separate", "native") if backbone == "ncsnpp" else tuple(manifest["variants"])
        )
        for variant in variants:
            run_one(backbone, variant, manifest["variants"][variant], args.gpu)


if __name__ == "__main__":
    main()

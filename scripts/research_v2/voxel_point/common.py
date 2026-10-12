"""Shared CLI, reproducibility, split and physical GPU checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from fpsgen.voxel_point.dataset import paired_frames, development_split, manifest_sha
from fpsgen.voxel_point.voxel_spec import VoxelGridSpec


def parser(description, *, training=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=0, help="Maximum distinct scenes; 0 uses manifest")
    p.add_argument("--smoke", action="store_true", help="Correctness only; never a quality/training readiness claim")
    if training:
        p.add_argument("--max-steps", type=int, default=0, help="Successful optimizer updates; 0 uses configured budget")
        p.add_argument("--gate-report", type=Path)
        p.add_argument("--precision", choices=("fp32", "fp16"), default=None)
    return p


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    geometry_path = Path(path).parent / cfg["geometry"]
    geometry = yaml.safe_load(geometry_path.read_text())["voxel"]
    cfg["voxel"] = geometry
    sha = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    return cfg, VoxelGridSpec(**geometry), sha


def frames_for(cfg, split, max_frames=0):
    data = cfg["data"]
    if data.get("split_manifest"):
        manifest = json.loads(Path(data["split_manifest"]).read_text())
    else:
        frames = paired_frames(data["root"], data["train_sequences"])
        manifest = development_split(frames, block_size=data["block_size"],
            guard_frames=data["guard_frames"], development_block=data["development_block"])
    if split == "test":
        if not cfg.get("evaluation", {}).get("threshold_frozen", False):
            raise ValueError("Freeze development threshold/config before loading seq08")
        frames = paired_frames(data["root"], ["08"])
    else:
        frames = manifest[split]
        if any(item["sequence"] == "08" for item in frames):
            raise ValueError("seq08 leakage into training/development")
    return frames[:max_frames] if max_frames else frames


def configure_runtime():
    # Check before CUDA initialization. Logical cuda:0/1 only, never cuda:2/3.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"2", "3", "2,3"}:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=2, 3, or 2,3 before launch")
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if smi.returncode:
        raise RuntimeError("GPU preflight failed: " + smi.stderr + smi.stdout)
    cache = ROOT / "env/voxel_point_cache"
    cache.mkdir(exist_ok=True)
    for key, suffix in (("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                        ("KEOPS_CACHE_FOLDER", "keops"), ("XDG_CACHE_HOME", "xdg")):
        os.environ.setdefault(key, str(cache / suffix))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after permitted physical-GPU preflight")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # KeOps embeds visible-device properties in its compiled NVRTC binder.
    # Reusing a one-GPU cache under two-GPU DDP makes logical device 1 invalid.
    # Isolate visibility and ranks before the first lazy KeOps import.
    keops_base = Path(os.environ["KEOPS_CACHE_FOLDER"])
    keops_cache = keops_base / ("visible_" + visible.replace(",","_")) / f"rank_{local_rank}"
    keops_cache.mkdir(parents=True,exist_ok=True)
    os.environ["KEOPS_CACHE_FOLDER"] = str(keops_cache)
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.init_process_group("nccl")
    return torch.device("cuda", local_rank), smi.stdout


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(model, path, spec, *, stage):
    if not path:
        raise ValueError(f"{stage} checkpoint is required")
    value = torch.load(path, map_location="cpu")
    if value["geometry"] != spec.to_dict() or value["stage"] != stage:
        raise ValueError("Checkpoint stage/geometry mismatch")
    model.load_state_dict(value["model"], strict=True)
    return value


def require_formal_gate(args, stage):
    if args.smoke:
        if args.max_steps > 32:
            raise ValueError("Smoke is limited to 32 successful optimizer updates")
        return
    if not args.gate_report:
        raise ValueError("Formal training requires --gate-report after correctness/quality gates")
    gate = json.loads(args.gate_report.read_text())
    if gate.get(stage) != "READY":
        raise ValueError(f"{stage} gate is not READY")

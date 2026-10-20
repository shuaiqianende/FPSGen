#!/usr/bin/env python
"""Rank Stage1 Encoder Conv3d calls on real Seq08 inputs (diagnostic only)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmark_hs_patchscene_inference import S1_SHA, S2_SHA, _admit
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import file_sha


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _group(name):
    for key in ("voxel_backbone", "voxel_neck", "local_backbone", "local_neck"):
        if key in name:
            return key
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, choices=(2, 3), default=2)
    parser.add_argument("--data-root", type=Path, default=Path("/data-12/M2024-HWZ/KITTI_Odometry"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml")
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_preflight/stage1_final_100kframes.pt"))
    parser.add_argument("--stage2-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/checkpoints/step025000.pt"))
    parser.add_argument("--frames", type=int, nargs="+", default=(2300, 700, 100))
    parser.add_argument("--warmup-frame", type=int, default=1600)
    parser.add_argument("--ncu-target", type=str, default=None,
                        help="Exact Conv3d module path to bracket with cudaProfilerStart/Stop for Nsight Compute")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly the selected physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    admission = _admit(args.physical_gpu)
    if file_sha(args.stage1_checkpoint) != S1_SHA or file_sha(args.stage2_checkpoint) != S2_SHA:
        raise ValueError("Checkpoint SHA mismatch against frozen V4 contract")
    cfg, _ = load_stage2_config(args.config)
    adapter = FrozenStage1OccupancyAdapter(**cfg["stage1"]["adapter"]).cuda().eval()
    device = torch.device("cuda:0")

    warm_np = input_xyz(args.data_root / "08" / "input_" / f"{args.warmup_frame:06d}.npy").astype(np.float32, copy=False)
    torch.manual_seed(20261019 + args.warmup_frame)
    torch.cuda.manual_seed_all(20261019 + args.warmup_frame)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        _ = adapter.model.encode_condition(torch.from_numpy(warm_np).to(device))
    torch.cuda.synchronize(device)

    calls: dict[str, list[dict]] = {}
    handles = []
    for name, module in adapter.model.encoder.named_modules():
        if not isinstance(module, nn.Conv3d):
            continue
        key = name or "<encoder>"
        pending = []

        def pre_hook(mod, inputs, _key=key, _pending=pending):
            x = _first_tensor(inputs)
            if args.ncu_target == _key:
                torch.cuda.profiler.start()
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            _pending.append((start, time.perf_counter(), x))

        def post_hook(mod, inputs, output, _key=key, _pending=pending):
            start, host_start, x = _pending.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if args.ncu_target == _key:
                torch.cuda.profiler.stop()
            y = _first_tensor(output)
            calls.setdefault(_key, []).append({
                "start": start, "end": end,
                "host_submit_ms": (time.perf_counter() - host_start) * 1000.0,
                "input_shape": list(x.shape) if x is not None else None,
                "output_shape": list(y.shape) if y is not None else None,
                "input_dtype": str(x.dtype) if x is not None else None,
                "output_dtype": str(y.dtype) if y is not None else None,
                "weight_dtype": str(mod.weight.dtype),
                "input_stride": list(x.stride()) if x is not None else None,
                "input_channels_last_3d": bool(x.is_contiguous(memory_format=torch.channels_last_3d)) if x is not None and x.ndim == 5 else None,
                "weight_channels_last_3d": bool(mod.weight.is_contiguous(memory_format=torch.channels_last_3d)),
                "kernel_size": list(mod.kernel_size),
                "stride": list(mod.stride),
                "padding": list(mod.padding),
                "dilation": list(mod.dilation),
                "groups": mod.groups,
                "input_bytes": int(x.numel() * x.element_size()) if x is not None else None,
                "output_bytes": int(y.numel() * y.element_size()) if y is not None else None,
                "weight_bytes": int(mod.weight.numel() * mod.weight.element_size()),
            })

        handles.extend((module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)))

    peak_start_alloc = torch.cuda.memory_allocated(device)
    peak_start_reserved = torch.cuda.memory_reserved(device)
    frame_rows = []
    try:
        for frame in args.frames:
            xyz_np = input_xyz(args.data_root / "08" / "input_" / f"{frame:06d}.npy").astype(np.float32, copy=False)
            seed = 20261019 + frame
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            calls.clear()
            xyz = torch.from_numpy(xyz_np).to(device)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                pyramid = adapter.model.encode_condition(xyz)
            torch.cuda.synchronize(device)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            serialized = {}
            for name, records in calls.items():
                # Use median for repeated module invocations; retain all call shapes and timings.
                timings = [float(r["start"].elapsed_time(r["end"])) for r in records]
                template = {k: v for k, v in records[0].items() if k not in ("start", "end")}
                serialized[name] = {
                    **template,
                    "group": _group(name),
                    "calls": len(records),
                    "cuda_ms_total": float(sum(timings)),
                    "cuda_ms_mean_per_call": float(np.mean(timings)),
                    "cuda_ms_median_per_call": float(np.median(timings)),
                    "host_submit_ms_total": float(sum(r["host_submit_ms"] for r in records)),
                    "per_call_cuda_ms": timings,
                    "all_call_input_shapes": [r["input_shape"] for r in records],
                }
            frame_rows.append({
                "frame": frame,
                "input_point_count": int(len(xyz_np)),
                "encoder_wall_ms": wall_ms,
                "pyramid_shapes": [list(x.shape) for x in pyramid],
                "pyramid_dtypes": [str(x.dtype) for x in pyramid],
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                "conv3d": serialized,
            })
            print(json.dumps({"frame": frame, "input_points": len(xyz_np), "encoder_ms": wall_ms,
                              "conv3d_layers": len(serialized),
                              "top5": sorted(((n, x["cuda_ms_total"]) for n, x in serialized.items()), key=lambda z: -z[1])[:5]}),
                  flush=True)
    finally:
        for handle in handles:
            handle.remove()

    # Layer-level summaries are medians across scenes and include repeated calls per frame.
    layer_names = sorted({name for row in frame_rows for name in row["conv3d"]})
    summary = []
    for name in layer_names:
        found = [row["conv3d"][name] for row in frame_rows if name in row["conv3d"]]
        summary.append({
            "name": name,
            "group": _group(name),
            "calls_per_frame": float(np.mean([v["calls"] for v in found])),
            "median_frame_cuda_ms": float(np.median([v["cuda_ms_total"] for v in found])),
            "mean_frame_cuda_ms": float(np.mean([v["cuda_ms_total"] for v in found])),
            "median_per_call_cuda_ms": float(np.median([t for v in found for t in v["per_call_cuda_ms"]])),
            "example": {k: found[0][k] for k in ("input_shape", "output_shape", "input_dtype", "output_dtype", "weight_dtype", "kernel_size", "stride", "padding", "groups", "input_stride", "input_channels_last_3d", "weight_channels_last_3d")},
        })
    summary.sort(key=lambda x: x["median_frame_cuda_ms"], reverse=True)
    try:
        ncu_version = subprocess.check_output(["ncu", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        ncu_version = None
    try:
        nsys_version = subprocess.check_output(["nsys", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        nsys_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
        cudnn_version_error = None
    except RuntimeError as exc:
        cudnn_version = None
        cudnn_version_error = str(exc)
    payload = {
        "status": "PROFILED",
        "physical_gpu": args.physical_gpu,
        "gpu_admission": admission,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": cudnn_version,
        "cudnn_version_error": cudnn_version_error,
        "stage1_checkpoint_sha256": S1_SHA, "stage2_checkpoint_sha256": S2_SHA,
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "stage2_checkpoint": str(args.stage2_checkpoint),
        "input_protocol": "SemanticKITTI seq08 input_ xyz; original Stage1 autocast FP16; batch=1; encoder only",
        "warmup_frame": args.warmup_frame,
        "memory_before_profile": {"allocated_gib": peak_start_alloc / 1024**3, "reserved_gib": peak_start_reserved / 1024**3},
        "ncu_version": ncu_version, "nsys_version": nsys_version,
        "frames": frame_rows,
        "conv3d_layer_ranking": summary,
        "notes": ["CUDA Event timings are per-module inclusive launch-to-completion times; summed nested modules overlap and are diagnostic, not additive.",
                  "host_submit_ms measures Python/module-call time without synchronizing each Conv3d.",
                  "Tensor Core/SM/DRAM utilization is not measured by PyTorch hooks; requires hardware performance counters."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "top20": [
        {"name": x["name"], "median_frame_cuda_ms": x["median_frame_cuda_ms"], "shape": x["example"]["input_shape"]}
        for x in summary[:20]
    ]}), flush=True)


if __name__ == "__main__":
    main()

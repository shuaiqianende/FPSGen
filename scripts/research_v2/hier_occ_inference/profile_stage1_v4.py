#!/usr/bin/env python
"""Diagnostic Stage1 profile on real Seq08 scans; profile timings are not benchmark timings."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from benchmark_hs_patchscene_inference import S1_SHA, S2_SHA, _admit
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_network import occupancy_lidar_features
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import file_sha


def _timed(fn, device, rows, name):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cpu0 = time.perf_counter()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize(device)
    rows.setdefault(name, []).append({"cuda_ms": float(start.elapsed_time(end)),
                                      "host_ms_including_sync": (time.perf_counter() - cpu0) * 1000})
    return out


def _attach_module_timers(root, prefix, device, events, host, handles):
    for name, module in root.named_modules():
        if not name or any(module.children()):
            continue
        key = f"{prefix}.{name}:{module.__class__.__name__}"
        pending = []
        def before(_module, _args, pending=pending):
            event = torch.cuda.Event(enable_timing=True); event.record()
            pending.append((event, time.perf_counter()))
        def after(_module, _args, _out, pending=pending, key=key):
            start, cpu0 = pending.pop()
            end = torch.cuda.Event(enable_timing=True); end.record()
            events.setdefault(key, []).append((start, end))
            host.setdefault(key, []).append((time.perf_counter() - cpu0) * 1000)
        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))


@contextmanager
def _time_tokenizer_ops(device, events, host):
    """Profile the two irregular tokenizer ops without changing their outputs."""
    import torch_scatter
    import fpsgen.hier_occ_flow.stage1_network as stage1_network
    original_unique, original_scatter = torch.unique, torch_scatter.scatter_max
    original_features = stage1_network.occupancy_lidar_features
    def wrap(fn, name):
        def timed(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            cpu0 = time.perf_counter(); start.record()
            out = fn(*args, **kwargs)
            end.record()
            events.setdefault(name, []).append((start, end))
            host.setdefault(name, []).append((time.perf_counter() - cpu0) * 1000)
            return out
        return timed
    torch.unique = wrap(original_unique, "tokenizer.torch_unique")
    torch_scatter.scatter_max = wrap(original_scatter, "tokenizer.scatter_max")
    stage1_network.occupancy_lidar_features = wrap(original_features, "encoder.input_filter_and_features")
    try:
        yield
    finally:
        torch.unique, torch_scatter.scatter_max = original_unique, original_scatter
        stage1_network.occupancy_lidar_features = original_features


@contextmanager
def _time_flow_functional_ops(events, host):
    import torch.nn.functional as functional
    original_cat, original_interp = torch.cat, functional.interpolate
    def wrap(fn, name):
        def timed(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            cpu0 = time.perf_counter(); start.record()
            out = fn(*args, **kwargs)
            end.record()
            events.setdefault(name, []).append((start, end))
            host.setdefault(name, []).append((time.perf_counter() - cpu0) * 1000)
            return out
        return timed
    torch.cat = wrap(original_cat, "flow_ops.torch_cat")
    functional.interpolate = wrap(original_interp, "flow_ops.interpolate")
    try:
        yield
    finally:
        torch.cat, functional.interpolate = original_cat, original_interp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu", type=int, choices=(2, 3), default=2)
    p.add_argument("--data-root", type=Path, default=Path("/data-12/M2024-HWZ/KITTI_Odometry"))
    p.add_argument("--config", type=Path, default=ROOT / "configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml")
    p.add_argument("--stage1-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_preflight/stage1_final_100kframes.pt"))
    p.add_argument("--stage2-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/checkpoints/step025000.pt"))
    p.add_argument("--frames", type=int, nargs=3, default=(2300, 700, 100), help="Representative low, mid and high parent-token scenes")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the selected physical GPU only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    admission = _admit(a.physical_gpu)
    if file_sha(a.stage1_checkpoint) != S1_SHA or file_sha(a.stage2_checkpoint) != S2_SHA:
        raise ValueError("Checkpoint SHA does not match frozen V3 baseline")
    cfg, _geometry = load_stage2_config(a.config)
    device = torch.device("cuda:0")
    adapter = FrozenStage1OccupancyAdapter(**cfg["stage1"]["adapter"]).to(device).eval()
    events, host, handles, rows = {}, {}, [], []
    _attach_module_timers(adapter.model.encoder, "encoder", device, events, host, handles)
    _attach_module_timers(adapter.model.flow, "flow", device, events, host, handles)

    # Warm up the full encoder/flow once before diagnostic samples.
    warm = input_xyz(a.data_root / "08" / "input_" / f"{a.frames[1]:06d}.npy").astype(np.float32, copy=False)
    xyz = torch.from_numpy(warm).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        features = adapter.model.encode_condition(xyz)
        score, _ = adapter.model.sample(xyz, nfe=2, condition=features,
                                        generator=torch.Generator(device=device).manual_seed(20261019))
    torch.cuda.synchronize(device)
    events.clear(); host.clear()

    for frame, label in zip(a.frames, ("low", "mid", "high")):
        xyz_np = input_xyz(a.data_root / "08" / "input_" / f"{frame:06d}.npy").astype(np.float32, copy=False)
        xyz = torch.from_numpy(xyz_np).to(device)
        timings = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            with _time_tokenizer_ops(device, events, host):
                features = _timed(lambda: adapter.model.encode_condition(xyz), device, timings, "encoder_total")
            observed = _timed(lambda: adapter.model.geometry.observed_mask(xyz), device, timings, "observed_mask")
            flow_context = adapter.model.flow.prepare_inference_context(observed, features)
            state = torch.randn((1, 1, *adapter.model.geometry.field_shape), device=device,
                                generator=torch.Generator(device=device).manual_seed(20261019 + frame)) * adapter.model.valid
            flow_timer_start, flow_timer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flow_timer_start.record()
            with _time_flow_functional_ops(events, host):
                for step in range(2):
                    velocity = adapter.model.flow(state, observed, None, state.new_tensor(step / 2),
                                                  inference_context=flow_context)
                    state = state + velocity.float() / 2
            flow_timer_end.record(); torch.cuda.synchronize(device)
            flow_ms = float(flow_timer_start.elapsed_time(flow_timer_end))
        stage_events, stage_host = {}, {}
        event_calls = {}
        for name, event_pairs in events.items():
            if name.startswith("encoder.") or name.startswith("flow.") or name.startswith("tokenizer.") or name.startswith("flow_ops."):
                stage = "encoder" if name.startswith("encoder.") else "flow"
                if name.startswith("tokenizer."): stage = "encoder_irregular_ops"
                if name.startswith("flow_ops."): stage = "flow_functional_ops"
                stage_events.setdefault(stage, {})[name] = sum(float(s.elapsed_time(e)) for s, e in event_pairs)
                stage_host.setdefault(stage, {})[name] = sum(host.get(name, []))
                event_calls.setdefault(stage, {})[name] = len(event_pairs)
        rows.append({"label": label, "frame": frame, "input_points": len(xyz_np),
                     "encoder_total_cuda_ms": timings["encoder_total"][0]["cuda_ms"],
                     "encoder_total_host_ms_including_sync": timings["encoder_total"][0]["host_ms_including_sync"],
                     "flow_two_steps_cuda_ms": flow_ms,
                     "module_leaf_cuda_ms_inclusive": stage_events,
                     "module_leaf_host_ms_inclusive": stage_host,
                     "module_leaf_call_count": event_calls,
                     "candidate_parents": int((score[0, 0] >= .5).sum().item())})
        events.clear(); host.clear()
        print(json.dumps(rows[-1]), flush=True)

    for handle in handles: handle.remove()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"admission": admission, "torch": torch.__version__, "cuda": torch.version.cuda,
               "stage1_checkpoint_sha256": S1_SHA, "stage2_checkpoint_sha256": S2_SHA,
               "frames": rows,
               "note": "Leaf module event timings are inclusive of their own CUDA work, not a decomposition of tokenizer functional ops. Host measurements include synchronization; module timings overlap only if custom wrappers nest leaves."}
    a.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

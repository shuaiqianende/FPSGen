#!/usr/bin/env python
"""Capture and replay the fixed two-step Stage1 Flow on a real Seq08 frame."""
from __future__ import annotations

import argparse
import json
import os
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu", type=int, choices=(2, 3), default=2)
    p.add_argument("--data-root", type=Path, default=Path("/data-12/M2024-HWZ/KITTI_Odometry"))
    p.add_argument("--config", type=Path, default=ROOT / "configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml")
    p.add_argument("--stage1-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_preflight/stage1_final_100kframes.pt"))
    p.add_argument("--stage2-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/checkpoints/step025000.pt"))
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--compile-flow", action="store_true",
                   help="Compile a validated one-step Flow wrapper before capturing CUDA Graph")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly the selected physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    admission = _admit(a.physical_gpu)
    if file_sha(a.stage1_checkpoint) != S1_SHA or file_sha(a.stage2_checkpoint) != S2_SHA:
        raise ValueError("Checkpoint SHA mismatch against frozen V4 contract")
    cfg, _ = load_stage2_config(a.config)
    adapter = FrozenStage1OccupancyAdapter(**cfg["stage1"]["adapter"]).cuda().eval()
    model = adapter.model
    device = torch.device("cuda:0")
    x_path = a.data_root / "08" / "input_" / f"{a.frame:06d}.npy"
    xyz_np = input_xyz(x_path).astype(np.float32, copy=False)
    xyz = torch.from_numpy(xyz_np).to(device)
    seed = 20261019 + a.frame
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        features = adapter.encode_features(xyz)
        observed = model.geometry.observed_mask(xyz)
    gen = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn((1, 1, *model.geometry.field_shape), device=device,
                         dtype=torch.float32, generator=gen) * model.valid
    times = torch.tensor([0.0, 0.5], device=device, dtype=torch.float32)

    class _ValidatedFlow(nn.Module):
        def __init__(self, flow):
            super().__init__()
            self.flow = flow
        def forward(self, state, hit_mask, c0, c1, c2, t):
            return self.flow.forward_inference_validated(state, hit_mask, (c0, c1, c2), t)

    flow_call = _ValidatedFlow(model.flow).eval()
    compile_seconds = 0.0
    if a.compile_flow:
        t0 = time.perf_counter()
        flow_call = torch.compile(flow_call, dynamic=False)
        compile_seconds = time.perf_counter() - t0

    def flow_pair(initial, time_values):
        state = initial
        for step in range(2):
            velocity = flow_call(state, observed, *features, time_values[step])
            velocity = velocity * model.valid
            state = state + velocity.float() / 2
        return state

    result = {"status": "NOT_RUN", "physical_gpu": a.physical_gpu, "gpu_admission": admission,
              "frame": a.frame, "input_points": len(xyz_np), "input_sha256": file_sha(x_path),
              "stage1_checkpoint_sha256": S1_SHA, "stage2_checkpoint_sha256": S2_SHA,
              "nfe": 2, "torch": torch.__version__, "cuda": torch.version.cuda,
              "compile_flow": bool(a.compile_flow), "compile_wrapper_setup_seconds": compile_seconds}
    try:
        first_call_ms = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            first_call_start = time.perf_counter()
            eager = flow_pair(source.clone(), times)
            torch.cuda.synchronize(device)
            if a.compile_flow:
                first_call_ms = (time.perf_counter() - first_call_start) * 1000
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream), torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(3):
                _ = flow_pair(source.clone(), times)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        eager_events = []
        eager_wall_start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(a.replays):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); _ = flow_pair(source.clone(), times); end.record()
                eager_events.append((begin, end))
        torch.cuda.synchronize(device)
        eager_wall_ms = (time.perf_counter() - eager_wall_start) * 1000 / a.replays
        eager_event_ms = [float(s.elapsed_time(e)) for s, e in eager_events]
        static_source = source.clone()
        static_times = times.clone()
        static_observed = observed.clone()
        static_features = tuple(t.clone() for t in features)
        # flow_pair closes over the static buffers, keeping the graph signature fixed.
        original_observed, original_features = observed, features
        observed, features = static_observed, static_features
        graph = torch.cuda.CUDAGraph()
        capture_start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            with torch.cuda.graph(graph):
                graph_output = flow_pair(static_source, static_times)
        torch.cuda.synchronize(device)
        capture_ms = (time.perf_counter() - capture_start) * 1000
        observed, features = original_observed, original_features
        graph.replay(); torch.cuda.synchronize(device)
        diff = (eager.float() - graph_output.float()).abs()
        starts, ends = [], []
        wall0 = time.perf_counter()
        for _ in range(a.replays):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); graph.replay(); end.record()
            starts.append(begin); ends.append(end)
        torch.cuda.synchronize(device)
        replay_wall_ms = (time.perf_counter() - wall0) * 1000 / a.replays
        replay_event_ms = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
        result.update({
            "status": "CAPTURED",
            "capture_ms": capture_ms,
            "replays": a.replays,
            "replay_event_mean_ms": float(np.mean(replay_event_ms)),
            "replay_event_median_ms": float(np.median(replay_event_ms)),
            "replay_wall_mean_ms": replay_wall_ms,
            "same_callable_eager_event_mean_ms": float(np.mean(eager_event_ms)),
            "same_callable_eager_event_median_ms": float(np.median(eager_event_ms)),
            "same_callable_eager_wall_mean_ms": eager_wall_ms,
            "compile_first_call_wall_ms": first_call_ms,
            "eager_vs_graph_max_abs": float(diff.max().item()),
            "eager_vs_graph_mean_abs": float(diff.mean().item()),
            "eager_vs_graph_relative_l2": float(torch.linalg.vector_norm(diff) /
                                                  torch.linalg.vector_norm(eager.float()).clamp_min(1e-12)),
            "eager_vs_graph_equal": bool(torch.equal(eager, graph_output)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        })
    except Exception as exc:
        result["status"] = "CAPTURE_FAILED"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

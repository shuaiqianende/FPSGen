"""Batch-1 HS-OccFlow vs PatchScene inference latency benchmark (5 warmups + 20 frames)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage1_layout import enable_channels_last_3d, remove_layout_hooks
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.condition import sample_pyramid, observed_children
from fpsgen.hier_occ_flow.inference_fast import gpu_select_occupancy_candidates, gpu_observed_children, gpu_occupied_fine_coords
from fpsgen.hier_occ_flow.inference import sample as sample_stage2
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords, pack_children
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from native_voxel_point_sampling import sample_native_occupied_voxels, sample_native_occupied_voxels_cuda
from inference_timing_utils import ModuleCudaTimer, event_elapsed_ms, summarize, timed_cuda_call

WARMUP_FRAMES = [0, 800, 1600, 2400, 3200]
BENCHMARK_FRAMES = [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900,
                    2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500, 3700, 3900]
POINT_SEED_BASE = 20261019
S1_SHA = "b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2"
S2_SHA = "4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe"
PS_SHA = "f50bf65f419c32bb40afa14e30440e1c4ea839dd92c678f8b4487e0d2b36140a"
PS_COMMIT = "49de6d848663956e57cdd2f95d3b364e7a7afbe6"


def _sha_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()


def _admit(gpu, allow_idle_service_context=False):
    q = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=name,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    p = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    processes = [x.strip() for x in p.splitlines() if x.strip()]
    process_mib = []
    for line in processes:
        fields = [part.strip() for part in line.split(",")]
        try:
            process_mib.append(int(fields[-1]))
        except (ValueError, IndexError):
            process_mib.append(10**9)
    info = {"physical_gpu": gpu, "name": q[0].strip(), "memory_free_mib": int(q[1]),
            "memory_total_mib": int(q[2]), "utilization_percent": int(q[3]),
            "temperature_c": int(q[4]), "power_w": float(q[5]), "sm_clock_mhz": int(q[6]),
            "compute_processes": processes}
    # GPU0 may host the idle local inference API (observed holding one 346 MiB
    # CUDA context at 0% utilization). Allow that specific low-footprint case
    # only when explicitly requested; never relax admission for GPUs 2/3.
    idle_service_context = (
        allow_idle_service_context and gpu == 0 and processes
        and sum(process_mib) <= 512 and info["memory_free_mib"] >= 18000
        and info["utilization_percent"] <= 1
    )
    if idle_service_context:
        info["admission_note"] = "explicitly allowed idle GPU0 service context; no process was stopped"
    if ((processes and not idle_service_context) or info["memory_free_mib"] < 18000
            or info["utilization_percent"] > 10):
        raise RuntimeError(f"GPU{gpu} admission rejected: {info}")
    return info


def _dump_outputs(out_dir, frame, xyz, fine_coords, geometry, jitter=None, jitter_meta=None):
    out = Path(out_dir) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    centers = geometry.centers(fine_coords, fine=True).astype(np.float32)
    payload = {"occupied_coords_xyz": np.asarray(fine_coords, dtype=np.int32),
               "voxel_center_xyz": centers, "input_xyz_sha256": _sha_array(xyz)}
    if jitter is not None:
        payload["jitter_xyz"] = jitter.astype(np.float32)
        payload["jitter_metadata"] = json.dumps(jitter_meta)
    np.savez_compressed(out / f"frame_{frame:06d}.npz", **payload)


def _run_hs_frame(xyz_np, frame, adapter, model, cfg, geom, device, *,
                  gpu_candidate_selection=False, gpu_observed_mask=False,
                  gpu_occupancy_decode=False, stage2_static_cache=False,
                  stage1_flow_condition_cache=False):
    # Every feature/flow/network event is queued asynchronously and synchronized
    # once after generation, so the measured path retains normal launch behavior.
    torch.cuda.reset_peak_memory_stats(device)
    baseline_alloc = torch.cuda.memory_allocated(device)
    wall0 = time.perf_counter()
    seed = scene_seed("08", frame, cfg["candidates"]["seed"])
    # The pinned tokenizer shuffles point order before its voxel max reduction.
    # Fix that RNG per frame so baseline/cache comparisons consume the same order.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    xyz = torch.from_numpy(np.asarray(xyz_np, dtype=np.float32)).to(device)
    encoder_value, encoder_events = timed_cuda_call(lambda: adapter.encode_features(xyz))
    features = encoder_value
    flow_timer = ModuleCudaTimer(adapter.model.flow)
    flow_context = (torch.autocast("cuda", enabled=False)
                    if getattr(adapter.model.flow, "_v6_explicit_fp16", False)
                    else torch.autocast("cuda", dtype=torch.float16))
    with flow_timer, flow_context:
        score, _state1 = adapter.model.sample(
            xyz, nfe=2, condition=features,
            generator=torch.Generator(device=device).manual_seed(seed),
            condition_cache=stage1_flow_condition_cache)
    predicted_policy = __import__("export_stage2_snapshot").predicted_policy
    policy = predicted_policy(cfg)
    policy["stage1_nfe"] = 2
    selection_started = time.perf_counter()
    candidate_cuda_start, candidate_cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    candidate_cuda_start.record()
    if gpu_candidate_selection:
        ids, scores, provenance = gpu_select_occupancy_candidates(score, geom, **policy["selection"])
        ids_np = None
    else:
        ids_np, scores_np, provenance = select_occupancy_candidates(
            score[0, 0].float().cpu().numpy(), geom, **policy["selection"])
        ids = torch.as_tensor(ids_np, device=device)
        scores = torch.as_tensor(scores_np, device=device)
    candidate_cuda_end.record()
    candidate_selection_submit_ms = (time.perf_counter() - selection_started) * 1000.0
    condition_started = time.perf_counter()
    condition_cuda_start, condition_cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    condition_cuda_start.record()
    centers_xyz = geom.centers(geom.unflatten(ids)).float()
    if gpu_observed_mask:
        observed = gpu_observed_children(xyz, ids, geom)
    else:
        if ids_np is None:
            ids_np = ids.cpu().numpy()
        observed = torch.as_tensor(observed_children(xyz, ids_np, geom), device=device)
    cond = {
        "lidar_features": sample_pyramid(features, centers_xyz, geom.origin, geom.maximum).detach().half(),
        "parent_score": scores,
        "observed_child": observed,
        "valid_child_mask": torch.as_tensor(geom.valid_children(ids), device=device),
    }
    if cond["lidar_features"].shape != (len(ids), 384):
        raise RuntimeError("Stage2 condition sampling shape mismatch")
    condition_cuda_end.record()
    stage2_condition_submit_ms = (time.perf_counter() - condition_started) * 1000.0
    noise = torch.randn((geom.num_parents, geom.k), device=device,
                        generator=torch.Generator(device=device).manual_seed(cfg["validation"]["seed"] + frame))
    source = noise[ids] * cond["valid_child_mask"]
    del noise
    s2_timer = ModuleCudaTimer(model)
    state = source
    context = None
    static_context_cuda_ms = 0.0
    stage2_context = (torch.autocast("cuda", enabled=False)
                      if model.ptv3_precision == "explicit_fp16"
                      else torch.autocast("cuda", dtype=torch.float16))
    if stage2_static_cache:
        static_cuda_start, static_cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        static_cuda_start.record()
        with stage2_context:
            context = model.prepare_inference_context(ids, **cond)
        static_cuda_end.record()
    with s2_timer, stage2_context:
        for step in range(4):
            t = state.new_tensor(step / 4)
            if context is None:
                velocity = model(ids, state=state, time=t, **cond)
            else:
                velocity = model(ids, state=state, time=t, inference_context=context)
            state = state + velocity.float() / 4
    if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()):
        raise RuntimeError(f"Non-finite HS endpoint at frame {frame}")
    decode_started = time.perf_counter()
    occupancy = (state > 0) & cond["valid_child_mask"]
    if gpu_occupancy_decode:
        fine = gpu_occupied_fine_coords(ids, occupancy, cond["valid_child_mask"], geom).cpu().numpy()
    else:
        fine = occupied_fine_coords(ids.cpu().numpy(),
                                    pack_children(occupancy.cpu().numpy(), geom), geom)
    if not len(fine):
        raise RuntimeError(f"Empty HS occupancy at frame {frame}")
    decode_wall_ms = (time.perf_counter() - decode_started) * 1000.0
    torch.cuda.synchronize(device)
    generation_wall_ms = (time.perf_counter() - wall0) * 1000.0
    candidate_selection_cuda_ms = float(candidate_cuda_start.elapsed_time(candidate_cuda_end))
    stage2_condition_cuda_ms = float(condition_cuda_start.elapsed_time(condition_cuda_end))
    if stage2_static_cache:
        static_context_cuda_ms = float(static_cuda_start.elapsed_time(static_cuda_end))
    encoder_ms = event_elapsed_ms([encoder_events])
    flow_module_ms = flow_timer.elapsed_ms()
    cache_events = adapter.model.last_condition_cache_cuda_events
    flow_context_ms = float(cache_events[0].elapsed_time(cache_events[1])) if cache_events else 0.0
    flow_ms = flow_module_ms + flow_context_ms
    stage2_ms = s2_timer.elapsed_ms()
    row = {
        "stage1_encoder_ms": encoder_ms,
        "stage1_flow_total_ms": flow_ms,
        "stage1_flow_module_ms": flow_module_ms,
        "stage1_flow_condition_projection_cache_ms": flow_context_ms,
        "stage1_flow_ms_per_step": flow_ms / 2,
        "stage1_flow_forward_calls": flow_timer.calls,
        "stage1_flow_condition_cache": bool(stage1_flow_condition_cache),
        "candidate_selection_submit_wall_ms": candidate_selection_submit_ms,
        "candidate_selection_cuda_event_ms": candidate_selection_cuda_ms,
        "stage2_condition_submit_wall_ms": stage2_condition_submit_ms,
        "stage2_condition_cuda_event_ms": stage2_condition_cuda_ms,
        "stage2_static_context_cuda_event_ms": static_context_cuda_ms,
        "stage2_network_total_ms": stage2_ms,
        "stage2_network_ms_per_step": stage2_ms / 4,
        "stage2_forward_calls": s2_timer.calls,
        "occupancy_decode_ms": decode_wall_ms,
        "network_forward_gpu_ms": encoder_ms + flow_ms + stage2_ms,
        "generation_wall_ms": generation_wall_ms,
        "occupied_voxels": int(len(fine)),
        "candidate_parents": int(len(ids)),
        "stage1_score_sha256": _sha_array(score[0, 0].detach().float().cpu().numpy()),
        "stage1_candidate_ids_sha256": _sha_array(ids.detach().cpu().numpy()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "incremental_peak_allocated_gib": max(0, torch.cuda.max_memory_allocated(device) - baseline_alloc) / 1024**3,
        "provenance": provenance,
        "gpu_candidate_selection": bool(gpu_candidate_selection),
        "gpu_observed_mask": bool(gpu_observed_mask),
        "gpu_occupancy_decode": bool(gpu_occupancy_decode),
        "stage2_static_cache": bool(stage2_static_cache),
    }
    return np.asarray(fine), row


@torch.inference_mode()
def _run_patchscene_frame(xyz_np, frame, generator, grid, steps, device):
    from patchscene_infer_seq08 import _infer_frame
    torch.cuda.reset_peak_memory_stats(device)
    baseline_alloc = torch.cuda.memory_allocated(device)
    unet_timer = ModuleCudaTimer(generator.denoise_fn)
    profile = {}
    wall0 = time.perf_counter()
    with unet_timer:
        coords = _infer_frame(generator, grid, xyz_np, steps=steps, device=device, profile=profile)
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    network_ms = unet_timer.elapsed_ms()
    expected_calls = int(profile["number_of_rings"]) * steps
    if unet_timer.calls != expected_calls:
        raise RuntimeError(f"PatchScene U-Net call count {unet_timer.calls} != rings*steps={expected_calls}")
    profile.update({
        "unet_forward_gpu_ms": network_ms,
        "unet_forward_calls": unet_timer.calls,
        "actual_denoising_steps": steps,
        "ring_generation_wall_ms": float(profile["ring_generation_wall_ms"]),
        "ring_sampling_fusion_transfer_residual_ms": max(0.0, float(profile["ring_generation_wall_ms"]) - network_ms),
        "generation_wall_ms": wall_ms,
        "network_forward_gpu_ms": network_ms,
        "occupied_voxels": int(len(coords)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "incremental_peak_allocated_gib": max(0, torch.cuda.max_memory_allocated(device) - baseline_alloc) / 1024**3,
    })
    return coords, profile


class _PatchGeometry:
    origin = (-50.0, -50.0, -3.75)
    maximum = (50.0, 50.0, 3.75)
    fine_size = (0.15625, 0.15625, 0.15625)
    fine_resolution = (640, 640, 48)
    sphere_radius = 50.0

    def centers(self, coords, fine=True):
        c = np.asarray(coords, dtype=np.float64)
        return np.asarray(self.origin) + (c + 0.5) * np.asarray(self.fine_size)


def _postprocess(fine, geometry, frame, *, jitter_backend="cpu_reference", device="cuda:0"):
    started = time.perf_counter()
    centers, _ = sample_native_occupied_voxels(fine, geometry, point_mode="voxel_center")
    center_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    sampler = sample_native_occupied_voxels_cuda if jitter_backend == "cuda" else sample_native_occupied_voxels
    kwargs = {"device": device} if jitter_backend == "cuda" else {}
    jitter, meta = sampler(fine, geometry, point_mode="native_fine_jitter", target_points=1_080_000,
                           seed=POINT_SEED_BASE + frame, **kwargs)
    if isinstance(jitter, torch.Tensor):
        jitter = jitter.detach().cpu().numpy()
    jitter_ms = (time.perf_counter() - started) * 1000.0
    if len(jitter) != 1_080_000 or not np.isfinite(jitter).all():
        raise RuntimeError("Native jitter did not emit exactly 1,080,000 finite points")
    return centers, jitter, center_ms, jitter_ms, meta


def _write_rows(output, rows, summary):
    import csv
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_frame.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    if flat:
        with (output / "per_frame.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    atomic_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "mean", "std", "median", "p90", "p95", "min", "max"])
        writer.writeheader()
        for metric, stats in summary["statistics"].items():
            writer.writerow({"model": summary["model"], "metric": metric, **stats})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("hs_occflow", "patchscene"), required=True)
    p.add_argument("--physical-gpu", choices=(0, 2, 3), type=int, default=2)
    p.add_argument("--allow-idle-gpu0-service-context", action="store_true",
                   help="Allow GPU0's observed <=512 MiB idle service context when utilization is <=1%%; never terminates it")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--stage2-checkpoint", type=Path)
    p.add_argument("--patchscene-root", type=Path)
    p.add_argument("--patchscene-checkpoint", type=Path)
    p.add_argument("--patchscene-steps", type=int, default=10, choices=(10, 15))
    p.add_argument("--warmup-frames", nargs="+", type=int, default=WARMUP_FRAMES)
    p.add_argument("--benchmark-frames", nargs="+", type=int, default=BENCHMARK_FRAMES)
    p.add_argument("--seed-base", type=int, default=20261019)
    p.add_argument("--no-save-predictions", action="store_true")
    p.add_argument("--skip-jitter", action="store_true",
                   help="Quality-only run: save native occupancy/centers without 1.08M-point jitter")
    p.add_argument("--allow-custom-frame-counts", action="store_true",
                   help="Allow a separate quality set; not a registered 5+20 latency benchmark")
    p.add_argument("--gpu-candidate-selection", action="store_true")
    p.add_argument("--gpu-observed-mask", action="store_true")
    p.add_argument("--gpu-occupancy-decode", action="store_true")
    p.add_argument("--stage2-static-cache", action="store_true")
    p.add_argument("--stage1-flow-condition-cache", action="store_true",
                   help="V4 inference-only cache of the three fixed Stage1 condition projections")
    p.add_argument("--stage1-encoder-layout", choices=("original", "channels_last_3d"), default="original")
    p.add_argument("--stage1-flow-layout", choices=("original", "channels_last_3d"), default="original")
    p.add_argument("--stage1-encoder-compile-target",
                   choices=("none", "voxel_backbone", "voxel_neck", "local_backbone", "classification_neck"),
                   default="none", help="V5 opt-in compile of one dense Encoder submodule")
    p.add_argument("--stage1-encoder-precision", choices=("amp_original", "fp16_full", "fp16_selective_norm"), default="amp_original")
    p.add_argument("--stage1-flow-precision", choices=("amp_original", "fp16_full", "fp16_selective_norm"), default="amp_original")
    p.add_argument("--stage2-weight-precision", choices=("amp_original", "fp16_full", "fp16_selective_norm"), default="amp_original")
    p.add_argument("--stage1-cudnn-benchmark", action="store_true")
    p.add_argument("--stage1-tf32", action="store_true",
                   help="Enable cuDNN and matmul TF32 flags (opt-in numeric variant)")
    p.add_argument("--ptv3-topology-cache", choices=("none", "serialization", "multilevel"), default="none",
                   help="V3 inference-only cache: top-level serialization or serialization+attention+pooling maps")
    p.add_argument("--ptv3-precision", choices=("original", "amp_fp16", "selective_fp16"), default="original",
                   help="V3 opt-in PTv3 precision policy; Euler state remains FP32")
    p.add_argument("--ptv3-batch1-fastpath", action="store_true",
                   help="Use validated single-scene norm/sparsify metadata fast paths")
    p.add_argument("--compile-target", choices=("none", "stage2_mlp", "stage1_flow"), default="none",
                   help="V3 opt-in torch.compile target; compile cold-start is recorded on first warmup")
    p.add_argument("--stage1-execution-mode", choices=("inference_mode", "no_grad"), default="inference_mode",
                   help="Use no_grad only for the Stage1 torch.compile compatibility experiment")
    p.add_argument("--dtype-audit", action="store_true",
                   help="Diagnostic only: record leaf-module input/output dtypes; do not use timings")
    p.add_argument("--jitter-backend", choices=("cpu_reference", "cuda"), default="cpu_reference")
    p.add_argument("--parity-reference-dir", type=Path,
                   help="Optional directory containing baseline frame_XXXXXX.npz predictions")
    p.add_argument("--quality-reference-dir", type=Path,
                   help="Optional numerical-variant reference; records score/candidate/occupancy deltas without enforcing exact parity")
    p.add_argument("--smoke-one-frame", action="store_true",
                   help="Engineering smoke only: run frame 100 once, without warmup; never a benchmark result")
    a = p.parse_args()
    if a.smoke_one_frame:
        a.warmup_frames, a.benchmark_frames = [], [100]
    if not a.smoke_one_frame and not a.allow_custom_frame_counts and (len(a.warmup_frames) != 5 or len(a.benchmark_frames) != 20):
        p.error("This registered run requires exactly 5 warmup and 20 timed frames")
    expected_unique = 1 if a.smoke_one_frame else len(a.warmup_frames) + len(a.benchmark_frames)
    if len(set(a.warmup_frames + a.benchmark_frames)) != expected_unique:
        p.error("Warmup and benchmark frame IDs must be pairwise distinct")
    env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_gpu != str(a.physical_gpu):
        p.error("CUDA_VISIBLE_DEVICES must contain only the selected physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    admission = _admit(a.physical_gpu, allow_idle_service_context=a.allow_idle_gpu0_service_context)
    torch.backends.cudnn.benchmark = bool(a.stage1_cudnn_benchmark)
    torch.backends.cudnn.allow_tf32 = bool(a.stage1_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(a.stage1_tf32)
    inputs = {}
    for frame in a.warmup_frames + a.benchmark_frames:
        path = a.data_root / "08" / "input_" / f"{frame:06d}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = input_xyz(path).astype(np.float32, copy=False)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite input {path}")
        inputs[frame] = {"path": path, "xyz": arr, "input_file_sha256": file_sha(path),
                         "input_xyz_sha256": _sha_array(arr), "input_point_count": int(len(arr))}
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Benchmark output must be new and empty: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": a.model, "physical_gpu": a.physical_gpu, "batch_size": 1,
                "warmup_frames": a.warmup_frames, "benchmark_frames": a.benchmark_frames,
                "all_frames_unique": True, "input_hashes": {str(f): {k: inputs[f][k] for k in
                    ("input_file_sha256", "input_xyz_sha256", "input_point_count")} for f in inputs},
                "point_seed_base": POINT_SEED_BASE, "target_jitter_points": 1_080_000,
                "input_disk_read_excluded": True, "metrics_and_file_writes_excluded": True,
                "admission": admission, "python_torch": torch.__version__, "torch_cuda": torch.version.cuda}
    rows = []
    if a.model == "hs_occflow":
        if not a.config or not a.stage2_checkpoint:
            p.error("HS run requires --config and --stage2-checkpoint")
        cfg, geom = load_stage2_config(a.config)
        if file_sha(a.stage2_checkpoint) != S2_SHA:
            raise ValueError("Stage2 checkpoint SHA mismatch")
        if cfg["stage1"]["adapter"]["checkpoint_sha256"] != S1_SHA:
            raise ValueError("Stage1 checkpoint identity mismatch")
        saved = torch.load(a.stage2_checkpoint, map_location="cpu")
        if saved["successful_updates"] != 25_000 or saved["sampler"]["successful_frame_exposures"] != 100_000:
            raise ValueError("Unexpected Stage2 training state")
        adapter = FrozenStage1OccupancyAdapter(**cfg["stage1"]["adapter"]).to(device).eval()
        from fpsgen.hier_occ_flow.full_fp16_inference import explicit_fp16_copy, parameter_dtype_audit
        precision_audits = {}
        precision_hooks = []
        if a.stage1_encoder_precision != "amp_original":
            precision_audits["stage1_encoder"], hooks = explicit_fp16_copy(
                adapter.model.encoder, selective_norm=a.stage1_encoder_precision == "fp16_selective_norm")
            precision_hooks.extend(hooks)
            adapter.model.encoder._v6_explicit_fp16 = True
        if a.stage1_flow_precision != "amp_original":
            precision_audits["stage1_flow"], hooks = explicit_fp16_copy(
                adapter.model.flow, selective_norm=a.stage1_flow_precision == "fp16_selective_norm")
            precision_hooks.extend(hooks)
            adapter.model.flow._v6_explicit_fp16 = True
        stage1_layout_handles = []
        if a.stage1_encoder_layout == "channels_last_3d":
            stage1_layout_handles.extend(enable_channels_last_3d(adapter.model.encoder))
        if a.stage1_flow_layout == "channels_last_3d":
            stage1_layout_handles.extend(enable_channels_last_3d(adapter.model.flow))
        model = Stage2OccupancyModel(geom, **cfg["model"]).to(device).eval()
        model.load_state_dict(saved["model"], strict=True)
        if a.stage2_weight_precision != "amp_original":
            precision_audits["stage2"], hooks = explicit_fp16_copy(
                model, selective_norm=a.stage2_weight_precision == "fp16_selective_norm")
            precision_hooks.extend(hooks)
            model.set_ptv3_precision("explicit_fp16")
        dtype_audit = {}
        dtype_hook_handles = []
        if a.dtype_audit:
            def dtype_name(value):
                if isinstance(value, torch.Tensor): return str(value.dtype)
                if hasattr(value, "feat") and isinstance(value.feat, torch.Tensor): return str(value.feat.dtype)
                if isinstance(value, (tuple, list)):
                    return "[" + ",".join(dtype_name(v) for v in value[:4]) + (",..." if len(value)>4 else "") + "]"
                if isinstance(value, dict): return "{" + ",".join(f"{k}:{dtype_name(v)}" for k,v in list(value.items())[:4]) + "}"
                return type(value).__name__
            for prefix, root_module in (("stage1_encoder", adapter.model.encoder),
                                        ("stage1_flow", adapter.model.flow),
                                        ("stage2_ptv3", model.trunk.net)):
                for module_name, module in root_module.named_modules():
                    if module_name and any(module.children()): continue
                    key = f"{prefix}.{module_name or '<root>'}:{module.__class__.__name__}"
                    dtype_audit[key] = {"input": set(), "output": set()}
                    def record_dtype(mod, args, output, _key=key):
                        dtype_audit[_key]["input"].add(dtype_name(args))
                        dtype_audit[_key]["output"].add(dtype_name(output))
                    dtype_hook_handles.append(module.register_forward_hook(record_dtype))
        model.ptv3_topology_cache = a.ptv3_topology_cache
        model.set_ptv3_precision("explicit_fp16" if a.stage2_weight_precision != "amp_original"
                                  else a.ptv3_precision)
        if a.stage2_weight_precision != "amp_original":
            model.set_explicit_fp16_norm_policy(a.stage2_weight_precision == "fp16_selective_norm")
        model.ptv3_batch1_fastpath = a.ptv3_batch1_fastpath
        compile_seconds = 0.0
        if a.compile_target != "none":
            from torch._dynamo.utils import counters
            counters.clear()
        if a.compile_target == "stage1_flow":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build")
            compile_start = time.perf_counter()
            adapter.model.flow = torch.compile(adapter.model.flow, dynamic=False)
            compile_seconds = time.perf_counter() - compile_start
        elif a.compile_target == "stage2_mlp":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build")
            compile_start = time.perf_counter()
            for name in ("local", "time", "fuse", "decoder"):
                setattr(model, name, torch.compile(getattr(model, name), dynamic=True))
            compile_seconds = time.perf_counter() - compile_start
        encoder_compile_setup_seconds = 0.0
        if a.stage1_encoder_compile_target != "none":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build")
            encoder = adapter.model.encoder
            target = a.stage1_encoder_compile_target
            compile_start = time.perf_counter()
            if target == "classification_neck":
                encoder._classification_neck = torch.compile(encoder._classification_neck, dynamic=False)
            else:
                setattr(encoder, target, torch.compile(getattr(encoder, target), dynamic=False))
            encoder_compile_setup_seconds = time.perf_counter() - compile_start
        protocol.update(stage1_checkpoint_sha256=S1_SHA, stage1_weights="ema", stage1_nfe=2,
                        stage1_checkpoint=cfg["stage1"]["adapter"]["checkpoint"],
                        stage2_checkpoint_sha256=S2_SHA, stage2_weights="online",
                        stage2_nfe=4, stage2_checkpoint=str(a.stage2_checkpoint.resolve()),
                        candidate_selection="predicted Stage1 threshold .5; original policy",
                        amp="per-module; explicit modules run with autocast disabled",
                        stage1_encoder_precision=a.stage1_encoder_precision,
                        stage1_flow_precision=a.stage1_flow_precision,
                        stage2_weight_precision=a.stage2_weight_precision,
                        stage2_condition_encoding="same single Stage1 feature encode",
                        stage1_flow_condition_cache=a.stage1_flow_condition_cache,
                        stage1_encoder_layout=a.stage1_encoder_layout,
                        stage1_flow_layout=a.stage1_flow_layout,
                        stage1_encoder_compile_target=a.stage1_encoder_compile_target,
                        stage1_encoder_compile_wrapper_setup_seconds=encoder_compile_setup_seconds,
                        stage1_cudnn_benchmark=a.stage1_cudnn_benchmark,
                        stage1_tf32=a.stage1_tf32,
                        stage1_execution_mode=a.stage1_execution_mode,
                        ptv3_topology_cache=a.ptv3_topology_cache, ptv3_precision=a.ptv3_precision,
                        compile_target=a.compile_target, compile_wrapper_setup_seconds=compile_seconds)
        protocol.update(explicit_precision_audit=precision_audits,
                        parameter_dtype_audits={
                            "stage1_encoder": parameter_dtype_audit(adapter.model.encoder),
                            "stage1_flow": parameter_dtype_audit(adapter.model.flow),
                            "stage2": parameter_dtype_audit(model),
                        })
        protocol["dtype_audit_enabled"] = bool(a.dtype_audit)
        protocol["optimization_flags"] = {
            "gpu_candidate_selection": a.gpu_candidate_selection,
            "gpu_observed_mask": a.gpu_observed_mask,
            "gpu_occupancy_decode": a.gpu_occupancy_decode,
            "stage2_static_cache": a.stage2_static_cache,
            "stage1_flow_condition_cache": a.stage1_flow_condition_cache,
            "stage1_encoder_layout": a.stage1_encoder_layout,
            "stage1_flow_layout": a.stage1_flow_layout,
            "stage1_cudnn_benchmark": a.stage1_cudnn_benchmark,
            "stage1_tf32": a.stage1_tf32,
            "jitter_backend": a.jitter_backend,
            "parity_reference_dir": str(a.parity_reference_dir.resolve()) if a.parity_reference_dir else None,
            "quality_reference_dir": str(a.quality_reference_dir.resolve()) if a.quality_reference_dir else None,
        }
        save_dir = a.output / "predictions"
        parity_rows = {}
        reference_dir = a.parity_reference_dir or a.quality_reference_dir
        if reference_dir:
            reference_run = reference_dir.parent.parent
            reference_jsonl = reference_run / "per_frame.jsonl"
            if reference_jsonl.is_file():
                parity_rows = {int(record["frame"]): record for record in
                               (json.loads(line) for line in reference_jsonl.read_text().splitlines() if line)}
        pending_outputs = []
        protocol["geometry_sha256"] = geom.sha256
        protocol["model_parameter_forwards"] = "encoder once; Stage1 flow twice; Stage2 model four times"
        compile_first_call = a.compile_target != "none" or a.stage1_encoder_compile_target != "none"
        for warmup, frame in [(True, f) for f in a.warmup_frames] + [(False, f) for f in a.benchmark_frames]:
            call_start = time.perf_counter() if compile_first_call else None
            execution_context = (torch.no_grad() if a.stage1_execution_mode == "no_grad"
                                 else torch.inference_mode())
            with execution_context:
                fine, timing = _run_hs_frame(
                    inputs[frame]["xyz"], frame, adapter, model, cfg, geom, device,
                    gpu_candidate_selection=a.gpu_candidate_selection,
                    gpu_observed_mask=a.gpu_observed_mask,
                    gpu_occupancy_decode=a.gpu_occupancy_decode,
                    stage2_static_cache=a.stage2_static_cache,
                    stage1_flow_condition_cache=a.stage1_flow_condition_cache)
            if call_start is not None:
                torch.cuda.synchronize(device)
                protocol["compile_first_call_wall_ms"] = (time.perf_counter() - call_start) * 1000.0
                protocol["compile_first_call_frame"] = frame
                protocol["compile_first_call_is_warmup"] = bool(warmup)
                compile_first_call = False
            if a.skip_jitter:
                t0 = time.perf_counter()
                centers, _ = sample_native_occupied_voxels(fine, geom, point_mode="voxel_center")
                center_ms = (time.perf_counter()-t0)*1000.0
                jitter = np.empty((0,3),dtype=np.float32); jitter_ms = 0.0
                jitter_meta = {"original_occupied_voxels": int(len(fine)), "skipped_for_quality_run": True}
            else:
                centers, jitter, center_ms, jitter_ms, jitter_meta = _postprocess(
                    fine, geom, frame, jitter_backend=a.jitter_backend, device=device)
            timing.update({"model": a.model, "frame": frame, "warmup": warmup, "batch_size": 1,
                           "voxel_center_ms": center_ms, "native_jitter_ms": jitter_ms,
                           "generation_plus_jitter_ms": timing["generation_wall_ms"] + jitter_ms,
                           "jitter_output_points": int(len(jitter)),
                           "jitter_backend": a.jitter_backend,
                           "jitter_occupied_voxels": jitter_meta["original_occupied_voxels"],
                           "jitter_cuda_fallback_used": jitter_meta.get("cuda_fallback_used"),
                           "jitter_cuda_fallback_points": jitter_meta.get("cuda_fallback_pending_proposals"),
                           "jitter_cuda_proposals": jitter_meta.get("cuda_proposals"),
                           "jitter_cuda_rejection_rounds": jitter_meta.get("cuda_rejection_rounds"),
                           "jitter_seed": int(POINT_SEED_BASE + frame), **inputs[frame]})
            if reference_dir and not warmup:
                ref_path = reference_dir / f"frame_{frame:06d}.npz"
                if not ref_path.is_file():
                    raise FileNotFoundError(f"Missing parity reference for frame {frame}: {ref_path}")
                with np.load(ref_path) as ref:
                    ref_coords = ref["occupied_coords_xyz"]
                exact = np.array_equal(fine, ref_coords)
                timing["parity_reference"] = str(ref_path.resolve())
                timing["parity_exact"] = bool(exact)
                timing["parity_reference_occupied_voxels"] = int(len(ref_coords))
                nx, ny, nz = geom.fine_resolution
                ref_ids = np.sort(ref_coords[:, 0].astype(np.int64) + nx * ref_coords[:, 1].astype(np.int64)
                                  + nx * ny * ref_coords[:, 2].astype(np.int64))
                cur_ids = np.sort(fine[:, 0].astype(np.int64) + nx * fine[:, 1].astype(np.int64)
                                  + nx * ny * fine[:, 2].astype(np.int64))
                overlap = int(np.intersect1d(ref_ids, cur_ids, assume_unique=True).size)
                timing["occupancy_symmetric_difference_voxels"] = int(len(ref_ids) + len(cur_ids) - 2 * overlap)
                reference_row = parity_rows.get(frame)
                if reference_row is not None:
                    timing["stage1_score_parity_exact"] = timing.get("stage1_score_sha256") == reference_row.get("stage1_score_sha256")
                    timing["stage1_candidate_ids_parity_exact"] = timing.get("stage1_candidate_ids_sha256") == reference_row.get("stage1_candidate_ids_sha256")
                    timing["candidate_parent_count_delta"] = int(timing.get("candidate_parents", 0) - reference_row.get("candidate_parents", 0))
                    if a.parity_reference_dir and not (timing["stage1_score_parity_exact"]
                                                       and timing["stage1_candidate_ids_parity_exact"]
                                                       and timing["candidate_parent_count_delta"] == 0):
                        raise RuntimeError(f"Stage1 parity failed at frame {frame}")
                if not exact and a.parity_reference_dir:
                    raise RuntimeError(f"Fine occupancy parity failed at frame {frame}")
            # Keep JSON records lean; arrays and paths remain in input identity.
            timing.pop("xyz", None); timing.pop("path", None)
            if not warmup:
                rows.append(timing)
                if not a.no_save_predictions:
                    pending_outputs.append((frame, inputs[frame]["xyz"], fine.copy(), geom,
                                            None if a.skip_jitter else jitter.copy(), jitter_meta))
            print(f"HS {'warmup' if warmup else 'timed'} frame={frame} gen={timing['generation_wall_ms']:.2f}ms "
                  f"net={timing['network_forward_gpu_ms']:.2f}ms center={center_ms:.2f}ms jitter={jitter_ms:.2f}ms", flush=True)
        expected_encoder_runs = len(a.warmup_frames) + len(a.benchmark_frames)
        if adapter.encoder_forward_count != expected_encoder_runs:
            raise RuntimeError(f"Expected {expected_encoder_runs} encoder runs, got {adapter.encoder_forward_count}")
        for frame, xyz_np, fine_np, output_geom, jitter_np, jitter_meta in pending_outputs:
            _dump_outputs(save_dir, frame, xyz_np, fine_np, output_geom, jitter_np, jitter_meta)
        for handle in dtype_hook_handles: handle.remove()
        remove_layout_hooks(stage1_layout_handles)
        if a.dtype_audit:
            protocol["module_dtype_audit"] = {
                name: {key: sorted(values) for key, values in record.items()}
                for name, record in dtype_audit.items()
            }
        if a.compile_target != "none":
            from torch._dynamo.utils import counters
            protocol["dynamo_counters"] = {group: dict(values) for group, values in counters.items()}
    else:
        if not a.patchscene_root or not a.patchscene_checkpoint:
            p.error("PatchScene run requires --patchscene-root and --patchscene-checkpoint")
        if file_sha(a.patchscene_checkpoint) != PS_SHA:
            raise ValueError("PatchScene official checkpoint SHA mismatch")
        root = a.patchscene_root.resolve(); sys.path.insert(0, str(root))
        try:
            import ipdb  # noqa: F401
        except ImportError:
            import types
            ipdb = types.ModuleType("ipdb"); ipdb.set_trace = lambda *args, **kwargs: None
            sys.modules["ipdb"] = ipdb
        from network.model_trainer import DiffusionModel
        from patchscene_infer_seq08 import _make_inference_grid
        from test_semantickitti import PRE_DEFINE
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if commit != PS_COMMIT:
            raise ValueError(f"PatchScene source commit {commit} != registered {PS_COMMIT}")
        diffusion = DiffusionModel.load_from_checkpoint(str(a.patchscene_checkpoint), map_location="cpu", strict=True).eval()
        generator = diffusion.ema_model.eval().to(device)
        grid = _make_inference_grid()
        ring_count = len(PRE_DEFINE)
        protocol.update(patchscene_commit=commit, patchscene_checkpoint_sha256=PS_SHA,
                        patchscene_weights="EMA", generate_method="Ring_Spread_V2",
                        patchscene_steps=a.patchscene_steps, number_of_rings=ring_count,
                        expected_unet_forward_calls_per_frame=ring_count * a.patchscene_steps,
                        temporal_context="none; per-frame independent Ring_Spread_V2")
        native_geom = _PatchGeometry()
        pending_outputs = []
        for warmup, frame in [(True, f) for f in a.warmup_frames] + [(False, f) for f in a.benchmark_frames]:
            torch.manual_seed(a.seed_base + frame); torch.cuda.manual_seed_all(a.seed_base + frame)
            fine, timing = _run_patchscene_frame(inputs[frame]["xyz"], frame, generator, grid, a.patchscene_steps, device)
            centers, jitter, center_ms, jitter_ms, jitter_meta = _postprocess(
                fine, native_geom, frame, jitter_backend=a.jitter_backend, device=device)
            timing.update({"model": a.model, "frame": frame, "warmup": warmup, "batch_size": 1,
                           "voxel_center_ms": center_ms, "native_jitter_ms": jitter_ms,
                           "generation_plus_jitter_ms": timing["generation_wall_ms"] + jitter_ms,
                           "jitter_output_points": int(len(jitter)),
                           "jitter_backend": a.jitter_backend,
                           "jitter_occupied_voxels": jitter_meta["original_occupied_voxels"],
                           "jitter_cuda_fallback_used": jitter_meta.get("cuda_fallback_used"),
                           "jitter_cuda_fallback_points": jitter_meta.get("cuda_fallback_pending_proposals"),
                           "jitter_cuda_proposals": jitter_meta.get("cuda_proposals"),
                           "jitter_cuda_rejection_rounds": jitter_meta.get("cuda_rejection_rounds"),
                           "jitter_seed": int(POINT_SEED_BASE + frame), **inputs[frame]})
            timing.pop("xyz", None); timing.pop("path", None)
            if not warmup:
                rows.append(timing)
                if not a.no_save_predictions:
                    pending_outputs.append((frame, inputs[frame]["xyz"], fine.copy(), native_geom, jitter.copy(), jitter_meta))
            print(f"PatchScene {'warmup' if warmup else 'timed'} frame={frame} gen={timing['generation_wall_ms']:.2f}ms "
                  f"net={timing['network_forward_gpu_ms']:.2f}ms calls={timing['unet_forward_calls']} "
                  f"center={center_ms:.2f}ms jitter={jitter_ms:.2f}ms", flush=True)
        for frame, xyz_np, fine_np, output_geom, jitter_np, jitter_meta in pending_outputs:
            _dump_outputs(a.output, frame, xyz_np, fine_np, output_geom, jitter_np, jitter_meta)
    metric_keys = sorted(k for k, v in rows[0].items() if isinstance(v, (int, float)) and k.endswith(("_ms", "_gib")))
    summary = {"model": a.model, "status": "SMOKE_ONLY" if a.smoke_one_frame else "COMPLETE",
               "timed_frames": len(rows), "warmup_frames": a.warmup_frames,
               "benchmark_frames": a.benchmark_frames, "statistics": {k: summarize([r[k] for r in rows]) for k in metric_keys},
               "network_forward_calls_total": int(sum(r.get("stage1_flow_forward_calls", 0) + r.get("stage2_forward_calls", 0) + r.get("unet_forward_calls", 0) for r in rows)),
               "network_forward_calls_per_frame": float(np.mean([r.get("stage1_flow_forward_calls", 0) + r.get("stage2_forward_calls", 0) + r.get("unet_forward_calls", 0) for r in rows]))}
    protocol.update(status="COMPLETE", timed_frames=len(rows),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    code_sha256=file_sha(Path(__file__)),
                    stage1_network_sha256=file_sha(ROOT / "fpsgen/hier_occ_flow/stage1_network.py"),
                    stage1_occupancy_sha256=file_sha(ROOT / "fpsgen/hier_occ_flow/stage1_occupancy.py"),
                    timing_util_sha256=file_sha(HERE / "inference_timing_utils.py"))
    atomic_json(a.output / "protocol.json", protocol)
    atomic_json(a.output / "admission.json", admission)
    _write_rows(a.output, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

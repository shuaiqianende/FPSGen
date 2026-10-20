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
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.condition import sample_pyramid, observed_children
from fpsgen.hier_occ_flow.inference import sample as sample_stage2
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords, pack_children
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from native_voxel_point_sampling import sample_native_occupied_voxels
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


def _admit(gpu):
    q = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=name,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    p = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    processes = [x.strip() for x in p.splitlines() if x.strip()]
    info = {"physical_gpu": gpu, "name": q[0].strip(), "memory_free_mib": int(q[1]),
            "memory_total_mib": int(q[2]), "utilization_percent": int(q[3]),
            "temperature_c": int(q[4]), "power_w": float(q[5]), "sm_clock_mhz": int(q[6]),
            "compute_processes": processes}
    if processes or info["memory_free_mib"] < 18000 or info["utilization_percent"] > 10:
        raise RuntimeError(f"GPU{gpu} admission rejected: {info}")
    return info


def _dump_outputs(out_dir, frame, xyz, fine_coords, geometry, jitter, jitter_meta):
    out = Path(out_dir) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    centers = geometry.centers(fine_coords, fine=True).astype(np.float32)
    np.savez_compressed(out / f"frame_{frame:06d}.npz",
                        occupied_coords_xyz=np.asarray(fine_coords, dtype=np.int32),
                        voxel_center_xyz=centers, jitter_xyz=jitter.astype(np.float32),
                        input_xyz_sha256=_sha_array(xyz), jitter_metadata=json.dumps(jitter_meta))


@torch.inference_mode()
def _run_hs_frame(xyz_np, frame, adapter, model, cfg, geom, device):
    # Every feature/flow/network event is queued asynchronously and synchronized
    # once after generation, so the measured path retains normal launch behavior.
    torch.cuda.reset_peak_memory_stats(device)
    baseline_alloc = torch.cuda.memory_allocated(device)
    wall0 = time.perf_counter()
    xyz = torch.from_numpy(np.asarray(xyz_np, dtype=np.float32)).to(device)
    encoder_value, encoder_events = timed_cuda_call(lambda: adapter.encode_features(xyz))
    features = encoder_value
    flow_timer = ModuleCudaTimer(adapter.model.flow)
    seed = scene_seed("08", frame, cfg["candidates"]["seed"])
    with flow_timer, torch.autocast("cuda", dtype=torch.float16):
        score, _state1 = adapter.model.sample(
            xyz, nfe=2, condition=features,
            generator=torch.Generator(device=device).manual_seed(seed))
    predicted_policy = __import__("export_stage2_snapshot").predicted_policy
    policy = predicted_policy(cfg)
    policy["stage1_nfe"] = 2
    selection_started = time.perf_counter()
    ids_np, scores_np, provenance = select_occupancy_candidates(
        score[0, 0].float().cpu().numpy(), geom, **policy["selection"])
    candidate_selection_ms = (time.perf_counter() - selection_started) * 1000.0
    ids = torch.as_tensor(ids_np, device=device)
    condition_started = time.perf_counter()
    centers_xyz = geom.centers(geom.unflatten(ids)).float()
    cond = {
        "lidar_features": sample_pyramid(features, centers_xyz, geom.origin, geom.maximum).detach().half(),
        "parent_score": torch.as_tensor(scores_np, device=device),
        "observed_child": torch.as_tensor(observed_children(xyz, ids_np, geom), device=device),
        "valid_child_mask": torch.as_tensor(geom.valid_children(ids_np), device=device),
    }
    if cond["lidar_features"].shape != (len(ids), 384):
        raise RuntimeError("Stage2 condition sampling shape mismatch")
    stage2_condition_ms = (time.perf_counter() - condition_started) * 1000.0
    noise = torch.randn((geom.num_parents, geom.k), device=device,
                        generator=torch.Generator(device=device).manual_seed(cfg["validation"]["seed"] + frame))
    source = noise[ids] * cond["valid_child_mask"]
    del noise
    s2_timer = ModuleCudaTimer(model)
    state = source
    with s2_timer, torch.autocast("cuda", dtype=torch.float16):
        for step in range(4):
            t = state.new_tensor(step / 4)
            velocity = model(ids, state=state, time=t, **cond)
            state = state + velocity.float() / 4
    if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()):
        raise RuntimeError(f"Non-finite HS endpoint at frame {frame}")
    decode_started = time.perf_counter()
    occupancy = (state > 0) & cond["valid_child_mask"]
    fine = occupied_fine_coords(ids.cpu().numpy(),
                                pack_children(occupancy.cpu().numpy(), geom), geom)
    if not len(fine):
        raise RuntimeError(f"Empty HS occupancy at frame {frame}")
    decode_wall_ms = (time.perf_counter() - decode_started) * 1000.0
    torch.cuda.synchronize(device)
    generation_wall_ms = (time.perf_counter() - wall0) * 1000.0
    encoder_ms = event_elapsed_ms([encoder_events])
    flow_ms = flow_timer.elapsed_ms()
    stage2_ms = s2_timer.elapsed_ms()
    row = {
        "stage1_encoder_ms": encoder_ms,
        "stage1_flow_total_ms": flow_ms,
        "stage1_flow_ms_per_step": flow_ms / 2,
        "stage1_flow_forward_calls": flow_timer.calls,
        "candidate_selection_ms": candidate_selection_ms,
        "stage2_condition_ms": stage2_condition_ms,
        "stage2_network_total_ms": stage2_ms,
        "stage2_network_ms_per_step": stage2_ms / 4,
        "stage2_forward_calls": s2_timer.calls,
        "occupancy_decode_ms": decode_wall_ms,
        "network_forward_gpu_ms": encoder_ms + flow_ms + stage2_ms,
        "generation_wall_ms": generation_wall_ms,
        "occupied_voxels": int(len(fine)),
        "candidate_parents": int(len(ids)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "incremental_peak_allocated_gib": max(0, torch.cuda.max_memory_allocated(device) - baseline_alloc) / 1024**3,
        "provenance": provenance,
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


def _postprocess(fine, geometry, frame):
    started = time.perf_counter()
    centers, _ = sample_native_occupied_voxels(fine, geometry, point_mode="voxel_center")
    center_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    jitter, meta = sample_native_occupied_voxels(
        fine, geometry, point_mode="native_fine_jitter", target_points=1_080_000,
        seed=POINT_SEED_BASE + frame)
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
    p.add_argument("--physical-gpu", choices=(2, 3), type=int, default=2)
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
    p.add_argument("--smoke-one-frame", action="store_true",
                   help="Engineering smoke only: run frame 100 once, without warmup; never a benchmark result")
    a = p.parse_args()
    if a.smoke_one_frame:
        a.warmup_frames, a.benchmark_frames = [], [100]
    if not a.smoke_one_frame and (len(a.warmup_frames) != 5 or len(a.benchmark_frames) != 20):
        p.error("This registered run requires exactly 5 warmup and 20 timed frames")
    expected_unique = 1 if a.smoke_one_frame else 25
    if len(set(a.warmup_frames + a.benchmark_frames)) != expected_unique:
        p.error("Warmup and benchmark frame IDs must be pairwise distinct")
    env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_gpu != str(a.physical_gpu):
        p.error("CUDA_VISIBLE_DEVICES must contain only the selected physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    admission = _admit(a.physical_gpu)
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
        model = Stage2OccupancyModel(geom, **cfg["model"]).to(device).eval()
        model.load_state_dict(saved["model"], strict=True)
        protocol.update(stage1_checkpoint_sha256=S1_SHA, stage1_weights="ema", stage1_nfe=2,
                        stage1_checkpoint=cfg["stage1"]["adapter"]["checkpoint"],
                        stage2_checkpoint_sha256=S2_SHA, stage2_weights="online",
                        stage2_nfe=4, stage2_checkpoint=str(a.stage2_checkpoint.resolve()),
                        candidate_selection="predicted Stage1 threshold .5; original policy",
                        amp="original fp16 autocast", stage2_condition_encoding="same single Stage1 feature encode")
        save_dir = a.output / "predictions"
        pending_outputs = []
        protocol["geometry_sha256"] = geom.sha256
        protocol["model_parameter_forwards"] = "encoder once; Stage1 flow twice; Stage2 model four times"
        for warmup, frame in [(True, f) for f in a.warmup_frames] + [(False, f) for f in a.benchmark_frames]:
            fine, timing = _run_hs_frame(inputs[frame]["xyz"], frame, adapter, model, cfg, geom, device)
            centers, jitter, center_ms, jitter_ms, jitter_meta = _postprocess(fine, geom, frame)
            timing.update({"model": a.model, "frame": frame, "warmup": warmup, "batch_size": 1,
                           "voxel_center_ms": center_ms, "native_jitter_ms": jitter_ms,
                           "generation_plus_jitter_ms": timing["generation_wall_ms"] + jitter_ms,
                           "jitter_output_points": int(len(jitter)),
                           "jitter_occupied_voxels": jitter_meta["original_occupied_voxels"],
                           "jitter_seed": int(POINT_SEED_BASE + frame), **inputs[frame]})
            # Keep JSON records lean; arrays and paths remain in input identity.
            timing.pop("xyz", None); timing.pop("path", None)
            if not warmup:
                rows.append(timing)
                if not a.no_save_predictions:
                    pending_outputs.append((frame, inputs[frame]["xyz"], fine.copy(), geom, jitter.copy(), jitter_meta))
            print(f"HS {'warmup' if warmup else 'timed'} frame={frame} gen={timing['generation_wall_ms']:.2f}ms "
                  f"net={timing['network_forward_gpu_ms']:.2f}ms center={center_ms:.2f}ms jitter={jitter_ms:.2f}ms", flush=True)
        expected_encoder_runs = len(a.warmup_frames) + len(a.benchmark_frames)
        if adapter.encoder_forward_count != expected_encoder_runs:
            raise RuntimeError(f"Expected {expected_encoder_runs} encoder runs, got {adapter.encoder_forward_count}")
        for frame, xyz_np, fine_np, output_geom, jitter_np, jitter_meta in pending_outputs:
            _dump_outputs(save_dir, frame, xyz_np, fine_np, output_geom, jitter_np, jitter_meta)
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
            centers, jitter, center_ms, jitter_ms, jitter_meta = _postprocess(fine, native_geom, frame)
            timing.update({"model": a.model, "frame": frame, "warmup": warmup, "batch_size": 1,
                           "voxel_center_ms": center_ms, "native_jitter_ms": jitter_ms,
                           "generation_plus_jitter_ms": timing["generation_wall_ms"] + jitter_ms,
                           "jitter_output_points": int(len(jitter)),
                           "jitter_occupied_voxels": jitter_meta["original_occupied_voxels"],
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
    protocol.update(status="COMPLETE", timed_frames=len(rows), code_sha256=file_sha(Path(__file__)),
                    timing_util_sha256=file_sha(HERE / "inference_timing_utils.py"))
    atomic_json(a.output / "protocol.json", protocol)
    atomic_json(a.output / "admission.json", admission)
    _write_rows(a.output, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

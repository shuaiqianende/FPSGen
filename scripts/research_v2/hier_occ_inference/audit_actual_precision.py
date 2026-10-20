#!/usr/bin/env python3
"""Runtime dtype audit for the registered HS-OccFlow 2+4 inference path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.condition import sample_pyramid, observed_children


def _dtype_tree(value):
    found = []
    if isinstance(value, torch.Tensor):
        found.append(str(value.dtype))
    elif hasattr(value, "feat") and isinstance(value.feat, torch.Tensor):
        found.append(str(value.feat.dtype))
    elif isinstance(value, (tuple, list)):
        for x in value:
            found.extend(_dtype_tree(x))
    elif isinstance(value, dict):
        for x in value.values():
            found.extend(_dtype_tree(x))
    return found


def _install_hooks(module, prefix, records):
    handles = []
    for name, child in module.named_modules():
        if not name or any(True for _ in child.children()):
            continue
        def hook(mod, inputs, output, name=name):
            input_types = sorted(set(_dtype_tree(inputs)))
            output_types = sorted(set(_dtype_tree(output)))
            weight = getattr(mod, "weight", None)
            records.append({
                "module": f"{prefix}.{name}",
                "class": type(mod).__name__,
                "weight_dtype": str(weight.dtype) if isinstance(weight, torch.Tensor) else None,
                "input_dtypes": input_types,
                "output_dtypes": output_types,
                "cuda_autocast_enabled_at_hook": bool(torch.is_autocast_enabled()),
            })
        handles.append(child.register_forward_hook(hook))
    return handles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu", choices=(2, 3), type=int, default=2)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage2-checkpoint", type=Path, required=True)
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--encoder-precision", choices=("original", "force_fp32", "autocast_fp16"), default="original")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose only the selected physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runtime precision audit")
    device = torch.device("cuda:0")
    cfg, geometry = load_stage2_config(a.config)
    adapter = FrozenStage1OccupancyAdapter(**cfg["stage1"]["adapter"]).to(device).eval()
    stage2 = Stage2OccupancyModel(geometry, **cfg["model"]).to(device).eval()
    checkpoint = torch.load(a.stage2_checkpoint, map_location="cpu")
    stage2.load_state_dict(checkpoint["model"], strict=True)
    xyz_np = input_xyz(a.data_root / "08" / "input_" / f"{a.frame:06d}.npy").astype(np.float32, copy=False)
    xyz = torch.from_numpy(xyz_np).to(device)

    def encode(mode):
        if mode == "original":
            return adapter.encode_features(xyz)
        if mode == "force_fp32":
            with torch.autocast("cuda", enabled=False):
                return adapter.model.encode_condition(xyz)
        with torch.autocast("cuda", dtype=torch.float16):
            return adapter.model.encode_condition(xyz)

    precision_runs = {}
    features_by_mode = {}
    reference_features = None
    selected_features = None
    for mode in ("original", "force_fp32", "autocast_fp16"):
        records = []
        handles = _install_hooks(adapter.model.encoder, f"stage1_encoder.{mode}", records)
        # Per-mode warm-up excludes first-call CUDA/cuDNN initialization from
        # the precision latency comparison.
        _ = encode(mode)
        torch.cuda.synchronize(device)
        timings, current = [], None
        for _repeat in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); current = encode(mode); end.record(); torch.cuda.synchronize(device)
            timings.append(float(start.elapsed_time(end)))
        for h in handles: h.remove()
        if reference_features is None:
            reference_features = current
        diff = []
        for ref, cur in zip(reference_features, current):
            delta = (ref.float() - cur.float()).abs()
            diff.append({"max_abs": float(delta.max()), "mean_abs": float(delta.mean())})
        precision_runs[mode] = {
            "feature_pyramid_dtypes": [str(x.dtype) for x in current],
            "encoder_cuda_event_ms_median": float(np.median(timings)),
            "encoder_cuda_event_ms_trials": timings,
            "feature_abs_difference_vs_original": diff,
            "module_hooks": records,
        }
        features_by_mode[mode] = current
        if mode == a.encoder_precision:
            selected_features = current
    features = selected_features

    from export_stage2_snapshot import predicted_policy
    policy = predicted_policy(cfg)
    policy["stage1_nfe"] = 2
    score_by_mode, candidate_by_mode = {}, {}
    for mode, mode_features in features_by_mode.items():
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast("cuda", dtype=torch.float16):
            mode_score, _ = adapter.model.sample(
                xyz, nfe=2, condition=mode_features,
                generator=torch.Generator(device=device).manual_seed(
                    scene_seed("08", a.frame, cfg["candidates"]["seed"])),
            )
        end.record(); torch.cuda.synchronize(device)
        precision_runs[mode]["stage1_flow_cuda_event_ms"] = float(start.elapsed_time(end))
        score_np = mode_score[0, 0].float().cpu().numpy()
        ids_np, _, _ = select_occupancy_candidates(score_np, geometry, **policy["selection"])
        score_by_mode[mode] = mode_score
        candidate_by_mode[mode] = ids_np
    ref_score = score_by_mode["original"].float()
    for mode in precision_runs:
        delta = (score_by_mode[mode].float() - ref_score).abs()
        precision_runs[mode]["score_mean_abs_vs_original"] = float(delta.mean())
        precision_runs[mode]["score_max_abs_vs_original"] = float(delta.max())
        precision_runs[mode]["candidate_count"] = int(len(candidate_by_mode[mode]))
        precision_runs[mode]["candidate_ids_exact_vs_original"] = bool(
            np.array_equal(candidate_by_mode[mode], candidate_by_mode["original"]))
        precision_runs[mode]["candidate_ids_sha256"] = hashlib.sha256(
            np.ascontiguousarray(candidate_by_mode[mode]).view(np.uint8)).hexdigest()

    flow_records = []
    handles = _install_hooks(adapter.model.flow, "stage1_flow", flow_records)
    score = score_by_mode[a.encoder_precision]
    for h in handles:
        h.remove()

    ids_np, scores_np, _ = select_occupancy_candidates(
        score[0, 0].float().cpu().numpy(), geometry, **policy["selection"])
    ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
    centers = geometry.centers(geometry.unflatten(ids)).float()
    cond = {
        "lidar_features": sample_pyramid(features, centers, geometry.origin, geometry.maximum).detach().half(),
        "parent_score": torch.as_tensor(scores_np, device=device),
        "observed_child": torch.as_tensor(observed_children(xyz, ids_np, geometry), device=device),
        "valid_child_mask": torch.as_tensor(geometry.valid_children(ids), device=device),
    }
    s2_records = []
    handles = _install_hooks(stage2, "stage2", s2_records)
    state = torch.randn((len(ids), geometry.k), device=device, generator=torch.Generator(device=device).manual_seed(319)).float()
    with torch.autocast("cuda", dtype=torch.float16):
        velocity = stage2(ids, state=state, time=torch.tensor(0., device=device), **cond)
    for h in handles:
        h.remove()
    torch.cuda.synchronize(device)

    result = {
        "physical_gpu": a.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "frame": a.frame,
        "encoder_precision_comparison": precision_runs,
        "stage1_encoder": {
            "scope": "FrozenStage1OccupancyAdapter.encode_features wraps model.encode_condition in torch.autocast(cuda,float16)",
            "feature_pyramid_dtypes": [str(x.dtype) for x in features],
            "selected_mode": a.encoder_precision,
        },
        "stage1_flow": {"scope": "outer torch.autocast(cuda,float16)", "module_hooks": flow_records},
        "stage2": {
            "scope": "outer autocast(cuda,float16) for input projections/decoder; PTv3 explicitly disables autocast and casts fused features to float32",
            "condition_dtypes": {k: str(v.dtype) for k, v in cond.items()},
            "output_dtype": str(velocity.dtype),
            "module_hooks": s2_records,
        },
        "conclusion": {
            "encoder_autocast": True,
            "stage2_ptv3_autocast": False,
            "stage2_input_projection_and_decoder_in_outer_amp_scope": True,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("physical_gpu", "gpu_name", "torch", "torch_cuda", "stage1_encoder", "stage2", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()

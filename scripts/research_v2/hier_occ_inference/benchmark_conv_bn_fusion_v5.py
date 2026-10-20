#!/usr/bin/env python
"""Measure inference-only Conv3d+BatchNorm3d fusion on real Seq08 input."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_hs_patchscene_inference import S1_SHA, S2_SHA, _admit
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import file_sha


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu", type=int, choices=(2, 3), default=2)
    p.add_argument("--data-root", type=Path, default=Path("/data-12/M2024-HWZ/KITTI_Odometry"))
    p.add_argument("--config", type=Path, default=ROOT / "configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml")
    p.add_argument("--stage1-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_preflight/stage1_final_100kframes.pt"))
    p.add_argument("--stage2-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/checkpoints/step025000.pt"))
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--group", choices=("all", "voxel_backbone", "voxel_neck", "local_backbone", "local_neck"), default="all")
    p.add_argument("--rounds", type=int, default=8)
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
    encoder = adapter.model.encoder
    eligible = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.modules.conv.Conv3d):
            # Inspect the parent ConvModule to identify its attached normalization.
            continue
        if hasattr(module, "conv") and isinstance(getattr(module, "conv", None), nn.Conv3d):
            norm = getattr(module, "norm", None)
            order = tuple(getattr(module, "order", ("conv", "norm", "act")))
            if (isinstance(norm, nn.BatchNorm3d) and getattr(module, "with_norm", True)
                    and "conv" in order and "norm" in order and order.index("conv") < order.index("norm")
                    and (a.group == "all" or name.startswith(a.group + "."))):
                eligible.append((name, module, module.conv, norm))
    if not eligible:
        raise RuntimeError("No ConvModule with BatchNorm3d found in frozen Encoder")
    fused = []
    for name, parent, conv, bn in eligible:
        # fuse_conv_bn_eval may rewrite the Conv module in-place; preserve the
        # live unfused branch and checkpoint-derived weights by fusing clones.
        fused_conv = fuse_conv_bn_eval(copy.deepcopy(conv).eval(), copy.deepcopy(bn).eval())
        fused.append({"name": name, "parent": parent, "conv": conv, "bn": bn,
                      "fused_conv": fused_conv})

    x_path = a.data_root / "08" / "input_" / f"{a.frame:06d}.npy"
    x_np = input_xyz(x_path).astype(np.float32, copy=False)
    x = torch.from_numpy(x_np).cuda()

    def set_fused(enabled):
        for row in fused:
            row["parent"].conv = row["fused_conv"] if enabled else row["conv"]
            if enabled:
                row["parent"].norm = nn.Identity()
            else:
                row["parent"].norm = row["bn"]

    def run(enabled, sample_seed):
        set_fused(enabled)
        seed_all(sample_seed)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = encoder(x)
        end.record()
        return out, begin, end

    # Alternate both implementations during warmup and measurement to reduce clock bias.
    for i in range(4):
        _, begin, end = run(bool(i % 2), 20261019 + a.frame)
    torch.cuda.synchronize()
    measurements = {"separate": [], "fused": []}
    outputs = {}
    for i in range(a.rounds):
        for label, enabled in (("separate", False), ("fused", True)) if i % 2 == 0 else (("fused", True), ("separate", False)):
            out, begin, end = run(enabled, 20261019 + a.frame)
            measurements[label].append((begin, end))
            outputs[label] = out
    torch.cuda.synchronize()
    times = {label: [float(s.elapsed_time(e)) for s, e in pairs]
             for label, pairs in measurements.items()}
    errors = []
    for x_ref, x_new in zip(outputs["separate"], outputs["fused"]):
        d = (x_ref.float() - x_new.float()).abs()
        errors.append({"shape": list(x_ref.shape), "reference_dtype": str(x_ref.dtype),
                       "fused_dtype": str(x_new.dtype), "max_abs": float(d.max().item()),
                       "mean_abs": float(d.mean().item()),
                       "relative_l2": float(torch.linalg.vector_norm(d) /
                                            torch.linalg.vector_norm(x_ref.float()).clamp_min(1e-12))})
    payload = {
        "status": "SMOKE_COMPLETE", "physical_gpu": a.physical_gpu,
        "gpu_admission": admission, "frame": a.frame, "input_point_count": len(x_np),
        "input_sha256": file_sha(x_path), "stage1_checkpoint_sha256": S1_SHA,
        "stage2_checkpoint_sha256": S2_SHA, "torch": torch.__version__,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "eligible_fused_conv_bn_layers": [{"name": name, "order": list(parent.order)} for name, parent, _, _ in eligible],
        "eligible_layer_count": len(eligible), "warmup_pairs": 2,
        "cuda_event_encoder_ms": times,
        "mean_ms": {k: float(np.mean(v)) for k, v in times.items()},
        "median_ms": {k: float(np.median(v)) for k, v in times.items()},
        "pyramid_errors": errors,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "notes": "Inference-only fuse_conv_bn_eval on cloned runtime modules; original checkpoint and training path are untouched. This is a single-frame screening, not a registered 5+20 result.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"eligible": len(eligible), "mean_ms": payload["mean_ms"], "pyramid_errors": errors}), flush=True)


if __name__ == "__main__":
    main()

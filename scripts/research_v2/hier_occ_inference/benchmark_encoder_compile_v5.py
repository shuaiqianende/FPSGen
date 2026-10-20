#!/usr/bin/env python
"""Single-scene smoke for compiling one dense Stage1 Encoder submodule."""
from __future__ import annotations

import argparse
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
from benchmark_hs_patchscene_inference import S1_SHA, S2_SHA, _admit
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import file_sha


def _reset_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _errors(ref, value):
    result = []
    for a, b in zip(ref, value):
        d = (a.float() - b.float()).abs()
        result.append({"shape": list(a.shape), "dtype_reference": str(a.dtype),
                       "dtype_variant": str(b.dtype), "max_abs": float(d.max().item()),
                       "mean_abs": float(d.mean().item()),
                       "relative_l2": float(torch.linalg.vector_norm(d) /
                                            torch.linalg.vector_norm(a.float()).clamp_min(1e-12))})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu", type=int, choices=(2, 3), default=2)
    p.add_argument("--data-root", type=Path, default=Path("/data-12/M2024-HWZ/KITTI_Odometry"))
    p.add_argument("--config", type=Path, default=ROOT / "configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml")
    p.add_argument("--stage1-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_preflight/stage1_final_100kframes.pt"))
    p.add_argument("--stage2-checkpoint", type=Path, default=Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/checkpoints/step025000.pt"))
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--target", choices=("voxel_backbone", "voxel_neck", "local_backbone", "classification_neck"), required=True)
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
    x_np = input_xyz(a.data_root / "08" / "input_" / f"{a.frame:06d}.npy").astype(np.float32, copy=False)
    x = torch.from_numpy(x_np).cuda()
    seed = 20261019 + a.frame
    _reset_rng(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        baseline = encoder(x)
    torch.cuda.synchronize()
    if a.target == "classification_neck":
        original = encoder._classification_neck
        compiled_cls = torch.compile(original, dynamic=False)
        encoder._classification_neck = compiled_cls
    else:
        original = getattr(encoder, a.target)
        setattr(encoder, a.target, torch.compile(original, dynamic=False))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        _reset_rng(seed)
        t0 = time.perf_counter()
        first = encoder(x)
        torch.cuda.synchronize()
        first_wall_s = time.perf_counter() - t0
        _reset_rng(seed)
        for _ in range(3):
            _ = encoder(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            output = encoder(x)
        torch.cuda.synchronize()
        warm_wall_ms = (time.perf_counter() - t0) * 1000 / 5
    from torch._dynamo.utils import counters
    payload = {
        "status": "SMOKE_COMPLETE",
        "physical_gpu": a.physical_gpu,
        "gpu_admission": admission,
        "frame": a.frame,
        "input_point_count": len(x_np),
        "input_sha256": file_sha(a.data_root / "08" / "input_" / f"{a.frame:06d}.npy"),
        "target": a.target,
        "stage1_checkpoint_sha256": S1_SHA,
        "stage2_checkpoint_sha256": S2_SHA,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "first_call_wall_seconds_including_lazy_compile": first_wall_s,
        "warm_encoder_mean_wall_ms": warm_wall_ms,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "pyramid_errors": _errors(baseline, output),
        "dynamo_counters": {g: dict(v) for g, v in counters.items()},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"target": a.target, "compile_first_call_s": first_wall_s,
                      "warm_encoder_ms": warm_wall_ms, "pyramid_errors": payload["pyramid_errors"]}), flush=True)


if __name__ == "__main__":
    main()

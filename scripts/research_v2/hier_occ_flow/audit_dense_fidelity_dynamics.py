#!/usr/bin/env python3
"""Measure R0/R1/R2 score dynamics and R0 Stage0 activation sensitivity.

This diagnostic uses only the registered Development split, identical input
frames and paired source-noise seeds. It does not alter checkpoints or select a
new threshold. It is a sensitivity proxy, not a substitute for retraining.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.storage import SparseShardDataset, file_sha, atomic_json

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from eval_stage1_occ import load_model  # noqa: E402


DEFAULT_BASE = Path("/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-omd3d-v1")
DEFAULT_CONFIGS = {
    "r0": DEFAULT_BASE / "configs/research_v2/hier_occ_flow/omd3d_v1_r0_original.yaml",
    "r1": DEFAULT_BASE / "configs/research_v2/hier_occ_flow/omd3d_v1_r1_bottleneck.yaml",
    "r2": DEFAULT_BASE / "configs/research_v2/hier_occ_flow/omd3d_v1_r2_omd3d.yaml",
}
DEFAULT_CHECKPOINTS = {
    name: DEFAULT_BASE / f"experiments/omd3d_v1/{subdir}/formal20k/checkpoints/exposure00020000.pt"
    for name, subdir in (("r0", "r0_original"), ("r1", "r1_bottleneck"), ("r2", "r2_omd3d"))
}
BLOCKS = (
    ("voxel_backbone.layers.0.0", "voxel_backbone", 0),
    ("voxel_backbone.layers.0.1", "voxel_backbone", 1),
    ("local_backbone.layers.0.0", "local_backbone", 0),
    ("local_backbone.layers.0.1", "local_backbone", 1),
    ("local_backbone.layers.0.2", "local_backbone", 2),
)
POPCOUNT8 = tuple(bin(i).count("1") for i in range(256))


def load_cfg(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["geometry"] = yaml.safe_load((path.parent / cfg["geometry"]).read_text())
    validate_config(cfg)
    return cfg


def load_dev_frames(cfg, geometry, limit):
    dataset = SparseShardDataset(cfg["development_dataset"], geometry, verify=True)
    indices = np.linspace(0, len(dataset) - 1, min(limit, len(dataset)), dtype=int)
    rows = []
    for idx in indices:
        record = dataset[int(idx)]
        frame = int(record["frame"])
        sequence = str(record["sequence"])
        path = Path(cfg["data_root"]) / sequence / "input_" / f"{frame:06d}.npy"
        xyz = input_xyz(path)
        rows.append({
            "sequence": sequence, "frame": frame, "record": record,
            "input_path": path, "input_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
            "xyz": xyz,
        })
    return rows


def score_metrics(score, record, valid_mask, fixed_ks):
    flat = score.reshape(-1).float()
    valid = valid_mask.reshape(-1).bool()
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    valid_scores = flat[valid_ids]
    gt_ids = torch.as_tensor(record["parent_ids"].astype(np.int64), device=score.device)
    threshold_ids = valid_ids[valid_scores >= 0.5]
    threshold_hit = torch.isin(threshold_ids, gt_ids)
    tp = int(threshold_hit.sum().item())
    pred_n = int(threshold_ids.numel())
    gt_n = int(gt_ids.numel())
    fn = gt_n - tp
    tn = None
    out = {
        "threshold_0_5": {
            "precision": tp / max(pred_n, 1), "recall": tp / max(gt_n, 1),
            "iou": tp / max(tp + pred_n - tp + fn, 1),
            "predicted_parents": pred_n, "gt_parents": gt_n, "tp": tp, "fp": pred_n - tp, "fn": fn,
        },
        "score": {},
    }
    gt_mask = torch.zeros_like(valid)
    gt_mask[gt_ids] = True
    for label, mask in (("positive", gt_mask & valid), ("negative", ~gt_mask & valid)):
        vals = flat[mask]
        if vals.numel():
            quantiles = torch.tensor([0, .25, .5, .9, .99, .999, 1], device=vals.device)
            q = torch.quantile(vals, quantiles)
            out["score"][label] = {str(k): float(v) for k, v in zip(
                ("min", "p25", "p50", "p90", "p99", "p999", "max"), q.tolist())}
        else:
            out["score"][label] = {}
    margins = (flat[valid] - 0.5).abs()
    out["score"]["threshold_margin_counts"] = {
        str(eps): int((margins < eps).sum().item()) for eps in (.001, .005, .01, .02, .05)
    }
    total_fine = int(sum(POPCOUNT8[int(v)] for v in record["child_occ_mask"]))
    for k in fixed_ks:
        k = min(int(k), valid_scores.numel())
        top = torch.topk(valid_scores, k=k, sorted=False).indices
        candidate_ids = valid_ids[top]
        selected = torch.isin(gt_ids, candidate_ids)
        hits = int(selected.sum().item())
        # child_occ_mask stores a bitset, so count occupied child bits exactly.
        chosen_bytes = record["child_occ_mask"][selected.cpu().numpy()]
        covered_fine = int(sum(POPCOUNT8[int(v)] for v in chosen_bytes))
        out[f"fixed_k_{k}"] = {
            "precision": hits / max(k, 1), "recall": hits / max(gt_n, 1),
            "iou": hits / max(k + gt_n - hits, 1),
            "fine_gt_recall_ceiling": covered_fine / max(total_fine, 1),
            "selected_gt_parents": hits,
        }
    return out


def make_block_lookup(model):
    lookup = {}
    for name, module in model.encoder.named_modules():
        if name in {x[0] for x in BLOCKS}:
            lookup[name] = module
    if len(lookup) != len(BLOCKS):
        raise RuntimeError(f"Expected five Stage0 blocks, found {sorted(lookup)}")
    return lookup


def deterministic_noise(shape, device, dtype, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def run_model(model, xyz, record, device, seed):
    with torch.inference_mode():
        condition = model.encode_condition(xyz)
        source_rng = torch.Generator(device=device)
        source_rng.manual_seed(int(seed))
        score, endpoint = model.sample(xyz, generator=source_rng, nfe=2, condition=condition)
    return score, endpoint


def run_r0_perturbations(model, item, device, fixed_ks, base_seed):
    xyz = torch.as_tensor(np.ascontiguousarray(item["xyz"]), device=device)
    baseline, _ = run_model(model, xyz, item["record"], device,
                            base_seed + item["frame"] * 13)
    base_flat = baseline.reshape(-1).float()
    rows = []
    lookup = make_block_lookup(model)
    for block_index, (name, _, _) in enumerate(BLOCKS):
        for strength in (.01, .05):
            seed = base_seed + item["frame"] * 1009 + block_index * 97 + int(strength * 10000)

            def perturb(_module, _inputs, output, *, _seed=seed, _strength=strength):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Expected Tensor output from {name}")
                rms = output.float().square().mean().sqrt()
                noise = deterministic_noise(output.shape, output.device, output.dtype, _seed)
                return output + noise * (rms.to(output.dtype) * _strength)

            hook = lookup[name].register_forward_hook(perturb)
            try:
                score, _ = run_model(model, xyz, item["record"], device,
                                     base_seed + item["frame"] * 13)
            finally:
                hook.remove()
            flat = score.reshape(-1).float()
            metrics = score_metrics(score, item["record"], model.valid, fixed_ks)
            changed = int((flat >= .5).logical_xor(base_flat >= .5).sum().item())
            row = {
                "sequence": item["sequence"], "frame": item["frame"], "block": name,
                "relative_activation_noise_std": strength, "perturbation_seed": seed,
                "score_max_abs_error": float((flat - base_flat).abs().max().item()),
                "score_mean_abs_error": float((flat - base_flat).abs().mean().item()),
                "threshold_candidate_flips": changed,
                "near_threshold_flip_counts": {}, "metrics": metrics,
            }
            for eps in (.001, .005, .01, .02, .05):
                near = (base_flat - .5).abs() < eps
                row["near_threshold_flip_counts"][str(eps)] = int(
                    ((flat >= .5).logical_xor(base_flat >= .5) & near).sum().item())
            rows.append(row)
            del score
    base_metrics = score_metrics(baseline, item["record"], model.valid, fixed_ks)
    return base_metrics, rows


def attach_omd_branch_audit(model):
    from fpsgen.hier_occ_flow.dense_encoder_variants import ObservationModulatedDense3D
    blocks = [(name, module) for name, module in model.encoder.named_modules()
              if isinstance(module, ObservationModulatedDense3D)]
    handles = []
    state = {}
    rows = []

    def branch_hook(block_name, key):
        def hook(_module, _inputs, output):
            state[block_name][key] = output
        return hook

    for name, block in blocks:
        state[name] = {}
        handles.extend([
            block.register_forward_pre_hook(lambda mod, inputs, _n=name: state[_n].update(obs=inputs[1])),
            block.local2.register_forward_hook(branch_hook(name, "local")),
            block.axial_x.register_forward_hook(branch_hook(name, "ax")),
            block.axial_y.register_forward_hook(branch_hook(name, "ay")),
            block.axial_z.register_forward_hook(branch_hook(name, "az")),
            block.gate.register_forward_hook(branch_hook(name, "gate_logits")),
        ])

        def finish(_module, _inputs, _output, *, _name=name, _block=block):
            values = state[_name]
            obs = values["obs"].mask.bool()
            obs_exp = obs.expand_as(values["local"])
            gate = values["gate_logits"].sigmoid().chunk(3, dim=1)
            axes = [values[x] for x in ("ax", "ay", "az")]
            context = sum(g * a for g, a in zip(gate, axes))

            def rms(x, mask=None):
                v = x.float().square()
                if mask is not None:
                    expanded = mask.expand_as(x)
                    v = v[expanded]
                return v.mean().sqrt()

            local = values["local"]
            stat_names = (
                "context_scale_signed", "context_scale_abs", "local_rms",
                "axial_x_rms", "axial_y_rms", "axial_z_rms", "combined_context_rms",
                "context_to_local_rms_ratio", "local_observed_rms", "local_unobserved_rms",
                "context_observed_rms", "context_unobserved_rms",
                "gate_x_mean", "gate_y_mean", "gate_z_mean",
                "gate_x_observed_mean", "gate_y_observed_mean", "gate_z_observed_mean",
                "gate_x_unobserved_mean", "gate_y_unobserved_mean", "gate_z_unobserved_mean",
            )
            output = torch.stack([
                _block.context_scale.detach().float(),
                _block.context_scale.detach().float().abs(),
                rms(local), rms(axes[0]), rms(axes[1]), rms(axes[2]), rms(context),
                rms(context) / (rms(local) + 1e-12),
                rms(local, obs_exp), rms(local, ~obs_exp),
                rms(context, obs), rms(context, ~obs),
                *(g.float().mean() for g in gate),
                *(g.float()[obs].mean() if bool(obs.any()) else g.float().mean() for g in gate),
                *(g.float()[~obs].mean() if bool((~obs).any()) else g.float().mean() for g in gate),
            ]).detach()
            rows.append({"block": _name, **dict(zip(stat_names, output.cpu().tolist()))})
            state[_name].clear()

        handles.append(block.register_forward_hook(finish))
    return handles, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=(2, 3))
    parser.add_argument("--r0-config", type=Path, default=DEFAULT_CONFIGS["r0"])
    parser.add_argument("--r1-config", type=Path, default=DEFAULT_CONFIGS["r1"])
    parser.add_argument("--r2-config", type=Path, default=DEFAULT_CONFIGS["r2"])
    parser.add_argument("--r0-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["r0"])
    parser.add_argument("--r1-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["r1"])
    parser.add_argument("--r2-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["r2"])
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.physical_gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must equal --physical-gpu")
    admission = gpu_preflight((args.physical_gpu,))
    if admission["status"] != "AVAILABLE":
        atomic_json(args.output, {"status": "BLOCKED", "gpu_admission": admission})
        return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    specs = {
        name: (load_cfg(getattr(args, f"{name}_config")), getattr(args, f"{name}_checkpoint"))
        for name in ("r0", "r1", "r2")
    }
    loaded = {}
    geometry = None
    for name, (cfg, checkpoint) in specs.items():
        model, g, ckpt = load_model(cfg, checkpoint, device, use_ema=True)
        loaded[name] = {"model": model, "cfg": cfg, "checkpoint": checkpoint, "checkpoint_state": ckpt}
        geometry = g
    frames = load_dev_frames(specs["r0"][0], geometry, args.frames)
    fixed_ks = (78694, 84963, 94087)
    output = {
        "status": "RUNNING", "physical_gpu": args.physical_gpu,
        "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "frame_limit": len(frames), "seed": args.seed, "fixed_ks": fixed_ks,
        "checkpoints": {name: {"path": str(value["checkpoint"]),
                               "sha256": file_sha(value["checkpoint"]),
                               "successful_updates": int(value["checkpoint_state"]["successful_updates"])}
                        for name, value in loaded.items()},
        "frame_manifest": [{k: item[k] for k in ("sequence", "frame", "input_path", "input_sha256")}
                           for item in frames],
        "scores": [], "r0_activation_perturbations": [], "r2_branch_stats": [],
        "r2_context_scale_ablation": [], "gpu_admission": admission,
    }
    for item in frames:
        xyz = torch.as_tensor(np.ascontiguousarray(item["xyz"]), device=device)
        for name in ("r0", "r1", "r2"):
            model = loaded[name]["model"]
            branch_rows = None
            handles = []
            if name == "r2":
                handles, branch_rows = attach_omd_branch_audit(model)
            try:
                score, endpoint = run_model(model, xyz, item["record"], device,
                                            args.seed + item["frame"] * 13)
            finally:
                for h in handles:
                    h.remove()
            output["scores"].append({
                "model": name, "sequence": item["sequence"], "frame": item["frame"],
                "metrics": score_metrics(score, item["record"], model.valid, fixed_ks),
                "score_mean": float(score.float().mean().item()),
                "score_std": float(score.float().std().item()),
                "endpoint_finite": bool(torch.isfinite(endpoint).all().item()),
            })
            if name == "r2":
                output["r2_branch_stats"].extend({
                    "sequence": item["sequence"], "frame": item["frame"], **row,
                } for row in branch_rows)
                modules = [m for m in model.encoder.modules()
                           if hasattr(m, "context_scale") and isinstance(m.context_scale, torch.nn.Parameter)]
                saved = [m.context_scale.detach().clone() for m in modules]
                with torch.no_grad():
                    for m in modules:
                        m.context_scale.zero_()
                try:
                    zero_score, zero_endpoint = run_model(model, xyz, item["record"], device,
                                                          args.seed + item["frame"] * 13)
                finally:
                    with torch.no_grad():
                        for m, value in zip(modules, saved):
                            m.context_scale.copy_(value)
                output["r2_context_scale_ablation"].append({
                    "sequence": item["sequence"], "frame": item["frame"],
                    "score_max_abs_error": float((zero_score.float() - score.float()).abs().max().item()),
                    "score_mean_abs_error": float((zero_score.float() - score.float()).abs().mean().item()),
                    "candidate_flips_at_0_5": int(((zero_score >= .5) != (score >= .5)).sum().item()),
                    "endpoint_max_abs_error": float((zero_endpoint.float() - endpoint.float()).abs().max().item()),
                    "metrics_with_context": score_metrics(score, item["record"], model.valid, fixed_ks),
                    "metrics_without_context_scale": score_metrics(zero_score, item["record"], model.valid, fixed_ks),
                })
            if name == "r0":
                base_metrics, perturb = run_r0_perturbations(
                    model, item, device, fixed_ks, args.seed)
                output["r0_activation_perturbations"].append({
                    "sequence": item["sequence"], "frame": item["frame"],
                    "baseline_metrics": base_metrics, "perturbations": perturb,
                })
            del score, endpoint
        del xyz
        torch.cuda.empty_cache()
    output["status"] = "COMPLETE"
    output["source_sha256"] = {str(Path(__file__).resolve()): file_sha(Path(__file__))}
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()

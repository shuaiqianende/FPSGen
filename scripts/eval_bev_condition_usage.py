#!/usr/bin/env python3
"""Measure whether a Stage-1 BEV model uses correct spatial conditions.

For fixed frames, noise and FM times, this compares correct conditions against
zeroed and cyclically shuffled conditions.  It deliberately reports loss-based
condition-use gains rather than treating output change as evidence of use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from fpsgen.datasets import datasets
from fpsgen.models.gen_img import FlowIMG


MODES = ("000", "100", "010", "001", "110", "101", "011", "111")
TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)
TIME_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def slice_batch(batch, size):
    return {key: value[:size] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def apply_mode(raw_pc, layout, mode):
    lidar, vehicle, road = (bit == "1" for bit in mode)
    return (
        raw_pc if lidar else torch.zeros_like(raw_pc),
        torch.cat((layout[:, 0:1] if vehicle else torch.zeros_like(layout[:, 0:1]),
                   layout[:, 1:2] if road else torch.zeros_like(layout[:, 1:2])), dim=1),
    )


def keep_for_mode(mode, batch, device):
    return torch.tensor([[bit == "1" for bit in mode]], device=device, dtype=torch.bool).expand(batch, -1)


def shuffled(raw_pc, layout, mode):
    if raw_pc.shape[0] < 2:
        raise ValueError("wrong-frame evaluation requires batch size >= 2; refusing self-roll")
    raw, semantic = apply_mode(raw_pc, layout, mode)
    shift = 1
    lidar, vehicle, road = (bit == "1" for bit in mode)
    if lidar:
        raw = raw.roll(shift, dims=0)
    if vehicle:
        semantic[:, 0:1] = semantic[:, 0:1].roll(shift, dims=0)
    if road:
        semantic[:, 1:2] = semantic[:, 1:2].roll(shift, dims=0)
    return raw, semantic


def velocity_loss(model, xt, t, target, raw_pc, layout, mode):
    keep = keep_for_mode(mode, xt.shape[0], xt.device)
    if getattr(model.model, "supports_condition_keep", False):
        velocity = model.model(xt, t, raw_pc, layout, keep)
    else:
        velocity = model.model(xt, t, raw_pc, layout)
    weights = torch.tensor((1.0, 2.0, 1.0), device=velocity.device).view(1, 3, 1, 1)
    return ((velocity - target).square() * weights).mean(), velocity


def summarize_records(values):
    """Compute condition gains for a nonempty set of fixed-time observations."""
    if not values["correct"]:
        return None
    correct = sum(values["correct"]) / len(values["correct"])
    zero = sum(values["zero"]) / len(values["zero"])
    shuffle_loss = sum(values["shuffle"]) / len(values["shuffle"])
    return {
        "loss_correct": correct,
        "loss_zero": zero,
        "loss_shuffle": shuffle_loss,
        "g_zero": 1.0 - correct / zero,
        "g_shuffle": 1.0 - correct / shuffle_loss,
        "delta_v": sum(values["delta_v"]) / len(values["delta_v"]),
    }


def merge_records(records):
    merged = {"correct": [], "zero": [], "shuffle": [], "delta_v": []}
    for values in records:
        for key in merged:
            merged[key].extend(values[key])
    return merged


def bin_label(lower, upper):
    return f"{lower:.1f}-{upper:.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Condition-usage evaluation requires one CUDA-visible GPU")

    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = state["hyper_parameters"]
    cfg["data"]["data_dir"] = str(args.data_root)
    cfg["data"]["train"] = ["08"]
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["num_workers"] = min(4, int(cfg["train"].get("num_workers", 4)))
    model = FlowIMG.load_from_checkpoint(str(args.checkpoint), hparams=cfg).cuda().eval()
    loader = datasets.dataloaders[cfg["data"]["dataloader"]](cfg)._dataloader(["08"], "train", shuffle=False)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = {
        mode: {
            time: {"correct": [], "zero": [], "shuffle": [], "delta_v": []}
            for time in TIMES
        }
        for mode in MODES
    }
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if seen >= args.frames:
                break
            batch = move(batch, "cuda")
            batch = slice_batch(batch, min(args.frames - seen, batch["pcd_full"].shape[0]))
            gt = batch["pcd_full"]
            raw_pc = model.model.get_raw_pc_bev(batch["pcd_part"])
            layout = model.processor.get_layout_bev(gt, batch["full_label"]) * 2.0 - 1.0
            target_bev = model.processor.points_to_bev_target(gt)
            x0 = torch.randn(target_bev.shape, generator=generator, device="cuda", dtype=target_bev.dtype)
            target_velocity = target_bev - x0
            for time in TIMES:
                t = torch.full((gt.shape[0],), time, device="cuda")
                xt = (1.0 - time) * x0 + time * target_bev
                zero_raw, zero_layout = apply_mode(raw_pc, layout, "000")
                zero_loss, zero_velocity = velocity_loss(model, xt, t, target_velocity, zero_raw, zero_layout, "000")
                for mode in MODES:
                    correct_raw, correct_layout = apply_mode(raw_pc, layout, mode)
                    shuffle_raw, shuffle_layout = shuffled(raw_pc, layout, mode)
                    correct_loss, correct_velocity = velocity_loss(model, xt, t, target_velocity, correct_raw, correct_layout, mode)
                    shuffle_loss, _ = velocity_loss(model, xt, t, target_velocity, shuffle_raw, shuffle_layout, mode)
                    records[mode][time]["correct"].append(float(correct_loss))
                    records[mode][time]["zero"].append(float(zero_loss))
                    records[mode][time]["shuffle"].append(float(shuffle_loss))
                    records[mode][time]["delta_v"].append(float((correct_velocity - zero_velocity).abs().mean() / (zero_velocity.abs().mean() + 1e-8)))
            seen += gt.shape[0]

    summary = {
        "checkpoint": str(args.checkpoint), "frames": seen, "seed": args.seed,
        "times": TIMES, "time_bins": [bin_label(*bounds) for bounds in TIME_BINS],
        "modes": {}, "headlines": {},
    }
    for mode, values in records.items():
        summary["modes"][mode] = summarize_records(merge_records(values.values()))
        summary["modes"][mode]["by_time"] = {
            f"{time:.1f}": summarize_records(values[time]) for time in TIMES
        }
        summary["modes"][mode]["by_time_bin"] = {
            bin_label(lower, upper): summarize_records(merge_records(
                values[time] for time in TIMES
                if lower <= time < upper or (upper == 1.0 and time == upper)
            ))
            for lower, upper in TIME_BINS
        }
    for mode in MODES:
        for upper, key in ((0.2, "t_lt_0_2"), (0.4, "t_lt_0_4")):
            gain = summarize_records(merge_records(
                records[mode][time] for time in TIMES if time < upper
            ))
            summary["headlines"][f"gshuffle_{mode}_{key}"] = gain["g_shuffle"] if gain else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

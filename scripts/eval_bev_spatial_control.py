#!/usr/bin/env python3
"""Short-run spatial-control diagnostics with a deterministic non-self shuffle.

It caches BEV tensors for the requested (small) evaluation split only, then
maps frame i to (i + 17) mod N.  This avoids the invalid B=1 ``roll`` case and
keeps correct/zero/wrong conditions paired with identical x0, t and GT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fpsgen.datasets import datasets
from fpsgen.models.gen_img import FlowIMG

MODES = ("100", "010", "001", "111")
TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)


def move(value, device):
    if isinstance(value, torch.Tensor): return value.to(device, non_blocking=True)
    if isinstance(value, dict): return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(move(item, device) for item in value)
    return value


def mode_keep(mode, batch, device):
    return torch.tensor([[value == "1" for value in mode]], device=device, dtype=torch.bool).expand(batch, -1)


def masked(raw, layout, mode):
    keep = mode_keep(mode, raw.shape[0], raw.device).to(raw.dtype)[:, :, None, None]
    return raw * keep[:, :1], torch.cat((layout[:, :1] * keep[:, 1:2], layout[:, 1:2] * keep[:, 2:3]), dim=1)


def call(model, xt, t, raw, layout, keep):
    if getattr(model.model, "supports_condition_keep", False): return model.model(xt, t, raw, layout, keep)
    return model.model(xt, t, raw, layout)


def regions(target, observed):
    occupied = target[:, 2:3] > 0
    completion = occupied & ~observed.bool()
    boundary = F.max_pool2d(occupied.float(), 3, 1, 1).bool() ^ F.max_pool2d((~occupied).float(), 3, 1, 1).bool()
    return {"full": torch.ones_like(occupied), "occupied": occupied, "completion": completion, "boundary": boundary}


def weighted_region_loss(prediction, target, region):
    per_cell = ((prediction - target).square() * prediction.new_tensor((1., 2., 1.)).view(1, 3, 1, 1)).mean(dim=1, keepdim=True)
    denom = region.float().sum().clamp_min(1.)
    return float((per_cell * region).sum() / denom)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA-visible GPU required")
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = state["hyper_parameters"]
    cfg["data"]["data_dir"] = str(args.data_root); cfg["data"]["train"] = ["08"]
    cfg["train"]["batch_size"] = args.batch_size; cfg["train"]["num_workers"] = min(4, cfg["train"].get("num_workers", 4))
    model = FlowIMG.load_from_checkpoint(str(args.checkpoint), hparams=cfg).cuda().eval()
    loader = datasets.dataloaders[cfg["data"]["dataloader"]](cfg)._dataloader(["08"], "train", shuffle=False)
    cached = []
    with torch.no_grad():
        for batch in loader:
            if len(cached) >= args.frames: break
            batch = move(batch, "cuda")
            raw = model.model.get_raw_pc_bev(batch["pcd_part"])
            layout = model.processor.get_layout_bev(batch["pcd_full"], batch["full_label"]) * 2 - 1
            target = model.processor.points_to_bev_target(batch["pcd_full"])
            observed = model.processor.points_to_bev_target(batch["pcd_part"])[:, 2:3] > 0
            for index in range(min(raw.shape[0], args.frames - len(cached))):
                cached.append(tuple(value[index].cpu().half() for value in (raw, layout, target, observed.float())))
    if len(cached) < 2: raise RuntimeError("need at least two frames for a real wrong-frame condition")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = {mode: {f"{time:.1f}": {name: {"correct": [], "zero": [], "shuffle": []} for name in ("full", "occupied", "completion", "boundary")} for time in TIMES} for mode in MODES}
    with torch.no_grad():
        for start in range(0, len(cached), args.batch_size):
            ids = list(range(start, min(start + args.batch_size, len(cached))))
            wrong_ids = [(index + 17) % len(cached) for index in ids]
            raw = torch.stack([cached[index][0] for index in ids]).cuda().float()
            layout = torch.stack([cached[index][1] for index in ids]).cuda().float()
            target = torch.stack([cached[index][2] for index in ids]).cuda().float()
            observed = torch.stack([cached[index][3] for index in ids]).cuda().bool()
            wrong_raw = torch.stack([cached[index][0] for index in wrong_ids]).cuda().float()
            wrong_layout = torch.stack([cached[index][1] for index in wrong_ids]).cuda().float()
            x0 = torch.randn(target.shape, device="cuda", generator=generator)
            velocity_target = target - x0
            for time in TIMES:
                t = torch.full((len(ids),), time, device="cuda")
                xt = (1 - time) * x0 + time * target
                zero = call(model, xt, t, torch.zeros_like(raw), torch.zeros_like(layout), mode_keep("000", len(ids), "cuda"))
                for mode in MODES:
                    correct_raw, correct_layout = masked(raw, layout, mode)
                    shuffled_raw, shuffled_layout = masked(wrong_raw, wrong_layout, mode)
                    correct = call(model, xt, t, correct_raw, correct_layout, mode_keep(mode, len(ids), "cuda"))
                    shuffle = call(model, xt, t, shuffled_raw, shuffled_layout, mode_keep(mode, len(ids), "cuda"))
                    for name, region in regions(target, observed).items():
                        records[mode][f"{time:.1f}"][name]["correct"].append(weighted_region_loss(correct, velocity_target, region))
                        records[mode][f"{time:.1f}"][name]["zero"].append(weighted_region_loss(zero, velocity_target, region))
                        records[mode][f"{time:.1f}"][name]["shuffle"].append(weighted_region_loss(shuffle, velocity_target, region))
    def summary(values):
        c, z, s = (sum(values[key]) / len(values[key]) for key in ("correct", "zero", "shuffle"))
        return {"correct": c, "zero": z, "shuffle": s, "g_zero": 1 - c / z, "g_shuffle": 1 - c / s}
    output = {"checkpoint": str(args.checkpoint), "frames": len(cached), "wrong_frame_offset": 17, "seed": args.seed, "modes": {}}
    for mode in MODES:
        output["modes"][mode] = {"by_time": {time: {region: summary(values) for region, values in regions_.items()} for time, regions_ in records[mode].items()}}
        for cutoff, label in ((0.2, "t_lt_0_2"), (0.4, "t_lt_0_4")):
            pooled = {region: {kind: [] for kind in ("correct", "zero", "shuffle")} for region in ("full", "occupied", "completion", "boundary")}
            for time in TIMES:
                if time < cutoff:
                    for region, values in records[mode][f"{time:.1f}"].items():
                        for kind in pooled[region]: pooled[region][kind].extend(values[kind])
            output["modes"][mode][label] = {region: summary(values) for region, values in pooled.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Frozen-frame condition evaluation, no augmentation or lossy feature cache.

Each frame has deterministic FP32 x0, shared across all conditions/backbones.
Wrong frames use a fixed non-self permutation independent of batch size.
The five time-bin centers are fixed quadrature probes, not dense integration.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fpsgen.datasets.dataloader.semantic_kitti import TemporalKITTISet
from fpsgen.models.gen_img import FlowIMG
from fpsgen.utils.collations import SparseSegmentCollationGen
from scripts.eval_bev_condition_usage import apply_mode
from scripts.eval_bevflow_lidar_only import _load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = _load_manifest(args.manifest)
    if len(frames) < 2 or len(set(frames)) != len(frames):
        raise ValueError("Unique frames and non-self shuffle required")
    state = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_epoch, checkpoint_step = state.get('epoch'), state.get('global_step')
    cfg = copy.deepcopy(state["hyper_parameters"])
    cfg.setdefault("runtime", {})["compile_dense"] = False
    model = FlowIMG(cfg)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.cuda().eval()
    dataset = TemporalKITTISet(str(args.data_root), ["08"], "test", .05, 180000, 50., gt_dir="gt_possion")
    lookup = {Path(p).stem: i for i,p in enumerate(dataset.points_datapath)}
    loader = DataLoader(Subset(dataset, [lookup[f] for f in frames]), batch_size=1,
                        num_workers=0, collate_fn=SparseSegmentCollationGen())
    cached = []
    with torch.no_grad():
        for batch in loader:
            part, full, labels = (batch[k].cuda() for k in ("pcd_part", "pcd_full", "full_label"))
            raw = model.model.get_raw_pc_bev(part)
            layout = model.processor.get_layout_bev(full, labels)*2-1
            target = model.processor.points_to_bev_target(full)
            cached.append(tuple(v.cpu() for v in (raw, layout, target)))
    rows = []
    weights = torch.tensor([1.,2.,1.], device="cuda").view(1,3,1,1)
    with torch.no_grad():
        for i, frame in enumerate(frames):
            raw, layout, target = (v.cuda() for v in cached[i])
            wrong_index = (i + (17 if len(frames)%17 else 1)) % len(frames)
            assert wrong_index != i
            wrong_raw, wrong_layout = (v.cuda() for v in cached[wrong_index][:2])
            generator = torch.Generator(device="cuda").manual_seed(20261001+int(frame))
            x0 = torch.randn(target.shape, device="cuda", generator=generator)
            u = target-x0
            for time in (.1,.3,.5,.7,.9):
                t = torch.full((1,), time, device="cuda")
                xt = (1-time)*x0+time*target
                zero = model.model(xt,t,torch.zeros_like(raw),torch.zeros_like(layout))
                z = float(((zero.float()-u).square()*weights).mean())
                for mode in ("100","010","001","111"):
                    correct = model.model(xt,t,*apply_mode(raw,layout,mode))
                    wrong = model.model(xt,t,*apply_mode(wrong_raw,wrong_layout,mode))
                    c = float(((correct.float()-u).square()*weights).mean())
                    s = float(((wrong.float()-u).square()*weights).mean())
                    rows.append({"frame":frame,"wrong_frame":frames[wrong_index],"mode":mode,"time":time,
                                 "correct":c,"zero":z,"shuffle":s,"delta_shuffle":s-c,
                                 "g_shuffle":(s-c)/(s+1e-8),"g_zero":(z-c)/(z+1e-8)})
            if (i+1)%10 == 0:
                print(f"CONDITION {i+1}/{len(frames)} frame={frame}",flush=True)
    if not all(np.isfinite(float(r[k])) for r in rows for k in ("correct","zero","shuffle")):
        raise FloatingPointError("Non-finite condition evaluation")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix(".csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={"checkpoint":str(args.checkpoint),"manifest":str(args.manifest),"frames":len(frames),
             "checkpoint_epoch":checkpoint_epoch,"checkpoint_global_step":checkpoint_step,
             "precision":"fp32 eager shared evaluator", "time_probe_policy":"fixed bin centers", "modes":{}}
    for mode in ("100","010","001","111"):
        summary["modes"][mode]={}
        for cutoff in (.2,.4,1.):
            subset=[r for r in rows if r["mode"]==mode and r["time"]<cutoff]
            c,s,z=(np.mean([r[k] for r in subset]) for k in ("correct","shuffle","zero"))
            summary["modes"][mode][str(cutoff)]={"g_shuffle":float((s-c)/(s+1e-8)),
                "g_zero":float((z-c)/(z+1e-8)),"loss_correct":float(c),"loss_shuffle":float(s)}
        summary["modes"][mode]["by_time_bin"]={}
        for j in range(5):
            subset=[r for r in rows if r["mode"]==mode and abs(r["time"]-(j*.2+.1))<1e-6]
            c,s,z=(float(np.mean([r[k] for r in subset])) for k in ("correct","shuffle","zero"))
            summary["modes"][mode]["by_time_bin"][f"{j*.2:.1f}-{(j+1)*.2:.1f}"]={
                "correct":c,"shuffle":s,"zero":z,"delta_shuffle":s-c,
                "g_shuffle":(s-c)/(s+1e-8),"g_zero":(z-c)/(z+1e-8)}
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()

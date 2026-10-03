#!/usr/bin/env python3
"""Same-card Euler10/CFG2, B1 and B8; 20 warmup and 100 timed generations."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fpsgen.datasets.dataloader.semantic_kitti import TemporalKITTISet
from fpsgen.models.gen_img import FlowIMG
from fpsgen.utils.collations import SparseSegmentCollationGen
from scripts.eval_bevflow_lidar_only import _load_manifest, sample_bev_cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--dense-backend",choices=("eager","inductor"),default="inductor")
    args=parser.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu")
    cfg=copy.deepcopy(state["hyper_parameters"])
    cfg.setdefault("runtime",{}).update(compile_dense=args.dense_backend=="inductor",compile_backend="inductor")
    model=FlowIMG(cfg);model.load_state_dict(state["state_dict"],strict=True);del state
    model.cuda().eval();model._eval_dense_forward=model._dense_forward;model._eval_fp16=True
    data=TemporalKITTISet(args.data_root,["08"],"test",.05,180000,50.,gt_dir="gt_possion")
    lookup={Path(p).stem:i for i,p in enumerate(data.points_datapath)}
    frames=_load_manifest(args.manifest)[:8]
    loader=DataLoader(Subset(data,[lookup[f] for f in frames]),batch_size=8,num_workers=0,
                      collate_fn=SparseSegmentCollationGen())
    batch=next(iter(loader));part=batch["pcd_part"].cuda()
    result={"gpu":torch.cuda.get_device_name(),"checkpoint":str(args.checkpoint),
            "precision":"fp16 dense / fp32 geometry", "backend":args.dense_backend, "euler_steps":10,
            "condition":"100", "cfg":2, "warmup_generations":20, "timed_generations":100,
            "includes_frontend":True,"batches":{}}
    with torch.no_grad():
        for size in (1,8):
            inputs=part[:size]
            layout=torch.zeros((size,2,256,256),device="cuda")
            startup=time.perf_counter();durations=[]
            for index in range(120):
                torch.cuda.synchronize();start=time.perf_counter()
                raw=model.model.get_raw_pc_bev(inputs)
                sample_bev_cfg(model,raw,layout,steps=10,guidance_scale=2,seed=20261001)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                if index==0:
                    first=time.perf_counter()-startup
                if index==20:
                    torch.cuda.reset_peak_memory_stats()
                if index>=20:
                    durations.append(elapsed)
            mean=float(np.mean(durations))
            result["batches"][str(size)]={"first_generation_including_compile_sec":first,
                "mean_ms_per_generation":mean*1000, "mean_ms_per_sample":mean*1000/size,
                "samples_per_second":size/mean,"p50_ms":float(np.median(durations))*1000,
                "p95_ms":float(np.percentile(durations,95))*1000,
                "peak_allocated_mb":torch.cuda.max_memory_allocated()/2**20,
                "peak_reserved_mb":torch.cuda.max_memory_reserved()/2**20}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

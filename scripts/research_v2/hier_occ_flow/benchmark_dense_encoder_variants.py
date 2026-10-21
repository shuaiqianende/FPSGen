#!/usr/bin/env python3
"""RTX 3090 microbenchmark for Stage0 R0/R1/R2 dense blocks.

Uses the production Stage0 tensor shape and supports forward-only inference or
forward+backward training measurements. The default protocol is 20 warmups and
100 timed iterations, with all GPU timing synchronized around each sample.
"""
import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fpsgen.hier_occ_flow.dense_encoder_variants import (
    DenseBottleneck3D, ObservationModulatedDense3D,
)
from fpsgen.hier_occ_flow.observation_context import build_observation_context


def original_block(norm_kind, *, first, drop_path_rate):
    from mmcv.cnn import ConvModule
    from timm.layers import DropPath
    from fpsgen.voxel_point.vendor.voxdet.resnet3d import BasicBlock3D
    norm_cfg = dict(type="BN3d") if norm_kind == "bn" else dict(type="GN", num_groups=32)
    skip = (ConvModule(128, 128, kernel_size=3, stride=1, padding=1,
                       bias=False, conv_cfg=dict(type="Conv3d"),
                       norm_cfg=norm_cfg, act_cfg=None) if first else None)
    return BasicBlock3D(128, 128, stride=1, downsample=skip,
                        drop_path_rate=drop_path_rate)


def blocks_for(norm_kind, first):
    original = original_block(norm_kind, first=first,
                              drop_path_rate=0.05 if not first else 0.)
    bottleneck = DenseBottleneck3D.from_original(original)
    omd = ObservationModulatedDense3D.from_original(original)
    return {"r0_original": original, "r1_bottleneck": bottleneck,
            "r2_omd3d": omd}


def make_context(batch, device, dtype, input_path):
    import numpy as np
    from fpsgen.hier_occ_flow.stage1_network import OccupancyGridSpec, occupancy_lidar_features
    spec=OccupancyGridSpec(x_min=-50.,y_min=-50.,z_min=-4.,x_max=50.,y_max=50.,
                           z_max=4.4,nx=384,ny=384,nz=32,xy_radius=50.,sphere_radius=50.)
    raw=np.load(input_path,allow_pickle=False)
    xyz=torch.as_tensor(np.array(raw[:,:3],dtype=np.float32,copy=True),device=device)
    _,indices=occupancy_lidar_features(xyz,spec)
    return build_observation_context([indices[0][0] for _ in range(batch)],
                                     (192,192,16),device=device,dtype=dtype)


def synchronize(device):
    torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--input-npy", type=Path, default=Path("/data-12/M2024-HWZ/KITTI_Odometry/00/input_/000100.npy"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--variants", nargs="+", default=["r0_original", "r1_bottleneck", "r2_omd3d"])
    parser.add_argument("--precisions", nargs="+", choices=["amp_fp16", "explicit_fp16"],
                        default=["amp_fp16", "explicit_fp16"])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    device = torch.device(args.device)
    if device.index not in (0, 1):
        raise ValueError("CUDA_VISIBLE_DEVICES must expose only physical GPU2/3")
    physical_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if args.physical_gpu not in physical_ids:
        raise ValueError("--physical-gpu and CUDA_VISIBLE_DEVICES disagree")
    torch.cuda.set_device(device)
    if not args.input_npy.is_file():
        raise FileNotFoundError(args.input_npy)
    results = []
    for batch in args.batches:
        for norm_kind, block_kind in (("bn", "voxel"), ("gn", "local")):
            synchronize(device); context_start=time.perf_counter()
            context = make_context(batch, device, torch.float32, args.input_npy)
            synchronize(device); context_build_ms=(time.perf_counter()-context_start)*1000
            for block_position in ("first", "regular"):
                if block_position == "regular":
                    # Production stage0 DropPath rates are .05 for voxel block1,
                    # .0333/.0667 for local blocks1/2. Use .05 representative.
                    first = False
                    if block_kind == "local":
                        continue  # the voxel and local regular topology is same
                else:
                    first = True
                blocks = blocks_for(norm_kind, first)
                for variant in args.variants:
                    for precision in args.precisions:
                        module = blocks[variant].to(device).train()
                        if precision == "explicit_fp16":
                            module = module.half()
                            x = torch.randn(batch,128,192,192,16,device=device,dtype=torch.float16)
                            amp = False
                            ctx = ObservationContextCast(context, torch.float16)
                        else:
                            x = torch.randn(batch,128,192,192,16,device=device,dtype=torch.float32)
                            amp = True
                            ctx = context
                        if variant != "r2_omd3d":
                            ctx_arg = None
                        else:
                            ctx_arg = ctx
                        def run():
                            module.zero_grad(set_to_none=True)
                            if x.grad is not None:
                                x.grad = None
                            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                                y = module(x) if variant == "r0_original" else module(x, ctx_arg)
                                loss = y.float().square().mean()
                            if args.backward:
                                loss.backward()
                            return y
                        for _ in range(args.warmups):
                            run()
                        synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        timings=[]
                        for _ in range(args.repeats):
                            synchronize(device); start=time.perf_counter()
                            run(); synchronize(device)
                            timings.append((time.perf_counter()-start)*1000)
                        params=sum(p.numel() for p in module.parameters())
                        results.append({"variant":variant,"norm_kind":norm_kind,
                            "block_position":block_position,"batch":batch,
                            "precision":precision,"backward":args.backward,
                            "input_shape":[batch,128,192,192,16],"parameters":params,
                            "mean_ms":statistics.mean(timings),"median_ms":statistics.median(timings),
                            "std_ms":statistics.pstdev(timings),"min_ms":min(timings),"max_ms":max(timings),
                            "peak_allocated_gib":torch.cuda.max_memory_allocated(device)/2**30,
                            "peak_reserved_gib":torch.cuda.max_memory_reserved(device)/2**30,
                            "observation_context_build_ms":context_build_ms if variant=="r2_omd3d" else None})
                        del module,x,ctx
                        torch.cuda.empty_cache()
    payload={"physical_gpu":args.physical_gpu,"device":torch.cuda.get_device_name(device),
             "torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
             "warmups":args.warmups,"repeats":args.repeats,"backward":args.backward,
             "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2))


class ObservationContextCast:
    def __init__(self, context, dtype):
        self.mask=context.mask.to(dtype=dtype)
        self.rho_x=context.rho_x.to(dtype=dtype)
        self.rho_y=context.rho_y.to(dtype=dtype)
        self.rho_z=context.rho_z.to(dtype=dtype)

    @property
    def features(self):
        return torch.cat((self.mask,self.rho_x,self.rho_y,self.rho_z),1)


if __name__ == "__main__":
    main()

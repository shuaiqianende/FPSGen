#!/usr/bin/env python3
"""Benchmark randomly initialized R0/R1/R2 Stage1 encoder inference."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--input-npy',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['original','bottleneck','omd3d'])
    p.add_argument('--precision',choices=['amp','explicit_fp16'],default='amp')
    p.add_argument('--warmups',type=int,default=20)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    visible=[int(v) for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v]
    device=torch.device(a.device)
    if not torch.cuda.is_available() or a.physical_gpu not in visible or device.index not in (0,1):
        raise RuntimeError('CUDA_VISIBLE_DEVICES / --physical-gpu mismatch or CUDA unavailable')
    if not a.input_npy.is_file(): raise FileNotFoundError(a.input_npy)
    torch.cuda.set_device(device)
    xyz_np=np.load(a.input_npy,allow_pickle=False)
    xyz=torch.as_tensor(np.array(xyz_np[:,:3],dtype=np.float32,copy=True),device=device)
    geom=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    results=[]
    for variant in a.variants:
        torch.manual_seed(20261021)
        model=Stage1DirectOccupancy(geom,objective='fm',widths=(8,16,32,64),
            activation_checkpointing=False,encoder_variant=variant,
            omd_config=({'internal_channels':64,'axial_kernels':(7,7,3),
                         'context_scale_init':1e-3,'use_observation_gate':True}
                        if variant=='omd3d' else None)).to(device).eval()
        if a.precision=='explicit_fp16':
            model.encoder.half();model.encoder._v6_explicit_fp16=True
        amp=a.precision=='amp'
        def run():
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                return model.encode_condition(xyz)
        for _ in range(a.warmups):
            out=run()
        torch.cuda.synchronize(device)
        shapes=[list(x.shape) for x in out]
        dtypes=[str(x.dtype) for x in out]
        del out
        torch.cuda.reset_peak_memory_stats(device)
        wall=[];events=[]
        for _ in range(a.repeats):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device);tick=time.perf_counter();start.record()
            out=run();end.record();torch.cuda.synchronize(device)
            wall.append((time.perf_counter()-tick)*1000)
            events.append(start.elapsed_time(end))
            del out
        results.append({'variant':variant,'precision':a.precision,'input':str(a.input_npy),
            'input_points':int(len(xyz)),'pyramid_shapes':shapes,'pyramid_dtypes':dtypes,
            'encoder_wall_mean_ms':statistics.mean(wall),'encoder_wall_median_ms':statistics.median(wall),
            'encoder_wall_p95_ms':float(np.percentile(wall,95)),
            'encoder_cuda_event_mean_ms':statistics.mean(events),
            'encoder_cuda_event_median_ms':statistics.median(events),
            'encoder_cuda_event_p95_ms':float(np.percentile(events,95)),
            'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,
            'parameters':sum(p.numel() for p in model.encoder.parameters()),
            'floating_parameter_bytes':sum(p.numel()*p.element_size() for p in model.encoder.parameters() if p.is_floating_point())})
        del model
        torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'physical_gpu':a.physical_gpu,
        'device_name':torch.cuda.get_device_name(device),'torch':torch.__version__,
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'warmups':a.warmups,'repeats':a.repeats,'results':results},indent=2))


if __name__=='__main__': main()

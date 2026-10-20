#!/usr/bin/env python
"""Diagnostic PTv3 profile on three real Seq08 scenes (not a benchmark)."""
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
sys.path.insert(0, str(HERE))

from benchmark_hs_patchscene_inference import (
    S1_SHA, S2_SHA, _admit, _run_hs_frame,
)
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.storage import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physical-gpu', type=int, choices=(2, 3), default=2)
    p.add_argument('--data-root', type=Path, default=Path('/data-12/M2024-HWZ/KITTI_Odometry'))
    p.add_argument('--config', type=Path, default=ROOT/'configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml')
    p.add_argument('--stage2-checkpoint', type=Path, default=Path('/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/checkpoints/step025000.pt'))
    p.add_argument('--stage1-checkpoint', type=Path, default=Path('/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_preflight/stage1_final_100kframes.pt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, nargs=3, default=(2300, 700, 100), help='low, median, high token scenes from V2 profile')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.physical_gpu):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the selected physical GPU only')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    admission = _admit(a.physical_gpu)
    if file_sha(a.stage1_checkpoint) != S1_SHA or file_sha(a.stage2_checkpoint) != S2_SHA:
        raise ValueError('Checkpoint SHA does not match the frozen V2 protocol')
    cfg, geom = load_stage2_config(a.config)
    device = torch.device('cuda:0')
    adapter = FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(device).eval()
    model = Stage2OccupancyModel(geom, **cfg['model']).to(device).eval()
    ckpt = torch.load(a.stage2_checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)

    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point, SerializedAttention, SerializedPooling, SerializedUnpooling
    events = []
    cpu = {}

    def wrap_method(owner, name, label):
        orig = getattr(owner, name)
        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); start.record()
            out = orig(*args, **kwargs)
            end.record(); cpu.setdefault(label, []).append((time.perf_counter()-t0)*1000)
            events.append((label, start, end))
            return out
        setattr(owner, name, wrapped)
    wrap_method(Point, 'serialization', 'point_serialization')
    wrap_method(Point, 'sparsify', 'point_sparsify')

    # Module hooks report inclusive CUDA time by module family. Parent modules
    # overlap their children, so these categories are diagnostic and must not
    # be summed as an exclusive wall-time decomposition.
    families = {'attention': SerializedAttention, 'pooling': SerializedPooling, 'unpooling': SerializedUnpooling}
    for family, cls in families.items():
        for mod in model.modules():
            if isinstance(mod, cls):
                pending = []
                def pre(m, inp, family=family, pending=pending):
                    s = torch.cuda.Event(enable_timing=True); s.record(); pending.append((s, time.perf_counter()))
                def post(m, inp, out, family=family, pending=pending):
                    s, t0 = pending.pop(); e = torch.cuda.Event(enable_timing=True); e.record()
                    events.append((family, s, e)); cpu.setdefault(family, []).append((time.perf_counter()-t0)*1000)
                mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)
    sparse_types = set()
    try:
        import spconv.pytorch as spconv
        for attr in ('SubMConv3d','SparseConv3d','SparseInverseConv3d'):
            if hasattr(spconv, attr): sparse_types.add(getattr(spconv, attr))
    except Exception:
        pass
    for mod in model.modules():
        if any(isinstance(mod, typ) for typ in sparse_types):
            pending=[]
            def pre(m, inp, pending=pending):
                s=torch.cuda.Event(enable_timing=True); s.record(); pending.append((s,time.perf_counter()))
            def post(m, inp, out, pending=pending):
                s,t0=pending.pop(); e=torch.cuda.Event(enable_timing=True); e.record()
                events.append(('spconv_module',s,e)); cpu.setdefault('spconv_module',[]).append((time.perf_counter()-t0)*1000)
            mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)

    rows=[]
    # One untimed real-frame warmup to stabilize kernels and module state.
    warm = input_xyz(a.data_root/'08'/'input_'/'000700.npy').astype(np.float32,copy=False)
    _run_hs_frame(warm,700,adapter,model,cfg,geom,device,gpu_candidate_selection=True,
                  gpu_observed_mask=True,gpu_occupancy_decode=True,stage2_static_cache=True)
    torch.cuda.synchronize(device); events.clear(); cpu.clear()
    for frame, label in zip(a.frames, ('low','median','high')):
        arr=input_xyz(a.data_root/'08'/'input_'/f'{frame:06d}.npy').astype(np.float32,copy=False)
        _, timing = _run_hs_frame(arr,frame,adapter,model,cfg,geom,device,gpu_candidate_selection=True,
                                  gpu_observed_mask=True,gpu_occupancy_decode=True,stage2_static_cache=True)
        torch.cuda.synchronize(device)
        grouped={}
        for name,s,e in events:
            grouped.setdefault(name,[]).append(float(s.elapsed_time(e)))
        rows.append({'label':label,'frame':frame,'candidate_parents':timing['candidate_parents'],
                     'generation_wall_ms':timing['generation_wall_ms'],'stage2_total_ms':timing['stage2_network_total_ms'],
                     'gpu_event_ms':{k:sum(v) for k,v in grouped.items()},
                     'gpu_event_calls':{k:len(v) for k,v in grouped.items()},
                     'cpu_submit_wall_ms':{k:sum(v) for k,v in cpu.items()},
                     'cpu_submit_calls':{k:len(v) for k,v in cpu.items()}})
        events.clear();cpu.clear()
        print(json.dumps(rows[-1]),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'admission':admission,'torch':torch.__version__,'cuda':torch.version.cuda,
        'stage1_sha256':S1_SHA,'stage2_sha256':S2_SHA,'note':'Nested module CUDA events are inclusive and not additive.',
        'rows':rows},indent=2)+'\n')

if __name__=='__main__': main()

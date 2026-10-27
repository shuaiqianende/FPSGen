"""Standalone Stage1 timing without quality hooks, GT metrics or timed disk IO."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,time
import numpy as np
import torch
from fpsgen.hier_occ_flow.meanflow_v2_training import load_config,make_model,admission,own_sha
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha
from fpsgen.hier_occ_flow.meanflow_sampling import sample_meanflow,stage1_score
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

WARMUP=[0,800,1600,2400,3200]
TIMED=[100,300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500,2700,2900,3100,3300,3500,3700,3900]


def stats(x):
    a=np.asarray(x)
    return dict(mean=float(a.mean()),median=float(np.median(a)),std=float(a.std()),p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,g=load_config(a.config);gpu=admission(cfg)
    if a.output.exists():raise FileExistsError('Preserve prior benchmark')
    a.output.mkdir(parents=True)
    saved=torch.load(a.checkpoint,map_location='cpu')
    if saved['counts']['updates']!=10000 or saved['identity']['source_sha256']!=own_sha() or saved['identity']['config_sha256']!=config_sha(cfg):raise ValueError('Own final20K model required')
    model=make_model(cfg,g,'cuda');model.load_state_dict(saved['ema'],strict=True);apply_explicit_policy(model);model.eval().requires_grad_(False);del saved
    # All loading and H2D precede formal timing. No input subsampling.
    xyz={f:torch.from_numpy(input_xyz(Path(cfg['data_root'])/'08/input_'/f'{f:06d}.npy')).cuda() for f in WARMUP+TIMED}
    rows=[]
    with torch.no_grad():
        for nfe in (1,2,4,8):
            for frame in WARMUP+TIMED:
                events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
                torch.cuda.synchronize();begin=time.perf_counter()
                events[0].record()
                with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
                    with torch.autocast('cuda',enabled=False):pyramid=model.stage1.encoder([xyz[frame]])
                    events[1].record();observed=model.stage1.geometry.observed_mask(xyz[frame]);valid=model.stage1.valid
                    source=torch.randn(valid.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',frame,20261031)))*valid
                    events[2].record()
                    fn=lambda x,t,s:model.velocity(x,observed,pyramid,t,s-t)*valid
                    endpoint=sample_meanflow(fn,source,valid,nfe=nfe,scenes=1)
                    events[3].record();score=stage1_score(endpoint)*valid
                    selected=torch.nonzero(score.reshape(-1)>.5).flatten()
                    events[4].record()
                torch.cuda.synchronize();wall=1000*(time.perf_counter()-begin)
                if frame in TIMED:
                    rows.append(dict(frame=frame,nfe=nfe,stage1_generation_wall_ms=wall,
                        encoder_cuda_ms=events[0].elapsed_time(events[1]),condition_cuda_ms=events[1].elapsed_time(events[2]),
                        flow_cuda_ms=events[2].elapsed_time(events[3]),decode_cuda_ms=events[3].elapsed_time(events[4]),active_parents=len(selected),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    atomic_json(a.output/'timing.json',dict(scope='Stage1 only, GPU-resident input and Parent IDs; no Stage2 or Jitter; one5+20 round per NFE',gpu=gpu,
        source_sha256=own_sha(),checkpoint_sha256=file_sha(a.checkpoint),warmup=WARMUP,timed=TIMED,rows=rows,
        summary={str(n):{k:stats([r[k] for r in rows if r['nfe']==n]) for k in ('stage1_generation_wall_ms','encoder_cuda_ms','flow_cuda_ms')} for n in (1,2,4,8)}))


if __name__=='__main__':main()

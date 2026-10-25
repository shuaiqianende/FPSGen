"""Exclusive, same-GPU3-round paired wall timing; no profiler/quality metrics."""
from common import ROOT
import argparse,csv,json,os
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.meanflow_training import load_config,admission
from fpsgen.hier_occ_flow.meanflow_evaluation import EvaluationModels
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import atomic_json

WARM=[0,800,1600,2400,3200]
TIMED=list(range(100,4000,200))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg,g=load_config(a.config)
    physical=int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'))
    if physical not in (2,3):raise ValueError('GPU2/3 only')
    report=gpu_preflight((physical,),minimum_free_gib=16.,allow_shared_compute=False)
    if report['status']!='AVAILABLE':
        atomic_json(a.output/'status.json',dict(status='DEFERRED_EXCLUSIVE_GPU',admission=report));return
    if (a.output/'per_frame_timing.csv').exists():raise FileExistsError('Preserve completed/partial paired timing')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    models=EvaluationModels(cfg,g,a.checkpoint,'cuda')
    # Input IO is outside generation wall. Retain only25 original scans.
    xyz={f:torch.from_numpy(input_xyz(Path(cfg['data_root'])/'08/input_'/f'{f:06d}.npy')).cuda() for f in set(WARM+TIMED)}
    atomic_json(a.output/'protocol.json',dict(warmup_frames=WARM,timed_frames=TIMED,rounds=3,physical_gpu=physical,
        backend='same originalAMP PyTorch, multilevel topology, native centers',checkpoint_sha256=models.checkpoint_sha,
        controls=models.control_audit,nfes=[1,2,4,8],no_quality_hooks=True,no_checkpoint_io_in_timing=True))
    rows=[]
    for nfe in (1,2,4,8):
        for round_id,order in enumerate((('matched_fm','meanflow'),('meanflow','matched_fm'),('matched_fm','meanflow')),1):
            for variant in order:
                for f in WARM:models.generate(xyz[f],'08',f,variant=variant,nfe=nfe)
                torch.cuda.reset_peak_memory_stats()
                for f in TIMED:
                    out=models.generate(xyz[f],'08',f,variant=variant,nfe=nfe)
                    row=dict(nfe=nfe,round=round_id,variant=variant,frame=f,**out['timing'],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    rows.append(row)
                # If another job arrives, invalidate this attempt instead of
                # silently publishing contended timings as a formal benchmark.
                live=gpu_preflight((physical,),minimum_free_gib=0.,allow_shared_compute=True)
                others=[pid for pid in live['gpus'][0]['existing_compute_pids'] if pid!=os.getpid()]
                if others:raise RuntimeError(f'New compute jobs {others}; paired timing invalid, retain raw evidence')
    with (a.output/'per_frame_timing.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[];rng=np.random.default_rng(20261032)
    for nfe in (1,2,4,8):
        for variant in ('matched_fm','meanflow'):
            r=[x for x in rows if x['nfe']==nfe and x['variant']==variant];v=np.array([x['generation_wall_ms'] for x in r])
            summary.append(dict(nfe=nfe,variant=variant,mean_ms=float(v.mean()),p50_ms=float(np.median(v)),std_ms=float(v.std()),
                p90_ms=float(np.quantile(v,.9)),p95_ms=float(np.quantile(v,.95)),min_ms=float(v.min()),max_ms=float(v.max())))
    paired=[]
    for nfe in (1,2,4,8):
        lookup={(r['round'],r['frame'],r['variant']):r['generation_wall_ms'] for r in rows if r['nfe']==nfe}
        values=np.array([[lookup[(j,f,'meanflow')]-lookup[(j,f,'matched_fm')] for f in TIMED] for j in (1,2,3)])
        resamples=[]
        for _ in range(2000):
            ix=rng.integers(0,len(TIMED),len(TIMED));resamples.append(values[:,ix].mean())
        paired.append(dict(nfe=nfe,candidate_minus_control_ms=float(values.mean()),frame_clustered_95ci=list(map(float,np.quantile(resamples,[.025,.975]))),
            round_delta_means=list(map(float,values.mean(1))),delta_std_ms=float(values.std()),claim_scope='three paired rounds, same20scenes; one training seed'))
    atomic_json(a.output/'results.json',dict(status='BENCHMARKED',rows=len(rows),summary=summary,paired=paired,
        speed_claim='Use all three rounds and per-frame paired differences; no fastest-round selection'))
    atomic_json(a.output/'status.json',dict(status='BENCHMARKED'))

if __name__=='__main__':main()

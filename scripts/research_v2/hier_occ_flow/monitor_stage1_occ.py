"""Real wall-clock observation, beginning only after actual successful updates.

Creating this script does NOT satisfy the30-minute observation gate. --observe
must remain running and collect live logs/GPU snapshots for the requested time.
"""
from common import ROOT
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from fpsgen.hier_occ_flow.storage import atomic_json


def read_rows(path):
    rows=[]
    if not path.exists():return rows
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass  # writer's current incomplete line
    return rows


def live_processes(output):
    process=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    return [line.strip() for line in process.splitlines() if 'train_stage1_occ.py' in line and str(output) in line]


def snapshot(output):
    rows=read_rows(output/'train_rank0.jsonl');successful=[r for r in rows if r.get('finite') and r['successful_updates']>0]
    recent=successful[-100:];losses=[r['loss'] for r in recent if r.get('loss') is not None]
    gpu=subprocess.run(['nvidia-smi','--id=2,3','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    result={'epoch_time':time.time(),'processes':live_processes(output),'gpu':gpu.stdout.strip(),'gpu_error':gpu.stderr.strip(),
        'disk_free_bytes':shutil.disk_usage(output if output.exists() else ROOT).free,
        'successful_updates':successful[-1]['successful_updates'] if successful else 0,
        'latest':successful[-1] if successful else None,'actual_training_observed':bool(successful),
        'last100_loss_mean':float(np.mean(losses)) if losses else None,
        'last50_loss_mean':float(np.mean(losses[-50:])) if losses else None,
        'loss_p50':float(np.percentile(losses,50)) if losses else None,'loss_p95':float(np.percentile(losses,95)) if losses else None,
        'mean_step_seconds':float(np.mean([r['seconds'] for r in recent])) if recent else None,
        'attempts':rows[-1]['attempts'] if rows else 0,'gradient_audits':[r for r in recent if r.get('gradient_details')],
        'checkpoint_exists':(output/'checkpoints/resume_latest.pt').exists()}
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--observe',action='store_true');p.add_argument('--minutes',type=float,default=30)
    args=p.parse_args();events=[];begin=None
    if args.observe and args.minutes<30:raise ValueError('Formal observation gate requires at least30 minutes')
    while True:
        row=snapshot(args.output);events.append(row)
        if begin is None and row['actual_training_observed']:begin=time.monotonic()
        duration=0 if begin is None else time.monotonic()-begin
        status='NOT_STARTED' if begin is None else 'OBSERVING'
        if args.observe and (begin is None or not row['processes']):status='BLOCKED_NO_LIVE_FORMAL_TRAINING'
        if duration>=args.minutes*60:status='OBSERVED_REQUESTED_DURATION'
        atomic_json(args.report,{'status':status,'actual_observed_seconds':duration,'events':events,
            'formal_training_started':row['actual_training_observed'],'process_alive':bool(row['processes']),
            'first30min_complete':args.observe and duration>=1800 and bool(row['processes'])})
        print(f"status={status} observed_seconds={duration:.1f} update={row['successful_updates']}",flush=True)
        if not args.observe or status.startswith('BLOCKED') or status=='OBSERVED_REQUESTED_DURATION':break
        time.sleep(60)

if __name__=='__main__':main()

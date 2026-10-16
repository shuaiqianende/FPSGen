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
    process=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    paths=(str(output),os.path.relpath(output.resolve(),ROOT))
    return [line.strip() for line in process.splitlines() if len(line.split(None,2))==3
        and line.split(None,2)[1].startswith('python') and 'train_stage1_occ.py' in line
        and any(path in line for path in paths)]


def snapshot(output):
    rows=read_rows(output/'train_rank0.jsonl');successful=[r for r in rows if r.get('finite') and r['successful_updates']>0]
    recent=successful[-100:];losses=[r['loss'] for r in recent if r.get('loss') is not None]
    loss_ema=None
    for r in successful:
        if r.get('loss') is not None:loss_ema=r['loss'] if loss_ema is None else .95*loss_ema+.05*r['loss']
    gpu=subprocess.run(['nvidia-smi','--id=2,3','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    cfg=json.loads((output/'config.json').read_text()) if (output/'config.json').exists() else {}
    launch=json.loads((output/'launch_parameters.json').read_text()) if (output/'launch_parameters.json').exists() else {}
    formal=cfg.get('scope')=='formal' and launch.get('actual_update_budget')==100000
    result={'epoch_time':time.time(),'processes':live_processes(output),'gpu':gpu.stdout.strip(),'gpu_error':gpu.stderr.strip(),
        'disk_free_bytes':shutil.disk_usage(output if output.exists() else ROOT).free,
        'successful_updates':successful[-1]['successful_updates'] if successful else 0,
        'latest':successful[-1] if successful else None,'actual_training_observed':bool(successful) and formal,
        'scope':cfg.get('scope'),'formal_budget_verified':formal,'expected_world_size':launch.get('world_size',1),
        'last100_loss_mean':float(np.mean(losses)) if losses else None,
        'loss_ema_alpha005':loss_ema,
        'last50_loss_mean':float(np.mean(losses[-50:])) if losses else None,
        'loss_p50':float(np.percentile(losses,50)) if losses else None,'loss_p95':float(np.percentile(losses,95)) if losses else None,
        'mean_step_seconds':float(np.mean([r['seconds'] for r in recent])) if recent else None,
        'attempts':rows[-1]['attempts'] if rows else 0,'gradient_audits':[r for r in recent if r.get('gradient_details')],
        'invalid_gradient_attempts':sum(not r.get('finite',False) for r in rows),
        'nonfinite_loss_incidents':sum(r.get('loss') is None for r in rows),
        'updates_per_second_including_intervals':(recent[-1]['successful_updates']-recent[0]['successful_updates'])/(recent[-1]['epoch_time']-recent[0]['epoch_time']) if len(recent)>1 else None,
        'mean_data_seconds':float(np.mean([r['data_seconds'] for r in recent])) if recent else None,
        'latest_update_age_seconds':time.time()-successful[-1]['epoch_time'] if successful else None,
        'checkpoint_exists':(output/'checkpoints/resume_latest.pt').exists()}
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--observe',action='store_true');p.add_argument('--minutes',type=float,default=30)
    p.add_argument('--wait-for-start',action='store_true');p.add_argument('--start-timeout',type=float,default=300)
    args=p.parse_args();events=[];begin=None;waiting_since=time.monotonic()
    if args.observe and args.minutes<30:raise ValueError('Formal observation gate requires at least30 minutes')
    while True:
        row=snapshot(args.output);events.append(row)
        if args.observe and row['scope'] not in (None,'formal'):raise ValueError('Pilot/smoke observations cannot satisfy formal30min gate')
        if begin is None and row['actual_training_observed']:begin=time.monotonic()
        duration=0 if begin is None else time.monotonic()-begin
        status='NOT_STARTED' if begin is None else 'OBSERVING'
        if args.observe and (begin is None or not row['processes']):
            status='WAITING_FOR_FIRST_SUCCESSFUL_UPDATE' if begin is None and args.wait_for_start and time.monotonic()-waiting_since<args.start_timeout else 'BLOCKED_NO_LIVE_FORMAL_TRAINING'
        healthy=len(row['processes'])>=row['expected_world_size'] and row.get('latest_update_age_seconds') is not None and row['latest_update_age_seconds']<180 and row['checkpoint_exists']
        if duration>=args.minutes*60:status='OBSERVED_REQUESTED_DURATION' if healthy else 'OBSERVED_DURATION_UNHEALTHY'
        atomic_json(args.report,{'status':status,'actual_observed_seconds':duration,'events':events,
            'formal_training_started':row['actual_training_observed'],'process_alive':bool(row['processes']),
            'first30min_complete':args.observe and duration>=1800 and healthy})
        print(f"status={status} observed_seconds={duration:.1f} update={row['successful_updates']}",flush=True)
        if not args.observe or status.startswith('BLOCKED') or status.startswith('OBSERVED_'):break
        time.sleep(1 if begin is None else 60)

if __name__=='__main__':main()

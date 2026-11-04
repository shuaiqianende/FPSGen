"""Train position + late-LiDAR Stage2 after the current GPU3 run finishes."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/stage2_child_b4_followup_v1/attention'
PREVIOUS = Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-stage2-child-injection-v1/experiments/stage2_child_b4_followup_v1/attention/formal/status.json')


def status(value):
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = BASE / 'queue_status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, BASE / 'queue_status.json')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise ValueError('Only physical GPU3 is registered')
    while True:
        previous = json.loads(PREVIOUS.read_text()) if PREVIOUS.exists() else {}
        if previous.get('status') == 'COMPLETE' and previous.get('frames') == 100000:
            break
        if previous.get('status') in ('BLOCKED', 'FAILED', 'PAUSED'):
            raise RuntimeError(f'Preceding training requires review: {previous}')
        status({'status': 'WAITING_PREVIOUS_GPU3_TRAINING', 'previous': previous.get('status'),
                'step': previous.get('step'), 'frames': previous.get('frames')})
        time.sleep(30)
    while True:
        free, util = map(float, subprocess.check_output(['nvidia-smi', '--id=3',
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
        active = subprocess.check_output(['nvidia-smi', '--id=3', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if free >= 18000 and util <= 10 and not active:
            break
        status({'status': 'WAITING_GPU3', 'free_mib': free, 'utilization': util, 'compute_pids': active})
        time.sleep(30)
    train = ROOT / 'scripts/research_v2/hier_occ_flow/train_stage2_child_b4.py'
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    for mode in ('gate', 'smoke', 'formal'):
        path = BASE / mode / 'status.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        expected = 'COMPLETE' if mode == 'formal' else 'PASS'
        if previous.get('status') == expected:
            continue
        if previous.get('status') in ('BLOCKED', 'FAILED'):
            raise RuntimeError(f'{mode} failed previously; preserve output and investigate')
        command = [sys.executable, '-u', str(train), '--arm', 'attention', '--mode', mode]
        if mode == 'formal' and (BASE / mode / 'checkpoints/resume_latest.pt').exists():
            command.append('--resume')
        status({'status': 'RUNNING_' + mode.upper(), 'gpu': 3,
                'experiment': 'position_plus_late_encoder', 'batch': 4})
        with (BASE / (mode + '.log')).open('a') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(path.read_text())
        if result.get('status') != expected:
            raise RuntimeError(f'{mode} incomplete: {result}')
    status({'status': 'TRAIN_COMPLETE', 'gpu': 3,
            'experiment': 'position_plus_late_encoder', 'evaluation': 'GPU2, not yet queued'})


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        status({'status': 'FAILED', 'reason': repr(e), 'traceback': traceback.format_exc()})
        raise

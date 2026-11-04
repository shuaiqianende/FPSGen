"""Start scratch U1 only after both 408-frame evaluation queues complete."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / 'experiments/desescene_eval408_nextstage2_v1'
BASE = ROOT / 'experiments/desescene_u1_fulltrain_explicit_fp16_100k_v1'


def status(value):
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = BASE / 'queue_status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, BASE / 'queue_status.json')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('Expected physical GPU2')
    while True:
        path = EVAL / 'joint_queue.json'
        upstream = json.loads(path.read_text()) if path.exists() else {}
        if upstream.get('status') == 'JOINT_COMPLETE':
            break
        if upstream.get('status') == 'FAILED':
            raise RuntimeError('Evaluation failed; inspect before training')
        status({'status': 'WAITING_ALL_408_EVALUATIONS', 'upstream': upstream.get('status')})
        time.sleep(30)
    while True:
        free, util = map(float, subprocess.check_output(['nvidia-smi', '--id=2',
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
        active = subprocess.check_output(['nvidia-smi', '--id=2', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if free >= 18000 and util <= 10 and not active:
            break
        status({'status': 'WAITING_GPU2', 'free_mib': free, 'utilization': util, 'compute_pids': active})
        time.sleep(30)
    train = ROOT / 'scripts/research_v2/hier_occ_flow/train_u1_explicit_100k.py'
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    def run(mode, extra=()):
        status({'status': 'RUNNING_' + mode.upper(), 'gpu': 2, 'extra': list(extra)})
        with (BASE / (mode + '.log')).open('a') as log:
            subprocess.run([sys.executable, '-u', str(train), '--mode', mode, *extra],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for mode in ('gate', 'smoke', 'formal'):
        statefile = BASE / mode / 'status.json'
        state = json.loads(statefile.read_text()) if statefile.exists() else {}
        expected = 'COMPLETE' if mode == 'formal' else 'PASS'
        if state.get('status') == expected:
            continue
        if state.get('status') == 'BLOCKED':
            raise RuntimeError(f'{mode} failed and requires review: {state.get("reason")}')
        ck = BASE / mode / 'checkpoints/resume_latest.pt'
        if mode == 'smoke':
            if not ck.exists():
                run(mode, ('--max-updates', '16'))
            partial = json.loads(statefile.read_text())
            if partial.get('status') != 'PARTIAL' or partial.get('updates') != 16:
                raise ValueError('Expected a completed 16-update smoke checkpoint')
            run(mode, ('--resume', '--max-updates', '32'))
        else:
            run(mode, ('--resume',) if mode == 'formal' and ck.exists() else ())
        state = json.loads(statefile.read_text())
        if state.get('status') != expected:
            raise RuntimeError(f'{mode} incomplete: {state}')
    status({'status': 'COMPLETE', 'gpu': 2, 'successful_exposures': 100000,
            'evaluation': 'not yet scheduled'})


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        status({'status': 'FAILED', 'reason': repr(e), 'traceback': traceback.format_exc()})
        raise

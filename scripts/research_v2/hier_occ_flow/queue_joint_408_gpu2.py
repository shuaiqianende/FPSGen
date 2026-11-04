"""Serial GPU2 408-frame joint evaluations after Stage1 completes."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/desescene_eval408_nextstage2_v1'
STAGE1 = BASE / 'stage1_queue.json'
RUNS = ('with_encoder', 'no_encoder', 'position', 'late_encoder')
LATE = Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-stage2-child-injection-v1/experiments/stage2_child_b4_followup_v1/attention/formal/status.json')


def write_status(value):
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = BASE / 'joint_queue.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, BASE / 'joint_queue.json')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('Expected physical GPU2')
    while True:
        state = json.loads(STAGE1.read_text()) if STAGE1.exists() else {}
        if state.get('status') == 'STAGE1_COMPLETE':
            break
        if state.get('status') == 'FAILED':
            raise RuntimeError('Stage1 evaluation failed; inspect before continuation')
        write_status({'status': 'WAITING_STAGE1_408', 'upstream': state.get('status')})
        time.sleep(30)
    while True:
        free, util = map(float, subprocess.check_output(['nvidia-smi', '--id=2',
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
        active = subprocess.check_output(['nvidia-smi', '--id=2', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if free >= 18000 and util <= 10 and not active:
            break
        write_status({'status': 'WAITING_GPU2', 'free_mib': free, 'utilization': util, 'compute_pids': active})
        time.sleep(30)
    script = ROOT / 'scripts/research_v2/hier_occ_inference/eval_joint_seq08_408.py'
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    for arm in RUNS:
        if arm == 'late_encoder':
            late = json.loads(LATE.read_text()) if LATE.exists() else {}
            if late.get('status') != 'COMPLETE' or late.get('step') != 25000 or late.get('frames') != 100000:
                raise RuntimeError(f'Late-encoder training has not completed: {late}')
        for phase, count in (('smoke', 1), ('seq08_skip10_408', 408)):
            out = BASE / ('joint_' + arm) / phase
            result = out / 'results.json'
            if result.exists():
                existing = json.loads(result.read_text())
                if existing.get('status') == 'COMPLETE' and len(existing.get('rows', [])) == count:
                    continue
                raise ValueError(f'Invalid previous result {result}')
            write_status({'status': 'RUNNING_JOINT', 'variant': arm, 'phase': phase, 'frames': count})
            out.parent.mkdir(parents=True, exist_ok=True)
            with (out.parent / (phase + '.log')).open('a') as log:
                subprocess.run([sys.executable, '-u', str(script), '--arm', arm, '--output', str(out),
                    '--max-frames', str(count)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result_data = json.loads(result.read_text())
            if result_data.get('status') != 'COMPLETE' or len(result_data.get('rows', [])) != count:
                raise ValueError(f'Incomplete evaluation: {result}')
            if any(r.get('predicted_points') != 1080000 for r in result_data['rows']):
                raise ValueError('Joint output point count does not match protocol')
    write_status({'status': 'JOINT_COMPLETE', 'variants': list(RUNS), 'frames_per_variant': 408})


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        write_status({'status': 'FAILED', 'reason': repr(e), 'traceback': traceback.format_exc()})
        raise

"""Lightweight 20-minute evidence log; never kills or restarts workloads."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/desescene_eval408_nextstage2_v1/monitor.jsonl'
W = Path('/data-12/M2024-HWZ/FPSGen/worktrees')
PATHS = {
    'gpu2_formal': W / 'desescene-stage2-child-b4-followup-v1/experiments/stage2_child_b4_followup_v1/position/formal/status.json',
    'gpu3_formal': W / 'desescene-stage2-child-injection-v1/experiments/stage2_child_b4_followup_v1/attention/formal/status.json',
    'gpu3_previous_queue': W / 'desescene-stage2-child-injection-v1/experiments/stage2_child_b4_followup_v1/attention/queue_status.json',
    'stage1_408': ROOT / 'experiments/desescene_eval408_nextstage2_v1/stage1_queue.json',
    'joint_408': ROOT / 'experiments/desescene_eval408_nextstage2_v1/joint_queue.json',
    'u1_100k': ROOT / 'experiments/desescene_u1_fulltrain_explicit_fp16_100k_v1/queue_status.json',
    'gpu3_followup': ROOT / 'experiments/stage2_child_b4_followup_v1/attention/queue_status.json',
}


def inspect():
    result = {'epoch': time.time()}
    for name, path in PATHS.items():
        try:
            result[name] = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            result[name] = {'read_error': str(e)}
    for name, cmd in {
        'gpu': ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        'gpu_processes': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
        'tmux': ['tmux', 'list-sessions', '-F', '#S'],
    }.items():
        try:
            result[name] = subprocess.check_output(cmd, text=True, timeout=30).strip()
        except (OSError, subprocess.SubprocessError) as e:
            result[name] = {'read_error': str(e)}
    return result


if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with OUT.open('a') as f:
            f.write(json.dumps(inspect(), allow_nan=False) + '\n')
        time.sleep(1200)

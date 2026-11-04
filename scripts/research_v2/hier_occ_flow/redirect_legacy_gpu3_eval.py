"""Retire the old GPU3 eval scheduler after its unchanged training finishes.

Only the scheduler is temporarily stopped; its child training process is never
signalled. The requested joint evaluation runs in the separate GPU2 queue.
"""
import json
import os
import signal
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-stage2-child-injection-v1/experiments/stage2_child_b4_followup_v1/attention')
OUT = ROOT / 'experiments/desescene_eval408_nextstage2_v1/gpu3_legacy_handoff.json'
QUEUE_PID = 3842156
TRAIN_PID = 3848173


def proc(pid):
    stat = Path(f'/proc/{pid}/stat')
    if not stat.exists():
        return None
    text = stat.read_text()
    tail = text[text.rfind(')') + 2:].split()
    return {'state': tail[0], 'ppid': int(tail[1]), 'starttime': tail[19],
            'cmdline': Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}


def status(value):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps({'epoch': time.time(), **value}, indent=2) + '\n')
    os.replace(tmp, OUT)


def main():
    queue, train = proc(QUEUE_PID), proc(TRAIN_PID)
    if (not queue or not train or train['ppid'] != QUEUE_PID or
        'queue_stage2_child_b4.py --arm attention' not in queue['cmdline'] or
        'train_stage2_child_b4.py --arm attention --mode formal' not in train['cmdline']):
        raise RuntimeError('Original scheduler/training identity changed; no signal sent')
    queue_start, train_start = queue['starttime'], train['starttime']
    held = False
    try:
        while True:
            queue, train = proc(QUEUE_PID), proc(TRAIN_PID)
            if queue is None or queue['starttime'] != queue_start:
                raise RuntimeError('Original scheduler disappeared or PID was reused')
            if train is None or train['starttime'] != train_start:
                raise RuntimeError('Original trainer disappeared before controlled handoff')
            formal = json.loads((OLD / 'formal/status.json').read_text())
            if formal.get('status') in ('BLOCKED', 'FAILED', 'PAUSED'):
                raise RuntimeError(f'Original training requires review: {formal.get("status")}')
            with (OLD / 'formal/train.jsonl').open('rb') as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 16384))
                lines = log.read().splitlines()
            last = json.loads(lines[-1]) if lines else {}
            if not held and last.get('step', 0) >= 24900:
                # Stop only the waiting scheduler, not the CUDA training child.
                os.kill(QUEUE_PID, signal.SIGSTOP)
                held = True
            if held and formal.get('status') == 'COMPLETE' and formal.get('frames') == 100000 and train['state'] == 'Z':
                os.kill(QUEUE_PID, signal.SIGTERM)
                os.kill(QUEUE_PID, signal.SIGCONT)
                held = False
                status({'status': 'TRAIN_COMPLETE_LEGACY_GPU3_EVAL_RETIRED', 'queue_pid': QUEUE_PID,
                        'train_pid': TRAIN_PID, 'step': last.get('step'),
                        'replacement': 'GPU2 joint_408 late_encoder'})
                return
            status({'status': 'SCHEDULER_HELD_TRAINING_CONTINUES' if held else 'WAITING_NEAR_TRAIN_END',
                    'queue_pid': QUEUE_PID, 'train_pid': TRAIN_PID, 'training_step': last.get('step'),
                    'train_state': train['state'], 'formal_status': formal.get('status')})
            time.sleep(10 if held else 30)
    finally:
        if held:
            current = proc(QUEUE_PID)
            if current and current['starttime'] == queue_start:
                os.kill(QUEUE_PID, signal.SIGCONT)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        status({'status': 'FAILED_REVIEW_REQUIRED', 'reason': repr(e), 'traceback': traceback.format_exc()})
        raise

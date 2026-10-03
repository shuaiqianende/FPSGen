#!/usr/bin/env python3
"""Read-only GPU23/queue monitoring; never kills, restarts or changes training."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def snapshot(root):
    queue = read_json(root/'queue_status.json')
    stage = queue.get('stage', '')
    manifest = read_json(root/stage/'run_manifest_rank0.json') if stage else {}
    window = manifest.get('last_stall_window', {})
    gpu = subprocess.run([
        'nvidia-smi', '-i', '2,3',
        '--query-gpu=index,utilization.gpu,memory.used,clocks.current.sm,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'SwapFree'):
            memory[key] = int(value.split()[0])//1024
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        queue_status=queue.get('status', 'STARTING'), stage=stage,
        run_status=manifest.get('status'), epoch=manifest.get('epoch'),
        step=manifest.get('optimizer_steps'), skipped=manifest.get('skipped_optimizer_steps'),
        window_seconds=window.get('wall_seconds'),
        multi_second_gaps=window.get('multi_second_gaps'),
        max_host_gap_seconds=window.get('max_host_gap_seconds'),
        speed_warning=window.get('warning'), would_stop=window.get('would_stop'),
        cpu_load1=os.getloadavg()[0], mem_available_mb=memory.get('MemAvailable'),
        swap_free_mb=memory.get('SwapFree'),
        gpu23=gpu.stdout.strip().replace('\n', ' | ') if gpu.returncode==0 else gpu.stderr.strip())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30)
    args=parser.parse_args()
    root=args.root.resolve()
    repo=Path(__file__).resolve().parents[1]
    if not root.is_relative_to(repo/'outputs') or args.interval < 10:
        raise ValueError('Use a repository output root and a low-frequency interval>=10s')
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects the previous monitoring record.
    with (root/'live_monitor.csv').open('x', newline='') as handle:
        writer=None
        while True:
            row=snapshot(root)
            if writer is None:
                writer=csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row['queue_status'] in ('COMPLETE', 'FAILED'):
                print('Queue reached terminal state; monitor stopped without modifying training.', flush=True)
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()

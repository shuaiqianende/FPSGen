"""Read-only live progress and ETA for a detached VoxDet training run."""

import argparse
from collections import deque
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import time


def recent_rows(path, window=120):
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = max(0, handle.tell() - 1024 * 1024)
        handle.seek(position)
        if position:
            handle.readline()  # Discard a partially read first record.
        lines = handle.read().splitlines()
    rows = deque(maxlen=window)
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The writer may still be appending its final record.
    return list(rows)


def duration(seconds):
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def render(run):
    launch = json.loads((run / "launch.json").read_text())
    now = datetime.now().astimezone()
    try:
        os.kill(launch["pid"], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    rows = [recent_rows(run / f"training_rank{rank}.jsonl") for rank in (0, 1)]
    progress = [rank[-1]["successful_updates"] if rank else 0 for rank in rows]
    completed = min(progress)
    target = launch["target_successful_updates"]
    status = "COMPLETE" if completed >= target else ("RUNNING" if alive else "STOPPED")
    fraction = min(1, completed / target)
    filled = int(40 * fraction)
    lines = ["VoxDet Density | physical GPU2 + GPU3",
             f"{now:%Y-%m-%d %H:%M:%S %Z}  Status: {status}  PID: {launch['pid']}",
             f"[{'#'*filled}{'.'*(40-filled)}] {completed:,}/{target:,} updates ({fraction:.2%})",
             f"Scene exposures: {completed*launch['global_batch']:,}/{launch['target_seen_frames']:,}",
             f"Batch/GPU: {launch['batch_per_gpu']}  Global batch: {launch['global_batch']}  GT ROI: {launch['gt_roi_policy']}", ""]
    rates = []
    for rank, records in enumerate(rows):
        if not records:
            lines.append(f"Rank {rank}: initializing")
            continue
        last = records[-1]
        updates = last["successful_updates"] - records[0]["successful_updates"]
        if updates > 0:
            rates.append(sum(row["step_seconds"] for row in records[1:]) / updates)
        loss = last["details"][0]["total"]
        gradients = "finite" if last["gradients_finite"] else "AMP overflow/skipped"
        filtered = sum(scene["gt_validation"]["filtered_roi_points"] for row in records for scene in row["details"])
        peak = last["peak_vram_bytes"] / 1024**3
        lines.append(f"Rank {rank}: update {progress[rank]:,} | loss {loss:.5f} | {last['step_seconds']:.3f}s/step | gradients {gradients}")
        lines.append(f"        Peak allocated {peak:.2f} GiB | filtered GT points in recent window: {filtered}")
    if rates:
        seconds_per_update = max(rates)
        remaining = (target - completed) * seconds_per_update
        eta = now + timedelta(seconds=remaining)
        lines.extend(["", f"Recent speed: {seconds_per_update:.3f}s/update | {launch['global_batch']/seconds_per_update:.2f} scenes/s (both GPUs)",
                      f"Estimated training total: {duration(target*seconds_per_update)}"])
        if status == "RUNNING":
            lines.append(f"Remaining: {duration(remaining)} | ETA: {eta:%Y-%m-%d %H:%M:%S %Z}")
        else:
            lines.append("ETA: -- (training finished or process stopped)")
    checkpoints = sorted(run.glob("checkpoint_*.pt"))
    lines.extend(["", f"Latest checkpoint: {checkpoints[-1].name if checkpoints else '(not saved yet)'}",
                  "Checkpoint policy: first at100; every1000; 25/50/75/100% milestones",
                  f"Run: {run}", f"Log: {run/'train.log'}", ""])
    gpu = subprocess.run(["nvidia-smi", "--id=2,3", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv"],
                         capture_output=True, text=True, timeout=10)
    lines.append(gpu.stdout.strip() if gpu.returncode == 0 else "GPU status unavailable")
    lines.append("Read-only progress view; detach with Ctrl-b, then d. Training continues independently.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        display = render(args.output.resolve())
        print(("" if args.once else "\033[2J\033[H") + display, flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one formal stage inside its dedicated tmux; report its real exit code."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def run_stage(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    directory = Path(spec['directory'])
    result = directory / 'worker_result.json'
    if result.exists() or (directory / 'console.log').exists():
        raise FileExistsError('Refusing overwritten stage result')
    environment = os.environ.copy()
    environment.update(spec['environment'])
    code, error = 1, None
    try:
        with (directory / 'console.log').open('w') as log:
            process = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            last_flush = time.monotonic()
            for line in process.stdout:
                log.write(line)
                print(line, end='', flush=True)
                if time.monotonic() - last_flush >= 5:
                    log.flush()
                    last_flush = time.monotonic()
            code = process.wait()
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        error = f'{type(exc).__name__}: {exc}'
    finally:
        temporary = result.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'returncode': code, 'error': error,
            'end': datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + '\n')
        temporary.replace(result)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_stage(args.spec))

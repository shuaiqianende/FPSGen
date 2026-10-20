"""Low-overhead CUDA-event hooks and latency summaries for inference benchmarks."""
from __future__ import annotations

import numpy as np
import torch


class ModuleCudaTimer:
    """Record module-call CUDA intervals without synchronizing inside forward."""

    def __init__(self, module):
        self.module = module
        self.events = []
        self._pending = []
        self._handles = []

    def __enter__(self):
        def before(_module, _args):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._pending.append((start, end))

        def after(_module, _args, _output):
            if not self._pending:
                raise RuntimeError("Unpaired module CUDA timing hook")
            start, end = self._pending.pop()
            end.record()
            self.events.append((start, end))

        self._handles = [self.module.register_forward_pre_hook(before),
                         self.module.register_forward_hook(after)]
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()

    @property
    def calls(self):
        return len(self.events)

    def elapsed_ms(self):
        if self._pending:
            raise RuntimeError("Unclosed module timing event")
        return float(sum(start.elapsed_time(end) for start, end in self.events))


def timed_cuda_call(fn):
    """Return function result and CUDA-event duration; caller synchronizes later."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    return result, (start, end)


def event_elapsed_ms(events):
    return float(sum(a.elapsed_time(b) for a, b in events))


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Cannot summarize empty/non-finite timings")
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "min": float(x.min()),
            "max": float(x.max())}

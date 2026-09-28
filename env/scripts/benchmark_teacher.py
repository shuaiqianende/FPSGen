"""Compatibility entrypoint for the real DCD-Teacher benchmark.

The implementation lives in :mod:`probe_teacher_amp`, which deliberately keeps
coordinates, P0, endpoint construction and DCD in FP32.  This wrapper avoids a
second, divergent benchmark implementation while retaining the planned public
entrypoint.
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("probe_teacher_amp.py")), run_name="__main__")

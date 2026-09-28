#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="$ROOT/env/venv_pt20_cu117"
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-11.7
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=8
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TMPDIR="$ROOT/env/tmp"
mkdir -p "$ROOT/env/tmp"
mkdir -p "$ROOT/env/logs"
"$PREFIX/bin/python" -m pip install --no-build-isolation -e "$ROOT/fpsgen/models/ChamferDistancePytorch/chamfer3D" \
  2>&1 | tee "$ROOT/env/logs/chamfer3d_build.log"

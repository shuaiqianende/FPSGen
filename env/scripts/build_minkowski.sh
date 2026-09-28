#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="$ROOT/env/venv_pt20_cu117"
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-11.7
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=8
export KEOPS_CACHE_DIR="$ROOT/env/keops_cache"
# The virtualenv deliberately uses the legacy Python 3.9 interpreter as its
# base, whose sysconfig compiler wrapper hides the host OpenMP headers.  Build
# this new ABI target with the host toolchain instead; this does not alter the
# legacy FPSGen environment.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TMPDIR="$ROOT/env/tmp"
mkdir -p "$ROOT/env/logs" "$ROOT/env/tmp" "$KEOPS_CACHE_DIR"
# Apply the narrow PyTorch-2 header-order compatibility patch once.  The source
# tree is deliberately excluded from Git; the patch itself is versioned.
SOURCE="$ROOT/env/src/MinkowskiEngine-0.5.4"
if ! grep -q '^#include <ATen/ATen.h>$' "$SOURCE/src/spmm.cu"; then
  patch -d "$SOURCE" -p1 < "$ROOT/env/patches/minkowskiengine-0.5.4-pytorch2-aten-include.patch"
fi
# Build the pinned local source tree against the new local PyTorch ABI.
"$PREFIX/bin/python" -m pip install --no-build-isolation --no-cache-dir "$ROOT/env/src/MinkowskiEngine-0.5.4" \
  2>&1 | tee "$ROOT/env/logs/minkowski_build.log"

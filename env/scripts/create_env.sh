#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="$ROOT/env/venv_pt20_cu117"
BASE_PYTHON="/home/hndx/anaconda3/envs/M2024-HWZ-CasFusionNet/bin/python"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
export PIP_CACHE_DIR="$ROOT/env/pip_cache"
export TMPDIR="$ROOT/env/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
# The host has no standalone Python 3.9. The legacy interpreter is used only
# as an immutable venv base; no package is installed into the legacy env.
"$BASE_PYTHON" -m venv "$PREFIX"
# Lightning 1.8.1 has legacy metadata rejected by pip >=24.1.
"$PREFIX/bin/python" -m pip install "pip<24.1" "setuptools<70" wheel
"$PREFIX/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 torch==2.0.1 torchvision==0.15.2
"$PREFIX/bin/python" -m pip install -r "$ROOT/requirements.txt"
echo "Activate with: source $PREFIX/bin/activate"

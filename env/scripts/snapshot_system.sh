#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
{
  date -Is
  nvidia-smi
  nvcc --version
  which nvcc
  gcc --version | head -1
  g++ --version | head -1
  which conda
  conda --version
  conda run -n M2024-HWZ-CasFusionNet python -c \
    "import torch, MinkowskiEngine as ME; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), getattr(ME, '__version__', 'import OK'))"
} > "$ROOT/env/baseline_system.txt"

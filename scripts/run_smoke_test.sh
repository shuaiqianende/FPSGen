#!/usr/bin/env bash
set -euo pipefail

# Select a physical GPU at invocation time. CUDA remaps it to logical cuda:0
# inside Python, so the checks remain independent of the host GPU numbering.
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python scripts/check_environment.py
python scripts/verify_cuda_ops.py

echo "CUDA preflight passed on physical GPU ${GPU_ID} (logical cuda:0)."
echo "For a one-epoch data smoke test, use the checked-in smoke configs:"
echo "  TRAIN_DATABASE=/path/to/KITTI_Odometry python -m fpsgen.train_bev --config configs/smoke_bev.yaml"

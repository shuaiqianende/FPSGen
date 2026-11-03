#!/usr/bin/env bash
set -euo pipefail
arm=${1:?arm required}
case "$arm" in
  with_encoder) export CUDA_VISIBLE_DEVICES=2 ;;
  no_encoder) export CUDA_VISIBLE_DEVICES=3 ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
python=/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python
runner=scripts/research_v2/hier_occ_flow/train_stage2_b4_paired.py
logs="experiments/stage2_b4_paired_v1/$arm"
mkdir -p "$logs"
"$python" -u "$runner" --arm "$arm" --mode gate > "$logs/gate.log" 2>&1
"$python" -u "$runner" --arm "$arm" --mode smoke > "$logs/smoke.log" 2>&1
exec "$python" -u "$runner" --arm "$arm" --mode formal > "$logs/formal.log" 2>&1

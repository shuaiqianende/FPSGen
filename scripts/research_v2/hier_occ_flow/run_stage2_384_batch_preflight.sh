#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_gpu=${1:?Physical GPU2 or3 required}
case "$task_gpu" in 2|3) ;; *) exit 2;; esac
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
export CUDA_VISIBLE_DEVICES="$task_gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export TORCH_EXTENSIONS_DIR="$task_main/env/voxel_point_cache/torch_extensions" TMPDIR="$task_main/env/tmp"
task_output=experiments/hier_occ_flow/stage2_384_preflight
"$task_python" -u scripts/research_v2/hier_occ_flow/benchmark_stage2_384_batch.py \
  --config configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_overfit.yaml \
  --physical-gpu "$task_gpu" --output "$task_output/packed_batch2_gpu${task_gpu}.json" \
  2>&1 | tee "$task_output/packed_batch2_gpu${task_gpu}.log"

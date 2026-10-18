#!/usr/bin/env bash
# Fresh Stage2 only. Data/actual GPU2+3/GB4 restore gates must pass first.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
test -x "$task_python"
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4
export TMPDIR="$task_main/env/tmp" TORCH_EXTENSIONS_DIR="$task_main/env/voxel_point_cache/torch_extensions"
task_preflight=experiments/hier_occ_flow/stage2_real_score_gt_only_morph/preflight
mkdir -p "$task_preflight"
"$task_python" -u scripts/research_v2/hier_occ_flow/start_stage2_morph_training.py \
  --config configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml \
  --smoke-config configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_smoke.yaml \
  --output experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes \
  --preflight-output "$task_preflight" 2>&1 | tee "$task_preflight/controller.log"

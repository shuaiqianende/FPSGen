#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python="$task_main/env/pwm_ptv3_pointflow/bin/python"
test -x "$task_python"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$task_main/env/tmp"
task_output=experiments/hier_occ_flow/occ384_full
mkdir -p "$task_output/logs"
exec 9>"$task_output/preprocessing.lock"
flock -n 9 || { echo 'This preprocessing task is already running' >&2; exit 2; }
for task_split in train development; do
  nice -n 10 ionice -c 2 -n 7 "$task_python" -u scripts/research_v2/hier_occ_flow/prepare_dataset.py \
    --geometry configs/research_v2/hier_occ_flow/geometry_384_8way.yaml \
    --geometry-decision configs/research_v2/hier_occ_flow/geometry_384_decision.json \
    --manifest "$task_output/split_manifest.json" --output "$task_output" \
    --split "$task_split" --shard-frames 32 --full --resume \
    2>&1 | tee -a "$task_output/logs/$task_split.log"
done
nice -n 10 ionice -c 2 -n 7 "$task_python" -u scripts/research_v2/hier_occ_flow/audit_occ384_full.py \
  --root "$task_output" --reference-frames 3 2>&1 | tee -a "$task_output/logs/full_audit.log"

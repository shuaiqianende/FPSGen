#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python="$task_main/env/pwm_ptv3_pointflow/bin/python"
test -x "$task_python"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$task_main/env/tmp"
task_output=experiments/hier_occ_flow/occ384_full_parallel
mkdir -p "$task_output/logs"
exec 9>"$task_output/preprocessing.lock"
flock -n 9 || { echo 'Parallel preprocessing already running' >&2; exit 2; }
nice -n 10 ionice -c 2 -n 7 "$task_python" -u scripts/research_v2/hier_occ_flow/prepare_occ384_parallel.py \
  --manifest experiments/hier_occ_flow/occ384_full/split_manifest.json \
  --train-source experiments/hier_occ_flow/occ384_full/train/source_manifest.json \
  --output "$task_output" --workers 8 2>&1 | tee -a "$task_output/logs/prepare.log"
nice -n 10 ionice -c 2 -n 7 "$task_python" -u scripts/research_v2/hier_occ_flow/audit_occ384_full.py \
  --root "$task_output" --reference-frames 3 2>&1 | tee -a "$task_output/logs/full_audit.log"

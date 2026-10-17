#!/usr/bin/env bash
# Bounded learning diagnosis after the400-update quality failure. No pilot queue.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
task_cfg=configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_overfit1000_diagnostic.yaml
task_output=experiments/hier_occ_flow/stage2_384_fm_gb4_overfit1000_diagnostic
task_log=experiments/hier_occ_flow/stage2_384_preflight/fm_gb4_overfit1000_diagnostic.log
exec > >(tee -a "$task_log") 2>&1
task_phase=bounded_oracle_diagnostic
trap 'task_result=$?; printf "DIAGNOSTIC phase=%s exit=%s. Original quality thresholds remain unchanged. No formal training queued.\n" "$task_phase" "$task_result"' EXIT
echo 'Prior400-update run completed safely but FAILED its quality gate. Continuing only the12-frame Oracle diagnostic to update1000.'
echo 'FM only; physical GPU2/3; two full variable-length scenes/GPU; global batch4.'
echo 'The optimizer/model/EMA/scaler continue; the cosine horizon changes400->1000 explicitly.'
if [[ -f "$task_output/checkpoints/resume_latest.pt" ]]; then
  bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_cfg" \
    --output "$task_output" --resume "$task_output/checkpoints/resume_latest.pt"
else
  bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_cfg" \
    --output "$task_output" --transition-from experiments/hier_occ_flow/stage2_384_fm_gb4_overfit400/checkpoints/step000400.pt
fi
task_phase=unchanged_quality_gate
"$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py overfit \
  --run "$task_output" --objective fm \
  --output experiments/hier_occ_flow/stage2_384_preflight/fm_gb4_overfit1000_diagnostic_evidence.json
echo 'Bounded overfit diagnostic complete. Inspect saved evidence before any full-split pilot.'

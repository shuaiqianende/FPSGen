#!/usr/bin/env bash
# Controlled progression: real GPU evidence -> DDP restore -> overfit ->1K.
# Any failed gate stops this queue. Never starts100K or stops another task.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
task_audit=experiments/hier_occ_flow/stage2_384_preflight
task_prefix=experiments/hier_occ_flow/stage2_384_final_stage1
task_config=configs/research_v2/hier_occ_flow/stage2_384_oracle_fm.yaml
task_gate_args=(--config "$task_config" --gpu-reports "$task_audit/gpu2_final_stage1.json" "$task_audit/gpu3_final_stage1.json")
task_stage=engineering_gate
trap 'task_exit=$?; printf "Stage2 pipeline stopped: phase=%s exit=%s\n" "$task_stage" "$task_exit"' EXIT
"$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py gate \
  "${task_gate_args[@]}" --output "$task_audit/engineering_gate.json"
task_stage=ddp_resume
bash scripts/research_v2/hier_occ_flow/smoke_stage2_384_resume.sh "${task_prefix}_ddp32"
task_gate_args+=(--resume-report "${task_prefix}_ddp32/resume_comparison.json")
"$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py gate \
  "${task_gate_args[@]}" --output "$task_audit/engineering_gate.json"
task_stage=oracle_overfit200
CUDA_VISIBLE_DEVICES=2 bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh \
  --config configs/research_v2/hier_occ_flow/stage2_384_oracle_bce.yaml \
  --output "${task_prefix}_oracle_bce_overfit200" 2>&1 | tee "$task_audit/oracle_bce_overfit200.log" | sed -u 's/^/[GPU2 BCE overfit] /' &
task_bce_pid=$!
CUDA_VISIBLE_DEVICES=3 bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh \
  --config configs/research_v2/hier_occ_flow/stage2_384_oracle_fm.yaml \
  --output "${task_prefix}_oracle_fm_overfit200" 2>&1 | tee "$task_audit/oracle_fm_overfit200.log" | sed -u 's/^/[GPU3 FM overfit] /' &
task_fm_pid=$!
echo "Oracle overfit running: GPU2 BCE PID=$task_bce_pid; GPU3 FM PID=$task_fm_pid"
task_bce_status=0;task_fm_status=0
wait "$task_bce_pid" || task_bce_status=$?
wait "$task_fm_pid" || task_fm_status=$?
if (( task_bce_status || task_fm_status )); then exit 2; fi
for task_objective in bce fm; do
  "$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py overfit \
    --run "${task_prefix}_oracle_${task_objective}_overfit200" --objective "$task_objective" \
    --output "$task_audit/oracle_${task_objective}_overfit_evidence.json"
done
task_gate_args+=(--overfit-reports "$task_audit/oracle_bce_overfit_evidence.json" "$task_audit/oracle_fm_overfit_evidence.json")
"$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py gate \
  "${task_gate_args[@]}" --output "$task_audit/engineering_gate.json"
task_stage=oracle_pilot1000
CUDA_VISIBLE_DEVICES=2 bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh \
  --config configs/research_v2/hier_occ_flow/stage2_384_oracle_bce_pilot1000.yaml \
  --output "${task_prefix}_oracle_bce_pilot1000" 2>&1 | tee "$task_audit/oracle_bce_pilot1000.log" | sed -u 's/^/[GPU2 BCE1K] /' &
task_bce_pid=$!
CUDA_VISIBLE_DEVICES=3 bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh \
  --config configs/research_v2/hier_occ_flow/stage2_384_oracle_fm_pilot1000.yaml \
  --output "${task_prefix}_oracle_fm_pilot1000" 2>&1 | tee "$task_audit/oracle_fm_pilot1000.log" | sed -u 's/^/[GPU3 FM1K] /' &
task_fm_pid=$!
echo "Matched full-split1K running: GPU2 BCE PID=$task_bce_pid; GPU3 FM PID=$task_fm_pid"
task_bce_status=0;task_fm_status=0
wait "$task_bce_pid" || task_bce_status=$?
wait "$task_fm_pid" || task_fm_status=$?
if (( task_bce_status || task_fm_status )); then exit 2; fi
task_stage=pilot_comparison
"$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py pilot-compare \
  --bce-run "${task_prefix}_oracle_bce_pilot1000" --fm-run "${task_prefix}_oracle_fm_pilot1000" \
  --output "$task_audit/matched_pilot1000.json"
"$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py gate \
  "${task_gate_args[@]}" --pilot-report "$task_audit/matched_pilot1000.json" --output "$task_audit/engineering_gate.json"
task_stage=ready_to_review_5k
echo 'Matched1K pilot gate completed. No100K or automatic PointFlow queue was started.'

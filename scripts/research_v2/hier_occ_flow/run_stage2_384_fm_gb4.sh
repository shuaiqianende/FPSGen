#!/usr/bin/env bash
# FM only: actual packed2/GPU, global4. Preserve the user's existing FM lineage.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
task_preflight=experiments/hier_occ_flow/stage2_384_preflight
task_prefix=experiments/hier_occ_flow/stage2_384_fm_gb4
task_cfg=configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_overfit.yaml
task_gate=(--config "$task_cfg" --base-gate "$task_preflight/engineering_gate.json" \
  --gpu-reports "$task_preflight/packed_batch2_gpu2.json" "$task_preflight/packed_batch2_gpu3.json" \
  --output "$task_preflight/engineering_gate_gb4.json")
task_phase=gb4_gpu_gate
trap 'task_result=$?; printf "FM GB4 pipeline phase=%s exit=%s\n" "$task_phase" "$task_result"' EXIT
"$task_python" -u scripts/research_v2/hier_occ_flow/prepare_stage2_gb4_gate.py "${task_gate[@]}"
task_phase=gb4_ddp_resume
task_smoke_cfg=configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_smoke.yaml
if [[ ! -e "${task_prefix}_ddp32" ]]; then
  HS_STAGE2_SMOKE_CONFIG="$task_smoke_cfg" bash scripts/research_v2/hier_occ_flow/smoke_stage2_384_resume.sh "${task_prefix}_ddp32"
else
  # Explicit same-identity recovery if admission stopped a transition between
  # our own already-finished ranks. The runner still enforces config/source SHA.
  if [[ ! -f "${task_prefix}_ddp32/continuous21.pt" ]]; then
    echo 'Existing smoke has no completed continuous21 reference; inspect it before proceeding' >&2;exit 2
  fi
  bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_smoke_cfg" \
    --output "${task_prefix}_ddp32" --resume "${task_prefix}_ddp32/checkpoints/step000020.pt" --max-updates 21
  "$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py resume \
    --continuous "${task_prefix}_ddp32/continuous21.pt" --resumed "${task_prefix}_ddp32/checkpoints/resume_latest.pt" \
    --output "${task_prefix}_ddp32/resume_comparison.json"
  bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_smoke_cfg" \
    --output "${task_prefix}_ddp32" --resume "${task_prefix}_ddp32/checkpoints/resume_latest.pt" --max-updates 32
fi
task_gate+=(--resume-report "${task_prefix}_ddp32/resume_comparison.json")
"$task_python" -u scripts/research_v2/hier_occ_flow/prepare_stage2_gb4_gate.py "${task_gate[@]}"
task_phase=continue_own_fm_overfit_gb4
task_checkpoint=$("$task_python" - "$task_cfg" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['transition']['checkpoint'])
PY
)
bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_cfg" \
  --output "${task_prefix}_overfit400" --transition-from "$task_checkpoint"
task_phase=fm_learning_gate
"$task_python" -u scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py overfit \
  --run "${task_prefix}_overfit400" --objective fm --output "$task_preflight/fm_gb4_overfit_evidence.json"
"$task_python" -u scripts/research_v2/hier_occ_flow/prepare_stage2_gb4_gate.py "${task_gate[@]}" \
  --fm-overfit-report "$task_preflight/fm_gb4_overfit_evidence.json"
task_phase=freeze_full_split_fm_pilot
"$task_python" - "$task_cfg" "${task_prefix}_overfit400" <<'PY'
import sys,yaml,hashlib,json
from pathlib import Path
path=Path(sys.argv[1]);cfg=yaml.safe_load(path.read_text());previous=Path(sys.argv[2]).resolve()
summary=json.loads((previous/'summary.json').read_text());checkpoint=previous/'checkpoints/resume_latest.pt'
cfg.pop('cohort_indices');cfg['cache']['enabled']=False
cfg['training'].update(scope='pilot1000',successful_updates=summary['successful_updates']+1000,
 checkpoint_every=100,milestone_every=1000,warmup_updates=100)
cfg['transition']={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
 'reason':'continue own FM model/optimizer/EMA onto the complete training split for1000 additional GB4 updates'}
path.with_name('stage2_384_fm_gb4_pilot1000.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
task_phase=full_split_fm_pilot1000
bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh \
  --config configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_pilot1000.yaml \
  --output "${task_prefix}_pilot1000" --transition-from "${task_prefix}_overfit400/checkpoints/resume_latest.pt"
task_phase=complete_requested_fm_pilot
echo 'FM-only GB4 pilot complete. No BCE or100K formal queue started.'

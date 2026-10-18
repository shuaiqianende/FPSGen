#!/usr/bin/env bash
# Full-scene DDP 21 vs.20+restore1, using identical run identity/horizon.
# Run ONLY after BOTH single-GPU full-scene gates have passed and GPUs are idle.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
task_output=${1:?Usage: smoke_stage2_384_resume.sh NEW_OUTPUT_DIRECTORY}
if [[ -e "$task_output" ]]; then
  echo "Refusing existing smoke output: $task_output" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=2,3
task_config=${HS_STAGE2_SMOKE_CONFIG:-configs/research_v2/hier_occ_flow/stage2_384_ddp_smoke.yaml}
bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_config" --output "$task_output" --max-updates 21
cp "$task_output/checkpoints/step000021.pt" "$task_output/continuous21.pt"
# step20 was saved by the continuous run with all per-rank RNG/cursor state.
# Resume the same identity, not a renamed experiment or a different horizon.
bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_config" --output "$task_output" \
  --resume "$task_output/checkpoints/step000020.pt" --max-updates 21
cp "$task_output/checkpoints/resume_latest.pt" "$task_output/resumed21.pt"
"$task_python" scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py resume \
  --continuous "$task_output/continuous21.pt" --resumed "$task_output/resumed21.pt" \
  --output "$task_output/resume_comparison.json"
# Continue to32 only if the registered resume comparison passed.
bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_config" --output "$task_output" \
  --resume "$task_output/checkpoints/resume_latest.pt" --max-updates 32

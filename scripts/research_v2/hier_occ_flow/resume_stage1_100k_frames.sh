#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python="$task_main/env/voxdet_density_py39/bin/python"
task_config=configs/research_v2/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4.yaml
task_output=experiments/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4
task_gate=experiments/hier_occ_flow/stage1_occ384_frames_engineering_gate.json
test -x "$task_python"
if [[ -d "$task_output/checkpoints" ]]; then
  echo 'Frame-budget continuation already has checkpoints; use explicit own-phase resume' >&2
  exit 2
fi
"$task_python" scripts/research_v2/hier_occ_flow/freeze_stage1_gate.py --config "$task_config" --output "$task_gate"
task_resume=$("$task_python" - "$task_config" "$task_gate" <<'PY'
import sys,json,yaml
from pathlib import Path
c=yaml.safe_load(Path(sys.argv[1]).read_text());g=json.loads(Path(sys.argv[2]).read_text())
if not g['formal_ready'] or not g['frame_budget_accounting']:raise SystemExit('Frame budget gate not ready')
print(c['budget_transition']['checkpoint'])
PY
)
task_steps=$("$task_python" - "$task_config" <<'PY'
import sys,yaml
from pathlib import Path
print(yaml.safe_load(Path(sys.argv[1]).read_text())['training']['successful_updates'])
PY
)
mkdir -p "$task_output/configs"
cp "$task_config" "$task_output/configs/frozen.yaml"
cp configs/research_v2/hier_occ_flow/geometry_384_8way.yaml "$task_output/configs/geometry.yaml"
cp "$task_gate" "$task_output/configs/engineering_gate.json"
export CUDA_VISIBLE_DEVICES=2,3
exec bash scripts/research_v2/hier_occ_flow/launch_stage1_occ.sh --config "$task_config" --output "$task_output" --resume "$task_resume" --max-updates "$task_steps"

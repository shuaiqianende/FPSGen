#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python="$task_main/env/voxdet_density_py39/bin/python"
test -x "$task_python"
export CUDA_VISIBLE_DEVICES=2,3
task_output=experiments/hier_occ_flow/stage1_occ384_fm_100k
if [[ -d "$task_output/checkpoints" ]]; then
  echo 'Fresh formal launch refuses existing checkpoints; use explicit own-experiment resume' >&2
  exit 2
fi
"$task_python" scripts/research_v2/hier_occ_flow/freeze_stage1_gate.py
"$task_python" - <<'PY'
import json
from pathlib import Path
gate=json.loads(Path('experiments/hier_occ_flow/stage1_occ384_engineering_gate.json').read_text())
if not gate['formal_ready']:raise SystemExit('Formal engineering gate is not ready')
PY
exec bash scripts/research_v2/hier_occ_flow/launch_stage1_occ.sh \
  --config configs/research_v2/hier_occ_flow/stage1_occ384_fm.yaml \
  --output "$task_output" --max-updates 100000

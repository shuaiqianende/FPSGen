#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_OCCFLOW_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
test -x "$task_python"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
case "$CUDA_VISIBLE_DEVICES" in 2|3|2,3) ;; *) exit 2;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$task_main/env/tmp" TORCH_EXTENSIONS_DIR="$task_main/env/voxel_point_cache/torch_extensions"
mkdir -p experiments/hier_occ_flow/admission
task_gate=$(mktemp "$task_root/experiments/hier_occ_flow/admission/gate.XXXXXXXX.json")
"$task_python" - "$task_gate" <<'PY'
import os,sys,time
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json
ids=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
gate=gpu_preflight(ids);gate.update(checked_at_epoch=time.time(),physical_ids=ids)
atomic_json(sys.argv[1],gate);print(gate,flush=True)
if gate['status']!='AVAILABLE':raise SystemExit('GPU busy; no training launched')
PY
export HS_OCCFLOW_ADMISSION="$task_gate"
if [[ "$CUDA_VISIBLE_DEVICES" == 2,3 ]]; then
  exec "$task_python" -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/research_v2/hier_occ_flow/train_stage1_occ.py "$@"
fi
exec "$task_python" scripts/research_v2/hier_occ_flow/train_stage1_occ.py "$@"

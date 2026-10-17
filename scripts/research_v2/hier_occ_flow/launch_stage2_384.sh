#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
case "$CUDA_VISIBLE_DEVICES" in 2|3|2,3) ;; *) exit 2;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export TMPDIR="$task_main/env/tmp" TORCH_EXTENSIONS_DIR="$task_main/env/voxel_point_cache/torch_extensions"
mkdir -p experiments/hier_occ_flow/stage2_384_admission
task_gate=$(mktemp "$task_root/experiments/hier_occ_flow/stage2_384_admission/gate.XXXXXXXX.json")
"$task_python" - "$task_gate" <<'PY'
import sys,os,time
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json
ids=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
for attempt in range(16):
 gate=gpu_preflight(ids)
 if gate['status']=='AVAILABLE':break
 rows=gate.get('gpus',[])
 # NVML utilization can describe the previous interval after our own ranks
 # have already exited. Wait for an idle sample; never ignore a live PID.
 settling=(bool(rows) and all(not r['existing_compute_pids'] and r['free_mib']>=16*1024 for r in rows))
 if not settling or attempt==15:break
 print('Waiting for GPU utilization sample to settle after rank shutdown',flush=True)
 time.sleep(2)
gate.update(physical_ids=ids,checked_at_epoch=time.time())
atomic_json(sys.argv[1],gate);print(gate,flush=True)
if gate['status']!='AVAILABLE':raise SystemExit('Stage2 blocked; no GPU competition with Stage1')
PY
export HS_STAGE2_ADMISSION="$task_gate"
if [[ "$CUDA_VISIBLE_DEVICES" == 2,3 ]]; then
  exec "$task_python" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/research_v2/hier_occ_flow/train_stage2_384.py "$@"
fi
exec "$task_python" scripts/research_v2/hier_occ_flow/train_stage2_384.py "$@"

#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_git_common=$(git rev-parse --git-common-dir)
task_main_root=$(dirname "$task_git_common")
task_python=${HS_OCCFLOW_PYTHON:-$task_main_root/env/voxdet_density_py39/bin/python}
test -x "$task_python"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
case "$CUDA_VISIBLE_DEVICES" in 2|3|2,3) ;; *) echo 'Only physical GPU2/3 allowed' >&2; exit 2;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$task_main_root/env/tmp"
export TORCH_EXTENSIONS_DIR="$task_main_root/env/voxel_point_cache/torch_extensions"
export XDG_CACHE_HOME="$task_main_root/env/voxel_point_cache/xdg"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
entry=${1:?Specify train_bce, train_fm, benchmark_memory, evaluate, or export}
shift
case "$entry" in train_bce|train_fm|benchmark_memory|evaluate|export) ;; *) exit 2;; esac
nvidia-smi --id=2,3
if [[ "$CUDA_VISIBLE_DEVICES" == 2,3 && "$entry" == train_* ]]; then
  exec "$task_python" -m torch.distributed.run --standalone --nproc_per_node=2 "scripts/research_v2/hier_occ_flow/$entry.py" "$@"
fi
exec "$task_python" "scripts/research_v2/hier_occ_flow/$entry.py" "$@"

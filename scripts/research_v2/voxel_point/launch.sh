#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
entry=${1:?Usage: launch.sh ENTRY_SCRIPT [script arguments]}
shift
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
case "$CUDA_VISIBLE_DEVICES" in 2|3|2,3) ;; *) echo 'Only physical GPU2/3 are allowed' >&2; exit 2;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PIP_CACHE_DIR="$repo_root/env/pip_cache"
export TORCH_EXTENSIONS_DIR="$repo_root/env/voxel_point_cache/torch_extensions"
export KEOPS_CACHE_FOLDER="$repo_root/env/voxel_point_cache/keops"
export XDG_CACHE_HOME="$repo_root/env/voxel_point_cache/xdg"
export TMPDIR="$repo_root/env/voxel_point_tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$KEOPS_CACHE_FOLDER"
case "$entry" in
    train_voxdet_density|eval_voxdet_density|sample_pipeline|audit_voxdet_architecture)
        python_bin="$repo_root/env/voxdet_density_py39/bin/python" ;;
    train_mapping_teacher|train_pointflow|diagnose_sinkhorn|diagnose_pointflow_sources|evaluate_final_points|check_ptv3_checkpoint)
        python_bin="$repo_root/env/pwm_ptv3_pointflow/bin/python" ;;
    *) echo "Unsupported GPU entry: $entry" >&2; exit 2 ;;
esac
nvidia-smi
if [[ "$CUDA_VISIBLE_DEVICES" == 2,3 && "$entry" == train_* ]]; then
    exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
        "scripts/research_v2/voxel_point/$entry.py" "$@"
fi
exec "$python_bin" "scripts/research_v2/voxel_point/$entry.py" "$@"

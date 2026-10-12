#!/usr/bin/env bash
# Reproducible independent compatibility track. Does not install into main FPSGen.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
case "$CUDA_VISIBLE_DEVICES" in 2|3|2,3) ;; *) echo 'Only physical GPU2/3 are allowed' >&2; exit 2;; esac
nvidia-smi
base_python=${VOXEL_POINT_BASE_PYTHON:?Set VOXEL_POINT_BASE_PYTHON to a Python 3.9 executable}
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3,9)'
export PIP_CACHE_DIR="$repo_root/env/pip_cache"
export TMPDIR="$repo_root/env/voxel_point_tmp"
export XDG_CACHE_HOME="$repo_root/env/voxel_point_cache/xdg"
export TORCH_EXTENSIONS_DIR="$repo_root/env/voxel_point_cache/torch_extensions"
export MAX_JOBS=2
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"
for environment in voxdet_density_py39 pwm_ptv3_pointflow; do
    if [[ -e "env/$environment/bin/python" ]]; then
        echo "Refusing to overwrite existing environment env/$environment" >&2
        exit 2
    fi
done
for environment in voxdet_density_py39 pwm_ptv3_pointflow; do
    "$base_python" -m venv "env/$environment"
    interpreter="env/$environment/bin/python"
    "$interpreter" -m pip install 'pip==23.3.2' 'setuptools==68.2.2' wheel
    "$interpreter" -m pip install 'torch==2.0.1+cu117' 'torchvision==0.15.2+cu117' --index-url https://download.pytorch.org/whl/cu117
    "$interpreter" -m pip install 'numpy==1.23.5' 'scipy==1.10.1' 'PyYAML==6.0.1' 'plyfile==1.0.3' 'pytest==8.3.5' 'addict==2.4.0' 'einops==0.8.0' 'timm==0.9.16' 'spconv-cu117==2.3.6' 'pykeops==2.3'
    "$interpreter" -m pip install 'torch-scatter==2.1.2' -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
    "$interpreter" -m pip install --no-build-isolation --no-deps 'flash-attn==2.5.9.post1'
done
env/voxdet_density_py39/bin/python -m pip install 'opencv-python==4.8.1.78' 'matplotlib==3.7.5' 'mmengine==0.10.7'
env/voxdet_density_py39/bin/python -m pip install --no-build-isolation 'mmcv==1.7.2'

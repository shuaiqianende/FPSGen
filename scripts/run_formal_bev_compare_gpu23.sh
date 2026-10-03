#!/usr/bin/env bash
set -euo pipefail
cd /data-12/M2024-HWZ/FPSGen
source env/venv_pt20_cu117/bin/activate
export TRAIN_DATABASE=/data-12/M2024-HWZ/KITTI_Odometry
export PYTHONPATH=/data-12/M2024-HWZ/FPSGen
exec python -u scripts/run_formal_bev_compare.py "$@"

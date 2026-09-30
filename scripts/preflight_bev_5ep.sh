#!/usr/bin/env bash
# Run a strict 16-step eager-FP32 preflight on one designated physical GPU.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "hdit" && "$1" != "dip" && "$1" != "ncsnpp" ) ]]; then
  echo "usage: $0 {hdit|dip|ncsnpp}" >&2
  exit 2
fi

ROOT="/data-12/M2024-HWZ/FPSGen"
VENV="${ROOT}/env/venv_pt20_cu117"
source "${VENV}/bin/activate"
export TRAIN_DATABASE="${TRAIN_DATABASE:-/data-12/M2024-HWZ/KITTI_Odometry}"
export TMPDIR="${ROOT}/env/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${ROOT}/env/xdg_cache"
export MPLCONFIGDIR="${ROOT}/env/mpl_cache"
export KEOPS_CACHE_FOLDER="${ROOT}/env/keops_cache_legacy"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${KEOPS_CACHE_FOLDER}" \
  "${ROOT}/outputs/research_v2/bev_train/logs"

if [[ "$1" == "hdit" ]]; then
  export CUDA_VISIBLE_DEVICES=2
  CONFIG="${ROOT}/configs/research_v2/preflight_bev_hdit_s_gt_possion_5ep_b8_gpu2.yaml"
  LOG="${ROOT}/outputs/research_v2/bev_train/logs/preflight_fpsgen_bev_hdit_s_5ep_b8_gpu2.log"
elif [[ "$1" == "dip" ]]; then
  export CUDA_VISIBLE_DEVICES=3
  CONFIG="${ROOT}/configs/research_v2/preflight_bev_dip_s_gt_possion_5ep_b8_gpu3.yaml"
  LOG="${ROOT}/outputs/research_v2/bev_train/logs/preflight_fpsgen_bev_dip_s_5ep_b8_gpu3.log"
else
  export CUDA_VISIBLE_DEVICES=1
  CONFIG="${ROOT}/configs/research_v2/preflight_bev_ncsnpp_s_gt_possion_5ep_b8_gpu1.yaml"
  LOG="${ROOT}/outputs/research_v2/bev_train/logs/preflight_fpsgen_bev_ncsnpp_s_5ep_b8_gpu1.log"
fi

cd "${ROOT}"
python "${ROOT}/fpsgen/train_bev.py" --config "${CONFIG}" 2>&1 | tee "${LOG}"

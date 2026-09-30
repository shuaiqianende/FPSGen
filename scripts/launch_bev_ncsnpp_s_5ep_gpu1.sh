#!/usr/bin/env bash
set -euo pipefail

ROOT="/data-12/M2024-HWZ/FPSGen"
VENV="${ROOT}/env/venv_pt20_cu117"
RUN_ID="fpsgen_bev_ncsnpp_s_5ep_b8_gpu1"
source "${VENV}/bin/activate"
export TRAIN_DATABASE="${TRAIN_DATABASE:-/data-12/M2024-HWZ/KITTI_Odometry}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TMPDIR="${ROOT}/env/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${ROOT}/env/xdg_cache"
export MPLCONFIGDIR="${ROOT}/env/mpl_cache"
export KEOPS_CACHE_FOLDER="${ROOT}/env/keops_cache_legacy"
export FPSGEN_OUTPUT_DIR="${ROOT}/outputs/research_v2/bev_train/visuals/${RUN_ID}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" \
  "${KEOPS_CACHE_FOLDER}" "${ROOT}/outputs/research_v2/bev_train/logs" \
  "${ROOT}/outputs/research_v2/bev_train/profiles" "${FPSGEN_OUTPUT_DIR}"
cd "${ROOT}"
exec python "${ROOT}/fpsgen/train_bev.py" \
  --config "${ROOT}/configs/research_v2/train_bev_ncsnpp_s_gt_possion_5ep_b8_gpu1.yaml" \
  2>&1 | tee "${ROOT}/outputs/research_v2/bev_train/logs/${RUN_ID}.log"

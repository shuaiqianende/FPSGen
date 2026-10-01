#!/usr/bin/env bash
# Serial, failure-tolerant GPU2/GPU3 queues for the Stage-1 spatial study.
set -uo pipefail

ROOT="/data-12/M2024-HWZ/FPSGen"
VENV="${ROOT}/env/venv_pt20_cu117"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 {2|3}" >&2; exit 2
fi
GPU="$1"
DATA_ROOT="${TRAIN_DATABASE:-/data-12/M2024-HWZ/KITTI_Odometry}"
source "${VENV}/bin/activate"
export TRAIN_DATABASE="${DATA_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TMPDIR="${ROOT}/env/tmp"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${ROOT}/env/xdg_cache"
export MPLCONFIGDIR="${ROOT}/env/mpl_cache"
export KEOPS_CACHE_FOLDER="${ROOT}/env/keops_cache_spatial_control"
export FPSGEN_OUTPUT_DIR="${ROOT}/outputs/research_v2/spatial_control/visuals"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${KEOPS_CACHE_FOLDER}" \
  "${ROOT}/outputs/research_v2/spatial_control/logs" "${ROOT}/outputs/research_v2/spatial_control"
cd "${ROOT}"

if [[ "${GPU}" == 2 ]]; then
  CONFIGS=(unet_generic_b8.yaml synflow_bev_s_b8.yaml cracksegflow_bev_s_b8.yaml)
elif [[ "${GPU}" == 3 ]]; then
  CONFIGS=(pixeldit_generic_b8.yaml pixelcontrol_bev_s_b8.yaml)
else
  echo "Only physical GPU2 and GPU3 are permitted." >&2; exit 2
fi

run_logged() {
  local name="$1"; shift
  echo "[$(date -Is)] START ${name}" | tee -a "${ROOT}/outputs/research_v2/spatial_control/logs/${name}.log"
  "$@" >> "${ROOT}/outputs/research_v2/spatial_control/logs/${name}.log" 2>&1
  local code=$?
  echo "[$(date -Is)] END ${name} status=${code}" | tee -a "${ROOT}/outputs/research_v2/spatial_control/logs/${name}.log"
  return ${code}
}

for config_name in "${CONFIGS[@]}"; do
  config="${ROOT}/configs/research_v2/spatial_control/${config_name}"
  run_id="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"]["id"])' "${config}")"
  # A failed preflight or run is recorded and the queue continues: no idle GPU
  # caused by one candidate failure.
  run_logged "${run_id}_preflight" python "${ROOT}/fpsgen/train_bev.py" --config "${config}" --preflight-steps 16 || continue
  run_logged "${run_id}_train" python "${ROOT}/fpsgen/train_bev.py" --config "${config}" || continue
  checkpoint="$(find "${ROOT}/experiments/${run_id}" -name '*.ckpt' -type f | sort | tail -n 1)"
  if [[ -z "${checkpoint}" ]]; then
    echo "[$(date -Is)] FAIL ${run_id}: checkpoint missing" | tee -a "${ROOT}/outputs/research_v2/spatial_control/logs/${run_id}_train.log"
    continue
  fi
  run_logged "${run_id}_condition_usage" python scripts/eval_bev_spatial_control.py --checkpoint "${checkpoint}" --data-root "${DATA_ROOT}" --output "outputs/research_v2/spatial_control/${run_id}_condition_usage_b100.json" --frames 100 --batch-size 8 || true
  run_logged "${run_id}_b20_cfg2" python scripts/eval_bevflow_lidar_only.py --bev-ckpt "${checkpoint}" --dataset-root "${DATA_ROOT}" --manifest configs/research_v2/gate_seq08_20.txt --samples-per-frame 1 --save-visuals 0 --base-seed 20261001 --output "outputs/research_v2/spatial_control/${run_id}_lidar_b20_cfg2" || true
  run_logged "${run_id}_b20_cfg1" python scripts/eval_bevflow_lidar_only.py --config configs/research_v2/eval_bevflow_lidar_only_cfg1.yaml --bev-ckpt "${checkpoint}" --dataset-root "${DATA_ROOT}" --manifest configs/research_v2/gate_seq08_20.txt --samples-per-frame 1 --save-visuals 0 --base-seed 20261001 --output "outputs/research_v2/spatial_control/${run_id}_lidar_b20_cfg1" || true
done

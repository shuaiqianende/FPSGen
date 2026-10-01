#!/usr/bin/env bash
# Paired N0/N1 queues.  Both seed-42 evaluations finish before seed-123 begins.
set -uo pipefail
if [[ $# -ne 1 || ( "$1" != 2 && "$1" != 3 ) ]]; then echo "usage: $0 {2|3}" >&2; exit 2; fi
ROOT=/data-12/M2024-HWZ/FPSGen; GPU="$1"; DATA_ROOT="${TRAIN_DATABASE:-/data-12/M2024-HWZ/KITTI_Odometry}"
source "${ROOT}/env/venv_pt20_cu117/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU}" TRAIN_DATABASE="${DATA_ROOT}" TMPDIR="${ROOT}/env/tmp"
export XDG_CACHE_HOME="${ROOT}/env/xdg_cache" MPLCONFIGDIR="${ROOT}/env/mpl_cache" KEOPS_CACHE_FOLDER="${ROOT}/env/keops_cache_spatial_control"
ROOT_OUT="${ROOT}/outputs/research_v2/ncsnpp_spade"; mkdir -p "${ROOT_OUT}/logs" "${ROOT_OUT}/barrier"
cd "${ROOT}"
if [[ "${GPU}" == 2 ]]; then PREFIX=n0_generic; OWN=n0; PEER=n1; else PREFIX=n1_spade; OWN=n1; PEER=n0; fi
run() { local name="$1"; shift; echo "[$(date -Is)] START ${name}" | tee -a "${ROOT_OUT}/logs/${name}.log"; "$@" >> "${ROOT_OUT}/logs/${name}.log" 2>&1; local code=$?; echo "[$(date -Is)] END ${name} status=${code}" | tee -a "${ROOT_OUT}/logs/${name}.log"; return $code; }
for seed in 42 123; do
  config="configs/research_v2/ncsnpp_spade/${PREFIX}_seed${seed}.yaml"
  run_id="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"]["id"])' "${config}")"
  run "${run_id}_preflight" python fpsgen/train_bev.py --config "${config}" --preflight-steps 16 || { touch "${ROOT_OUT}/barrier/${OWN}_seed${seed}.ready"; continue; }
  run "${run_id}_train" python fpsgen/train_bev.py --config "${config}" || { touch "${ROOT_OUT}/barrier/${OWN}_seed${seed}.ready"; continue; }
  ckpt="$(find "experiments/${run_id}" -name '*.ckpt' -type f | sort | tail -1)"
  if [[ -n "${ckpt}" ]]; then
    run "${run_id}_condition" python scripts/eval_bev_spatial_control.py --checkpoint "${ckpt}" --data-root "${DATA_ROOT}" --output "${ROOT_OUT}/${run_id}_condition_usage_b100.json" --frames 100 --batch-size 8 || true
    run "${run_id}_cfg2" python scripts/eval_bevflow_lidar_only.py --bev-ckpt "${ckpt}" --dataset-root "${DATA_ROOT}" --manifest configs/research_v2/gate_seq08_20.txt --samples-per-frame 1 --save-visuals 0 --base-seed 20261001 --output "${ROOT_OUT}/${run_id}_lidar_b20_cfg2" || true
    run "${run_id}_cfg1" python scripts/eval_bevflow_lidar_only.py --config configs/research_v2/eval_bevflow_lidar_only_cfg1.yaml --bev-ckpt "${ckpt}" --dataset-root "${DATA_ROOT}" --manifest configs/research_v2/gate_seq08_20.txt --samples-per-frame 1 --save-visuals 0 --base-seed 20261001 --output "${ROOT_OUT}/${run_id}_lidar_b20_cfg1" || true
  fi
  touch "${ROOT_OUT}/barrier/${OWN}_seed${seed}.ready"
  if [[ "${seed}" == 42 ]]; then
    while [[ ! -f "${ROOT_OUT}/barrier/${PEER}_seed42.ready" ]]; do sleep 15; done
  fi
done

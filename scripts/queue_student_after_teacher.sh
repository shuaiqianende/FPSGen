#!/usr/bin/env bash
# Queue the first DCD+Sinkhorn Student short run behind a named Teacher job.
# It never starts without a completed Teacher checkpoint.
set -euo pipefail

ROOT="${ROOT:-/data-12/M2024-HWZ/FPSGen}"
ENV_NAME="${ENV_NAME:-M2024-HWZ-CasFusionNet}"
TEACHER_MATCH="${TEACHER_MATCH:-finetune_teacher_dcd_gt_possion_1ep_ddp2_bs2.yaml}"
TEACHER_DIR="${TEACHER_DIR:-${ROOT}/experiments/teacher_dcd_gt_possion_ft1ep_ddp2_bs2_lr1e4}"
STUDENT_CONFIG="${STUDENT_CONFIG:-${ROOT}/configs/research_v2/train_student_dcd_sinkhorn_gt_possion_500_ddp2_bs2.yaml}"
DATA_ROOT="${DATA_ROOT:-/data-12/M2024-HWZ/KITTI_Odometry}"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/research_v2/student_train/logs}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

echo "[$(date -Is)] waiting for Teacher command matching: ${TEACHER_MATCH}"
while pgrep -f "${TEACHER_MATCH}" >/dev/null; do
    sleep 60
done

mapfile -t CHECKPOINTS < <(find "${TEACHER_DIR}" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | awk '{print $2}')
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "[$(date -Is)] ERROR: Teacher process exited without a checkpoint under ${TEACHER_DIR}" >&2
    exit 1
fi
TEACHER_CKPT="${CHECKPOINTS[-1]}"
echo "[$(date -Is)] Teacher checkpoint selected: ${TEACHER_CKPT}"

# Activate only after the dependency succeeds; GPU2/3 remain untouched while
# this queue is waiting.
source /home/hndx/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=2,3
export TRAIN_DATABASE="${DATA_ROOT}"
exec python fpsgen/train_student.py --config "${STUDENT_CONFIG}" \
    --teacher-checkpoint "${TEACHER_CKPT}" 2>&1 | tee "${LOG_DIR}/student_dcd_sinkhorn_gt_possion_500_ddp2_bs2.log"

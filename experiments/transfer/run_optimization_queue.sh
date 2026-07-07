#!/usr/bin/env bash
set -uo pipefail

# Train all 48 optimization-based steering vectors, one job per GPU.
#
# 6 jobs (3 models x 2 dst_modes) across GPUs 0..5, GPU 6 held in reserve.
# Each job is single-GPU (no sharding), batch=1, max_iters=20, with
# --target_loss=0.001 as the early-stop threshold.
#
# Usage:
#   bash experiments/transfer/run_optimization_queue.sh
#
# Outputs:
#   experiments/transfer/vectors/optimization/<dst_mode>/<model>/<scenario>.pt
#   experiments/transfer/optimization_logs/<timestamp>/gpu<i>_<model>_<dst_mode>.log

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="experiments/transfer/optimization_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

LR="${LR:-0.05}"
MAX_ITERS="${MAX_ITERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
STARTING_NORM="${STARTING_NORM:-1.0}"
MAX_NORM="${MAX_NORM:-5.0}"
TARGET_LOSS="${TARGET_LOSS:-0.001}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-30}"
MULTI_GPU_MODE="${MULTI_GPU_MODE:-single}"

# Round-robin model order to avoid simultaneous cold weight loads from the
# shared HF cache.
JOBS=(
  "gemma-3-4b-it specific"
  "gemma-3-12b-it specific"
  "qwen3.5-9b specific"
  "gemma-3-4b-it generic"
  "gemma-3-12b-it generic"
  "qwen3.5-9b generic"
)

# 6 GPUs for 6 jobs, GPU 6 held in reserve.
GPUS=(0 1 2)

declare -A PID_TO_GPU
declare -A PID_TO_JOB
FAILED_JOBS=()

cleanup() {
  echo "Caught signal; killing remaining child jobs..." >&2
  for pid in "${!PID_TO_GPU[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  exit 130
}
trap cleanup INT TERM

launch_job() {
  local gpu="$1"
  local job="$2"
  local model_slug dst_mode
  model_slug="$(echo "${job}" | awk '{print $1}')"
  dst_mode="$(echo "${job}" | awk '{print $2}')"

  local log="${LOG_DIR}/gpu${gpu}_${model_slug}_${dst_mode}.log"
  echo "Launching ${job} on GPU ${gpu}; log=${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python -u \
    experiments/transfer/train_optimization_vectors.py \
    --model_slug "${model_slug}" \
    --dst_mode "${dst_mode}" \
    --lr "${LR}" \
    --max_iters "${MAX_ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --starting_norm "${STARTING_NORM}" \
    --max_norm "${MAX_NORM}" \
    --target_loss "${TARGET_LOSS}" \
    --model_dtype "${MODEL_DTYPE}" \
    --multi_gpu_mode "${MULTI_GPU_MODE}" \
    > "${log}" 2>&1 &

  local pid="$!"
  PID_TO_GPU["${pid}"]="${gpu}"
  PID_TO_JOB["${pid}"]="${job}"
  echo "${pid}" > "${LOG_DIR}/gpu${gpu}_${model_slug}_${dst_mode}.pid"
}

echo "Logs: ${LOG_DIR}"
echo "MAX_ITERS=${MAX_ITERS} BATCH_SIZE=${BATCH_SIZE} MAX_NORM=${MAX_NORM} "\
"TARGET_LOSS=${TARGET_LOSS} MODEL_DTYPE=${MODEL_DTYPE} MULTI_GPU_MODE=${MULTI_GPU_MODE}"

next_job=0
active=0
for gpu in "${GPUS[@]}"; do
  if [[ "${next_job}" -lt "${#JOBS[@]}" ]]; then
    launch_job "${gpu}" "${JOBS[${next_job}]}"
    next_job=$((next_job + 1))
    active=$((active + 1))
    sleep "${STARTUP_STAGGER_SECONDS}"
  fi
done

while [[ "${active}" -gt 0 ]]; do
  done_pid=""
  if wait -n -p done_pid; then
    status=0
  else
    status="$?"
  fi
  gpu="${PID_TO_GPU[${done_pid}]:-unknown}"
  job="${PID_TO_JOB[${done_pid}]:-unknown}"
  echo "Finished pid=${done_pid} gpu=${gpu} job=${job} status=${status}"
  unset "PID_TO_GPU[${done_pid}]"
  unset "PID_TO_JOB[${done_pid}]"
  active=$((active - 1))
  if [[ "${status}" != "0" ]]; then
    echo "Job FAILED: ${job} on GPU ${gpu} (status=${status}). See ${LOG_DIR}" >&2
    FAILED_JOBS+=("${job} (gpu=${gpu}, status=${status})")
  fi
  if [[ "${next_job}" -lt "${#JOBS[@]}" ]]; then
    launch_job "${gpu}" "${JOBS[${next_job}]}"
    next_job=$((next_job + 1))
    active=$((active + 1))
  fi
done

echo "All optimization training jobs finished."
echo "Logs are in: ${LOG_DIR}"

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "Failed jobs (${#FAILED_JOBS[@]}):" >&2
  for job in "${FAILED_JOBS[@]}"; do
    echo "  - ${job}" >&2
  done
  exit 1
fi

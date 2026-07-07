#!/usr/bin/env bash
set -uo pipefail

# Build all 48 synthetic steering vectors across 3 GPUs (one per model).
# Each worker loads its model once, iterates (split_type, scenario), prefills
# every (prompt + completion), captures mean activation over completion tokens
# at every layer, picks the meek-probe layer, and writes the DoM vector.
#
# Usage:
#   bash experiments/transfer/run_synthetic_queue.sh
#
# Outputs:
#   experiments/transfer/activations_synthetic/<split>/<model>/<scenario>.pt
#   experiments/transfer/vectors/synthetic/<split>/<model>/<scenario>.pt
#   experiments/transfer/synthetic_logs/<timestamp>/gpu<i>_<model>.log

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE_DIR="${HF_XET_CACHE_DIR:-${HF_HOME}/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="experiments/transfer/synthetic_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

SPLIT_TYPES="${SPLIT_TYPES:-meek giovanni}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-bfloat16}"

# Per-model batch sizes for prefill on a single 80GB GPU.
declare -A BATCH_FOR_MODEL=(
  ["gemma-3-4b-it"]=16
  ["qwen3.5-9b"]=8
  ["gemma-3-12b-it"]=4
)

JOBS=(
  "0 gemma-3-4b-it"
  "1 qwen3.5-9b"
  "2 gemma-3-12b-it"
)

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
  local job="$1"
  local gpu model_slug
  read -r gpu model_slug <<< "${job}"
  local batch_size="${BATCH_FOR_MODEL[${model_slug}]}"
  local log="${LOG_DIR}/gpu${gpu}_${model_slug}.log"
  echo "Launching ${model_slug} on GPU ${gpu} batch=${batch_size}; log=${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python -u \
    experiments/transfer/build_synthetic_vectors.py \
    --model_slug "${model_slug}" \
    --split_types ${SPLIT_TYPES} \
    --batch_size "${batch_size}" \
    --model_dtype "${MODEL_DTYPE}" \
    --save_dtype "${SAVE_DTYPE}" \
    > "${log}" 2>&1 &

  local pid="$!"
  PID_TO_GPU["${pid}"]="${gpu}"
  PID_TO_JOB["${pid}"]="${model_slug}"
  echo "${pid}" > "${LOG_DIR}/gpu${gpu}_${model_slug}.pid"
}

echo "Logs: ${LOG_DIR}"
echo "SPLIT_TYPES='${SPLIT_TYPES}' MAX_SEQ_LEN=${MAX_SEQ_LEN} MODEL_DTYPE=${MODEL_DTYPE}"

for job in "${JOBS[@]}"; do
  launch_job "${job}"
  sleep 30
done

active="${#JOBS[@]}"
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
done

echo "All synthetic vector jobs finished."
echo "Logs are in: ${LOG_DIR}"
if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "Failed jobs (${#FAILED_JOBS[@]}):" >&2
  for job in "${FAILED_JOBS[@]}"; do
    echo "  - ${job}" >&2
  done
  exit 1
fi

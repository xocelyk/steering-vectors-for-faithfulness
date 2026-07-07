#!/usr/bin/env bash
set -uo pipefail

# Run transfer activation extraction while avoiding simultaneous cold loads of
# the same model from the shared Hugging Face cache.
#
# Usage:
#   bash experiments/transfer/run_activation_extraction_queue.sh
#
# Behavior on failure:
#   A single failed job is logged and its GPU is freed for the next queued job.
#   The script continues running healthy jobs and only exits non-zero at the end
#   if any job ultimately failed. SIGINT/SIGTERM cleanly kills all children.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

LOG_DIR="experiments/transfer/activation_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-90}"

# Order matters: round-robin by model so we do not start three identical model
# loads at the same time (avoids FUSE I/O contention on the HF cache). The
# scheduler still fills freed GPUs as jobs finish.
JOBS=(
  "gemma-3-4b-it bbh"
  "gemma-3-12b-it bbh"
  "qwen3.5-9b bbh"
  "gemma-3-4b-it gpqa"
  "gemma-3-12b-it gpqa"
  "qwen3.5-9b gpqa"
  "gemma-3-4b-it mmlu"
  "gemma-3-12b-it mmlu"
  "qwen3.5-9b mmlu"
)

GPUS=(0 1 2 3 4 5 6)

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
  local model_slug dataset
  model_slug="$(echo "${job}" | awk '{print $1}')"
  dataset="$(echo "${job}" | awk '{print $2}')"

  local log="${LOG_DIR}/gpu${gpu}_${model_slug}_${dataset}.log"

  echo "Launching ${job} on GPU ${gpu}; log=${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python experiments/transfer/collect_transfer_activations.py \
    --model_slug "${model_slug}" \
    --datasets "${dataset}" \
    --cues stanford xml grader insider \
    --output_dir experiments/transfer/activations \
    > "${log}" 2>&1 &

  local pid="$!"
  PID_TO_GPU["${pid}"]="${gpu}"
  PID_TO_JOB["${pid}"]="${job}"
  echo "${pid}" > "${LOG_DIR}/gpu${gpu}_${model_slug}_${dataset}.pid"
}

echo "Logs: ${LOG_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "STARTUP_STAGGER_SECONDS=${STARTUP_STAGGER_SECONDS}"

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
    echo "Job FAILED: ${job} on GPU ${gpu} (status=${status}). Continuing other jobs; see ${LOG_DIR}" >&2
    FAILED_JOBS+=("${job} (gpu=${gpu}, status=${status})")
  fi

  if [[ "${next_job}" -lt "${#JOBS[@]}" ]]; then
    launch_job "${gpu}" "${JOBS[${next_job}]}"
    next_job=$((next_job + 1))
    active=$((active + 1))
  fi
done

echo "All activation extraction jobs finished."
echo "Logs are in: ${LOG_DIR}"

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "Failed jobs (${#FAILED_JOBS[@]}):" >&2
  for job in "${FAILED_JOBS[@]}"; do
    echo "  - ${job}" >&2
  done
  exit 1
fi

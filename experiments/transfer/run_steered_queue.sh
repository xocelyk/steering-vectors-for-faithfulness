#!/usr/bin/env bash
set -uo pipefail

# Generate steered traces (vLLM + nnsight injection) across 7 GPUs.
#
# Cells are partitioned by scenario across workers per model:
#   - gemma-3-4b-it:   1 GPU,  all 8 scenarios
#   - qwen3.5-9b:      2 GPUs, half scenarios each
#   - gemma-3-12b-it:  4 GPUs, ~2 scenarios each
# Each worker loads its model once and iterates only over its assigned
# scenarios; cell paths are deterministic so workers do not collide.
#
# Sampling params come from MODEL_PRESENCE_PENALTY / MODEL_MAX_NEW_TOKENS in
# generate_steered_traces.py — they match the baseline-trace configs:
#   gemma-*:    temp=0  top_p=1  presence_penalty=0.0  max_new_tokens=5000
#   qwen3.5-9b: temp=0  top_p=1  presence_penalty=1.5  max_new_tokens=10000
#
# Outputs:
#   experiments/transfer/runs_steered/<vector_type>/<split_type>/<model>/<scenario>/<eval_d>/<eval_c>/traces_*.jsonl
#     (<vector_type> = basename of VECTORS_DIR, e.g. contrastive | synthetic)
#   experiments/transfer/steered_logs/<timestamp>/gpu<i>_<model>_<tag>.log

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE_DIR="${HF_XET_CACHE_DIR:-${HF_HOME}/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="experiments/transfer/steered_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

ALPHA="${ALPHA:-5.0}"
SPLIT_TYPES="${SPLIT_TYPES:-meek}"
VECTORS_DIR="${VECTORS_DIR:-experiments/transfer/vectors/contrastive}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-30}"

# Each entry: "<gpu> <model_slug> <tag> <space-separated scenarios>"
JOBS=(
  "0 gemma-3-4b-it all gpqa_grader gpqa_insider gpqa_stanford gpqa_xml gpqa_all stanford_bbh stanford_mmlu stanford_all"
  "1 qwen3.5-9b half_a gpqa_grader gpqa_insider gpqa_stanford stanford_all"
  "2 qwen3.5-9b half_b gpqa_xml gpqa_all stanford_bbh stanford_mmlu"
  "3 gemma-3-12b-it q1 gpqa_grader gpqa_insider"
  "4 gemma-3-12b-it q2 gpqa_stanford gpqa_xml"
  "5 gemma-3-12b-it q3 gpqa_all stanford_bbh"
  "6 gemma-3-12b-it q4 stanford_mmlu stanford_all"
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
  local gpu model_slug tag
  read -r gpu model_slug tag rest <<< "${job}"
  local scenarios_args="${rest}"
  local log="${LOG_DIR}/gpu${gpu}_${model_slug}_${tag}.log"
  echo "Launching ${model_slug}/${tag} on GPU ${gpu}"
  echo "  scenarios: ${scenarios_args}"
  echo "  log: ${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python -u \
    experiments/transfer/generate_steered_traces.py \
    --model_slug "${model_slug}" \
    --split_types ${SPLIT_TYPES} \
    --scenarios ${scenarios_args} \
    --vectors_dir "${VECTORS_DIR}" \
    --alpha "${ALPHA}" \
    --batch_size "${BATCH_SIZE}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    > "${log}" 2>&1 &

  local pid="$!"
  PID_TO_GPU["${pid}"]="${gpu}"
  PID_TO_JOB["${pid}"]="${model_slug}/${tag}"
  echo "${pid}" > "${LOG_DIR}/gpu${gpu}_${model_slug}_${tag}.pid"
}

echo "Logs: ${LOG_DIR}"
echo "ALPHA=${ALPHA} SPLIT_TYPES='${SPLIT_TYPES}' VECTORS_DIR=${VECTORS_DIR} BATCH_SIZE=${BATCH_SIZE}"

for job in "${JOBS[@]}"; do
  launch_job "${job}"
  sleep "${STARTUP_STAGGER_SECONDS}"
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

echo "All steered trace jobs finished."
echo "Logs are in: ${LOG_DIR}"

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "Failed jobs (${#FAILED_JOBS[@]}):" >&2
  for job in "${FAILED_JOBS[@]}"; do
    echo "  - ${job}" >&2
  done
  exit 1
fi

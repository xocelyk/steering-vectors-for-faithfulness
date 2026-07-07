#!/usr/bin/env bash
set -uo pipefail

# Detached orchestrator: generate synthetic-vector steered traces for the meek
# split, all 3 models, sequentially on a single GPU.
#
# Designed to be launched fully detached (setsid + nohup) so it survives the
# Claude session, SSH disconnect, or terminal close. It is RESUMABLE: every
# cell whose traces_*.jsonl already exists is skipped (--skip_existing default),
# so re-running this exact script after an interruption continues where it
# stopped. (A full instance/VM shutdown still kills the process — on reopen,
# just re-run this script and it resumes at cell granularity.)
#
# Launch:
#   cd <repo-root>   # (or run from anywhere; the script cd's to the repo root)
#   setsid nohup bash experiments/transfer/run_all_steered_synthetic_meek.sh \
#     > experiments/transfer/steered_logs/orchestrator.out 2>&1 < /dev/null &
#
# Watch progress (any time, after reconnecting):
#   cat experiments/transfer/steered_logs/STATUS_synthetic_meek.txt

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE_DIR="${HF_XET_CACHE_DIR:-${HF_HOME}/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# vLLM 0.20.1 crashes at startup on the DeepGEMM FP8 warmup because deep_gemm
# is not installed; these models are bf16 and never need it.
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

GPU="${GPU:-0}"
ALPHA="${ALPHA:-5.0}"
SPLIT_TYPES="${SPLIT_TYPES:-meek}"
VECTORS_DIR="${VECTORS_DIR:-experiments/transfer/vectors/synthetic}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODELS=(gemma-3-4b-it qwen3.5-9b gemma-3-12b-it)

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="experiments/transfer/steered_logs"
mkdir -p "${LOG_DIR}"
STATUS="${LOG_DIR}/STATUS_synthetic_meek.txt"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${STATUS}"; }

count_cells() {  # $1 = model_slug
  find "experiments/transfer/runs_steered/synthetic/${SPLIT_TYPES}/$1" \
    -name '*.jsonl' 2>/dev/null | wc -l
}

wait_for_free_gpu() {
  # Wait until GPU memory is low enough to load a model (handles handoff from a
  # just-killed run). Times out after ~5 min and proceeds anyway.
  local i=0
  while [ "${i}" -lt 60 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | tr -d ' ')
    [ -z "${used}" ] && return 0
    if [ "${used}" -lt 3000 ]; then return 0; fi
    sleep 5; i=$((i+1))
  done
}

say "=== orchestrator START (run id ${TS}) ==="
say "models=${MODELS[*]} split=${SPLIT_TYPES} alpha=${ALPHA} vectors=${VECTORS_DIR} gpu=${GPU}"

FAILED=()
for model in "${MODELS[@]}"; do
  log="${LOG_DIR}/synthetic_${SPLIT_TYPES}_${model}_${TS}.log"
  say "--- ${model}: waiting for free GPU ---"
  wait_for_free_gpu
  say "--- ${model}: START (log=${log}; ${model} cells already done=$(count_cells "${model}")) ---"
  CUDA_VISIBLE_DEVICES="${GPU}" uv run --no-sync python -u \
    experiments/transfer/generate_steered_traces.py \
    --model_slug "${model}" \
    --split_types ${SPLIT_TYPES} \
    --vectors_dir "${VECTORS_DIR}" \
    --alpha "${ALPHA}" \
    --batch_size "${BATCH_SIZE}" \
    --gpu_memory_utilization 0.9 \
    > "${log}" 2>&1
  rc=$?
  done_cells=$(count_cells "${model}")
  if [ "${rc}" -eq 0 ]; then
    say "--- ${model}: DONE rc=0  cells=${done_cells} ---"
  else
    say "--- ${model}: EXIT rc=${rc}  cells=${done_cells}  (see ${log}) ---"
    FAILED+=("${model}(rc=${rc})")
  fi
done

if [ "${#FAILED[@]}" -eq 0 ]; then
  say "=== orchestrator COMPLETE — all models finished ==="
else
  say "=== orchestrator FINISHED with failures: ${FAILED[*]} ==="
fi

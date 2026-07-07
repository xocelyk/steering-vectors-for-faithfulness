#!/usr/bin/env bash
set -uo pipefail

# Detached, resumable orchestrator: generate steered traces for the meek split,
# all 3 models, sequentially on a single GPU, for ANY vector family.
#
# The vector family is the basename of VECTORS_DIR (contrastive | synthetic |
# optimization); STATUS file and per-model logs are labelled with it so runs of
# different families don't clobber each other's status. generate_steered_traces
# already writes to runs_steered/<vtype>/... so outputs never collide either.
#
# RESUMABLE: cells whose traces_*.jsonl exist are skipped (--skip_existing),
# so re-running this exact command continues where it stopped.
#
# Launch (contrastive, the default):
#   cd <repo-root>   # (or run from anywhere; the script cd's to the repo root)
#   setsid nohup bash experiments/transfer/run_all_steered_meek.sh \
#     >> experiments/transfer/steered_logs/orchestrator.out 2>&1 < /dev/null &
# Override family/split/alpha via env, e.g.:
#   VECTORS_DIR=experiments/transfer/vectors/synthetic SPLIT_TYPES=giovanni ...
#
# Watch: cat experiments/transfer/steered_logs/STATUS_<vtype>_<split>.txt

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE_DIR="${HF_XET_CACHE_DIR:-${HF_HOME}/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# vLLM 0.20.1 crashes at startup on the DeepGEMM FP8 warmup (deep_gemm not
# installed; these models are bf16 and never need it).
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

GPU="${GPU:-0}"
ALPHA="${ALPHA:-5.0}"
SPLIT_TYPES="${SPLIT_TYPES:-meek}"          # vector groups: meek/giovanni, or specific/generic for optimization
EVAL_SPLIT="${EVAL_SPLIT:-}"               # held-out test split; REQUIRED for optimization (meek|giovanni)
VECTORS_DIR="${VECTORS_DIR:-experiments/transfer/vectors/contrastive}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODELS=(gemma-3-4b-it qwen3.5-9b gemma-3-12b-it)
VTYPE="$(basename "${VECTORS_DIR}")"
EVAL_ARGS=()
[ -n "${EVAL_SPLIT}" ] && EVAL_ARGS=(--eval_split "${EVAL_SPLIT}")
TAG="$(echo "${SPLIT_TYPES}" | tr ' ' '-')${EVAL_SPLIT:+_eval-${EVAL_SPLIT}}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="experiments/transfer/steered_logs"
mkdir -p "${LOG_DIR}"
STATUS="${LOG_DIR}/STATUS_${VTYPE}_${TAG}.txt"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${STATUS}"; }

count_cells() {  # $1 = model_slug — count across all groups for this vtype/model
  find "experiments/transfer/runs_steered/${VTYPE}" -path "*/$1/*" \
    -name '*.jsonl' 2>/dev/null | wc -l
}

wait_for_free_gpu() {
  local i=0
  while [ "${i}" -lt 60 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | tr -d ' ')
    [ -z "${used}" ] && return 0
    if [ "${used}" -lt 3000 ]; then return 0; fi
    sleep 5; i=$((i+1))
  done
}

say "=== orchestrator START (run id ${TS}) vtype=${VTYPE} ==="
say "models=${MODELS[*]} split=${SPLIT_TYPES} alpha=${ALPHA} vectors=${VECTORS_DIR} gpu=${GPU}"

FAILED=()
for model in "${MODELS[@]}"; do
  log="${LOG_DIR}/steered_${VTYPE}_${TAG}_${model}_${TS}.log"
  say "--- ${model}: waiting for free GPU ---"
  wait_for_free_gpu
  say "--- ${model}: START (log=${log}; cells already done=$(count_cells "${model}")) ---"
  CUDA_VISIBLE_DEVICES="${GPU}" uv run --no-sync python -u \
    experiments/transfer/generate_steered_traces.py \
    --model_slug "${model}" \
    --split_types ${SPLIT_TYPES} \
    "${EVAL_ARGS[@]}" \
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
  say "=== orchestrator COMPLETE — all models finished (vtype=${VTYPE}) ==="
else
  say "=== orchestrator FINISHED with failures: ${FAILED[*]} ==="
fi

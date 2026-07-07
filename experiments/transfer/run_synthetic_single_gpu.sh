#!/usr/bin/env bash
set -euo pipefail

# Build all 48 synthetic steering vectors on a single GPU, sequentially:
#   gemma-3-4b-it -> qwen3.5-9b -> gemma-3-12b-it
# Each model loads, prefills the synthetic D+/D- examples, computes the DoM
# vector at the meek probe-best layer, and saves to
#   experiments/transfer/vectors/synthetic/<split>/<model>/<scenario>.pt
#
# Usage:
#   bash experiments/transfer/run_synthetic_single_gpu.sh
#   GPU=2 bash experiments/transfer/run_synthetic_single_gpu.sh
#
# Final output: 48 .pt files at experiments/transfer/vectors/synthetic/...

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU="${GPU:-0}"
SPLIT_TYPES="${SPLIT_TYPES:-meek giovanni}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-bfloat16}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE_DIR="${HF_XET_CACHE_DIR:-${HF_HOME}/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_one() {
  local model_slug="$1"
  local batch_size="$2"
  echo "=== ${model_slug} (batch=${batch_size}) on GPU ${GPU} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" uv run --no-sync python -u \
    experiments/transfer/build_synthetic_vectors.py \
    --model_slug "${model_slug}" \
    --split_types ${SPLIT_TYPES} \
    --batch_size "${batch_size}" \
    --model_dtype "${MODEL_DTYPE}" \
    --save_dtype "${SAVE_DTYPE}"
}

run_one gemma-3-4b-it 16
run_one qwen3.5-9b     8
run_one gemma-3-12b-it 4

echo
echo "Done. Vectors at experiments/transfer/vectors/synthetic/..."
find experiments/transfer/vectors/synthetic -name "*.pt" | wc -l

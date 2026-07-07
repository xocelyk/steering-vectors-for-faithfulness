#!/bin/bash
set -e

uv run python experiments/phase3/generate_steered_traces.py \
    --vectors_dir experiments/phase3/results/vectors \
    --layers 0 8 11 12 16 23 31 --alphas 0.25 0.5 0.75 1.0 \
    --strategy every_step_last_token \
    --n_gpqa 50 --batch_size 50

uv run python experiments/phase2/score_verbosity.py \
    --input experiments/phase3/results/steered_traces/

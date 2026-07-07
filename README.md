# Steering Vectors for CoT Faithfulness & Verbosity

Code for training and evaluating **steering vectors that make a model's
chain-of-thought more monitorable** — more verbose and more likely to
acknowledge cues that influence its answer. It bridges two lines of work:
steering vectors for reasoning behaviors, and faithfulness/verbosity as
measures of CoT monitorability.

> Paper: *forthcoming* (workshop paper / write-up). This repository accompanies
> that write-up and reproduces its figures and tables.

## What's here

The pipeline extracts steering vectors for **verbosity** and **cue
acknowledgment**, applies them during generation on standard benchmarks (BBH,
GPQA, MMLU), and scores the resulting traces with the monitorability
framework's LLM-as-judge. It compares two extraction methods (difference-of-
means / contrastive vs. optimization-based) and tests whether the vectors
transfer across model families.

- **Subject model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Transfer models:** `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, and a
  Qwen variant (see `MODEL_REGISTRY` in `config.py`)
- **Judges:** OpenRouter / OpenAI models via `src/grading/`

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). GPU inference uses
vLLM + nnsight; scoring calls hosted judge models.

```bash
# 1. Get the code
git clone <this-repo-url>
cd <repo>

# 2. Add the evaluation framework (measuring_cot_monitorability) as a pinned
#    git submodule. Once this is committed, future clones can instead use
#    `git clone --recurse-submodules`.
git submodule add https://github.com/ajmeek/measuring_cot_monitorability \
    third_party/measuring_cot_monitorability
git -C third_party/measuring_cot_monitorability checkout \
    7da0cfb310db80b71d842d310b9c44c9b5e8ee37

# 3. Install this project (editable) + deps, and the framework's own deps
uv sync
uv pip install -e third_party/measuring_cot_monitorability

# 4. Configure secrets
cp .env.example .env      # then fill in HF_TOKEN and a judge API key
```

This project depends on
[measuring_cot_monitorability](https://github.com/ajmeek/measuring_cot_monitorability)
(the scorers, cue system, and benchmark data), vendored as a git submodule under
`third_party/`. Override its location with `MONITORABILITY_ROOT` if you keep a
checkout elsewhere.

## Layout

```
src/steering_vectors_for_faithfulness/config.py   # central paths, model ids, HF/env helpers
src/qa/            # dataset loaders, cue injection, prompt rendering
src/grading/       # LLM-as-judge client, parsers, IO
src/prompts/       # prompt-template engine + templates
src/steering/      # steering-vector generation utilities

experiments/
  phase1/   # LDA "bias" vector + logit-lens exploration
  phase2/   # CoT trace generation; faithfulness/verbosity scoring; contrastive datasets
  phase3/   # activation collection; vector extraction; steered generation; plots
  phase4/   # optimization-based steering vectors
  transfer/ # cross-model transfer: build vectors, steer, score, probe (run_*.sh orchestrate)

third_party/measuring_cot_monitorability/   # git submodule (eval framework + data)
```

## Running the pipeline

Each phase is a set of scripts run with `uv run python experiments/<phase>/<script>.py`;
pass `--help` to any script for its flags. Roughly, in order:

1. **phase2 — data & scoring.** `generate_cot_traces.py` produces CoT traces;
   `score_faithfulness.py` / `score_verbosity.py` grade them; `build_contrastive_dataset.py`
   forms the high/low contrastive splits.
2. **phase3 — extract & apply.** `collect_activations.py` and `extract_vectors.py`
   build difference-of-means vectors; `generate_steered_traces.py` /
   `apply_steering_hf.py` generate steered traces; `run_steering_sweep.sh` sweeps
   the steering magnitude; `plot_*.py` make the figures.
3. **phase4 — optimization vectors.** `create_*_examples.py` +
   `steering_opt_batched.py` / `create_vector_optimized.py` train optimization-based
   vectors for comparison.
4. **transfer — cross-model.** The `run_*.sh` scripts orchestrate building
   vectors (contrastive / synthetic / optimization), generating steered traces
   across models and GPUs, scoring (`batch_scoring.py`), and probing
   (`run_transfer_probes.py`). Behavior is driven by env vars (`ALPHA`,
   `SPLIT_TYPES`, `VECTORS_DIR`, `HF_HOME`, …) — see each script's header.

## Data & artifacts

The trained **steering vectors** ship in-repo under `experiments/*/vectors*/`
and `experiments/transfer/vectors/` (small `.pt` files). Large intermediate
artifacts — raw/scored trace `.jsonl`, collected activations, and `.eval` logs —
are **regenerable** with the scripts above and are not committed; regenerate
them or request a data release.

## License

[MIT](LICENSE).

## Built on

| | |
|---|---|
| Steering vectors for reasoning behaviors | [Understanding Reasoning in Thinking LLMs via Steering Vectors](https://arxiv.org/abs/2506.18167) · [cvenhoff/steering-thinking-llms](https://github.com/cvenhoff/steering-thinking-llms) |
| Monitorability (faithfulness + verbosity) | [Measuring CoT Monitorability Through Faithfulness and Verbosity](https://arxiv.org/abs/2510.27378) · [ajmeek/measuring_cot_monitorability](https://github.com/ajmeek/measuring_cot_monitorability) |
| Post-hoc rationalization detection | [CoT Reasoning In The Wild Is Not Always Faithful](https://arxiv.org/abs/2503.08679) · [jettjaniak/chainscope](https://github.com/jettjaniak/chainscope) |
| Reasoning-behavior directions | [Base Models Know How to Reason, Thinking Models Learn When](https://arxiv.org/abs/2510.07364) |

"""Generate CoT traces with optional steering-vector intervention.

Two prompt input modes:

  Mode A (load from disk): --prompts_input <prompts.jsonl>
      Load prompts produced by `grading.io.write_prompts_jsonl`. The same
      prompt set can be reused across baseline + multiple steered runs.

  Mode B (build inline): --dataset {bbh,gpqa,mmlu} --cue_type {...} [--n N]
      Build prompts via `qa.build_cued_dataset` and write the resulting
      prompts.jsonl into --output_dir for downstream grading.

Examples:

    # Baseline only (no steering vector)
    uv run python -m steering.generate \
        --dataset gpqa --cue_type sycophancy --n 4 \
        --alphas 0.0 \
        --output_dir ./results/baseline/

    # Steered sweep over (layer, alpha)
    uv run python -m steering.generate \
        --vectors path/to/steering_vectors.pt \
        --layers 16 --alphas 3.0 5.0 \
        --strategy every_step_all_tokens \
        --dataset gpqa --cue_type sycophancy --n 4 \
        --output_dir ./results/steered/

    # Reuse an existing prompts file across runs
    uv run python -m steering.generate \
        --prompts_input ./results/baseline/prompts.jsonl \
        --vectors v.pt --layers 16 --alphas 5.0 \
        --output_dir ./results/steered_alt/

Outputs in --output_dir:

    prompts.jsonl                                     (mode B only; written via write_prompts_jsonl)
    config.json                                       (full run metadata; overwritten each run)
    baseline.jsonl                                    (when 0.0 is in --alphas; condition="baseline")
    layer_{L}_alpha_{a}_{strategy}.jsonl              (one per non-zero (layer, alpha) combo)

Trace files use `grading.io.write_traces_jsonl`, which appends and dedupes by
trace_id. Re-running with the same args is a no-op; running with new alphas
appends only the new rows. The prompts file is required by downstream grading
(`grading.io.load_traces` needs a prompt dict to rehydrate Trace.prompt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Optional, Union

import torch

from grading import Trace
from grading.io import load_prompts, write_prompts_jsonl, write_traces_jsonl
from qa import Prompt, build_cued_dataset
from qa.prompt import CUE_NAME_BY_TYPE, CueType, DatasetName
from qa.render import render_for_model

Strategy = Literal[
    "prompt_last_token",
    "prompt_all_tokens",
    "every_step_last_token",
    "every_step_all_tokens",
]
STRATEGIES: tuple[Strategy, ...] = (
    "prompt_last_token",
    "prompt_all_tokens",
    "every_step_last_token",
    "every_step_all_tokens",
)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_OUTPUT_DIR = Path("./results/steered_traces")
CUE_CHOICES = (*CUE_NAME_BY_TYPE.keys(), "none")
DATASET_CHOICES: tuple[DatasetName, ...] = ("bbh", "gpqa", "mmlu")


# ---------------------------------------------------------------------------
# Vector + model loading
# ---------------------------------------------------------------------------


def load_vector(path: Union[str, Path], layer: int) -> torch.Tensor:
    """Load a steering vector for `layer` from a .pt file.

    Each returned tensor has shape (hidden_size,) and is added to the residual
    stream at `model.model.layers[layer].output[0]` during generation, scaled
    by `alpha`.

    Accepted file shapes (torch.load result):

      1. Multi-layer dict-of-dicts (preferred — what most extraction scripts
         emit, including this project's vectors at experiments/phase3/results/
         vectors/steering_vectors.pt):
             {
               "vectors":  {0: tensor, 8: tensor, 16: tensor, ...},
               "norms":    {...},          # optional, ignored here
               "metadata": {...},          # optional, ignored here
             }

      2. Plain layer dict:
             {0: tensor, 8: tensor, 16: tensor, ...}

      3. Bare tensor (assumed to be the vector for the requested layer; the
         `layer` argument is used only by callers, not for indexing here):
             tensor of shape (hidden_size,)

    Raises KeyError if `layer` is missing in shapes 1 or 2.
    """
    blob = torch.load(Path(path), weights_only=False)
    if isinstance(blob, dict) and "vectors" in blob and isinstance(blob["vectors"], dict):
        per_layer = blob["vectors"]
    elif isinstance(blob, dict):
        per_layer = blob
    else:
        return torch.as_tensor(blob)

    if layer not in per_layer:
        available = sorted(per_layer.keys()) if per_layer else []
        raise KeyError(f"Layer {layer} not in vectors file; available: {available}")
    return torch.as_tensor(per_layer[layer])


def load_model(name: str, gpu_memory_utilization: float, max_model_len: int = 30000):
    """Load `name` (a HuggingFace model id) wrapped in nnsight's VLLM.

    The returned object exposes:
      - `model.trace(...)` / `tracer.invoke(...)`    — for the steering hook
      - `model.model.layers[i]`                       — residual-stream access
      - `model.vllm_entrypoint.generate`              — vLLM's own batch generate
      - `model.tokenizer`                             — the HF tokenizer

    `max_model_len` caps the vLLM context window. Default 30000 fits the
    DeepSeek-R1-Distill-Llama-8B traces produced in this project (think + answer
    rarely exceed ~25k tokens). Lower it for smaller-context models.
    """
    from nnsight.modeling.vllm import VLLM

    model = VLLM(
        name,
        dispatch=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    return model


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------


def _condition(layer: Optional[int], alpha: float, strategy: Strategy) -> str:
    """Build the `condition` string stored on each Trace.

    Two forms:
      - "baseline"                                   when alpha == 0 or no vector
      - "steered_L{layer}_a{alpha}_{strategy}"       otherwise

    The condition is the only place where steering metadata is recorded on the
    Trace object — layer/alpha/strategy are NOT separate fields. Downstream
    code that needs to filter by these values must parse the string.
    """
    if layer is None or alpha == 0.0:
        return "baseline"
    return f"steered_L{layer}_a{alpha}_{strategy}"


def _generate_batch(
    model,
    rendered_prompts: list[str],
    *,
    vector: Optional[torch.Tensor],
    layer: Optional[int],
    alpha: float,
    strategy: Strategy,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Run one batched trace call and return one generated string per prompt.

    Strategy semantics (where the residual-stream hook fires):

      prompt_last_token       prefill only, last position only
      prompt_all_tokens       prefill only, every position
      every_step_last_token   every autoregressive step, last position only
      every_step_all_tokens   every autoregressive step, every position
                              (the strategy used in this project's
                              optimization-based runs)

    `tracer.iter[:1]` scopes a hook to the prefill pass; `tracer.all()` scopes
    it to every forward pass during generation.

    The output-capture trick: nnsight's VLLM wrapper invokes vLLM's
    `vllm_entrypoint.generate` internally and discards the returned
    `RequestOutput` objects. We monkey-patch that method for the duration of
    the trace to keep them, then read `out.outputs[0].text` for each prompt.
    There is currently no other way to recover generated text from this
    wrapper (as of nnsight 0.6.x).
    """
    apply_hook = vector is not None and layer is not None and alpha != 0.0
    layer_module = model.model.layers[layer] if apply_hook else None

    captured_outputs: list = []
    orig_generate = model.vllm_entrypoint.generate

    def _capturing_generate(*args, **kwargs):
        results = orig_generate(*args, **kwargs)
        captured_outputs.extend(results)
        return results

    model.vllm_entrypoint.generate = _capturing_generate
    try:
        with model.trace(
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        ) as tracer:
            for prompt_text in rendered_prompts:
                with tracer.invoke(prompt_text):
                    if not apply_hook:
                        continue
                    if strategy == "prompt_last_token":
                        with tracer.iter[:1]:
                            hs = layer_module.output[0].clone()
                            hs[-1] += alpha * vector
                            layer_module.output = (hs, layer_module.output[1])
                    elif strategy == "prompt_all_tokens":
                        with tracer.iter[:1]:
                            hs = layer_module.output[0].clone()
                            hs[:] += alpha * vector
                            layer_module.output = (hs, layer_module.output[1])
                    elif strategy == "every_step_last_token":
                        with tracer.all():
                            hs = layer_module.output[0].clone()
                            hs[-1] += alpha * vector
                            layer_module.output = (hs, layer_module.output[1])
                    elif strategy == "every_step_all_tokens":
                        with tracer.all():
                            hs = layer_module.output[0].clone()
                            hs[:] += alpha * vector
                            layer_module.output = (hs, layer_module.output[1])
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
    finally:
        model.vllm_entrypoint.generate = orig_generate

    return [out.outputs[0].text for out in captured_outputs]


def generate(
    model,
    prompts: list[Prompt],
    *,
    vector: Optional[torch.Tensor] = None,
    layer: Optional[int] = None,
    alpha: float = 0.0,
    strategy: Strategy = "every_step_all_tokens",
    max_new_tokens: int = 10000,
    temperature: float = 0.6,
    top_p: float = 0.9,
    batch_size: int = 25,
) -> list[Trace]:
    """Generate CoT traces for `prompts`, optionally with a steering hook.

    Pipeline per prompt:
      1. Render with `qa.render.render_for_model(prompt, "distilled")` — picks
         the Jinja template at src/prompts/templates/cot/{dataset}/distilled_model.md
         and substitutes the question, choices, and (if any) cue text.
      2. Submit to vLLM via `tracer.invoke()` in chunks of `batch_size`. With
         a steering hook if vector + layer are provided and alpha != 0.
      3. Wrap the response as a Trace via `Trace.from_response(prompt,
         condition, response)`. This runs regex answer extraction on the
         response (`grading.parse.extract_answer_letter`) but does NOT call a
         judge — judging is a separate downstream step.

    Baseline mode is triggered when `vector is None` or `alpha == 0.0`. In
    that case the hook is skipped entirely and the run is plain vLLM.

    Args:
      model: an nnsight VLLM-wrapped model from `load_model`.
      prompts: list of Prompt objects (from build_cued_dataset or load_prompts).
      vector: 1-D tensor of shape (hidden_size,). Cast to float and moved to
        cuda by this function — caller doesn't need to.
      layer: int index into model.model.layers. Ignored when no hook is applied.
      alpha: scalar magnitude. The hook adds `alpha * vector` to the residual
        stream at the chosen positions.
      strategy: see `_generate_batch` for the four semantics.
      max_new_tokens, temperature, top_p, batch_size: standard sampling knobs.
        DeepSeek recommends temperature in [0.5, 0.7]; project default is 0.6.

    Returns:
      list[Trace], one per input prompt, in input order.
    """
    if vector is not None and alpha != 0.0:
        if layer is None:
            raise ValueError("`layer` is required when applying a steering vector")
        vector = vector.float().to("cuda")

    condition = _condition(layer, alpha, strategy)
    rendered = [render_for_model(p, "distilled") for p in prompts]

    responses: list[str] = []
    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start : start + batch_size]
        responses.extend(
            _generate_batch(
                model,
                chunk,
                vector=vector,
                layer=layer,
                alpha=alpha,
                strategy=strategy,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

    return [
        Trace.from_response(prompt, condition, response)
        for prompt, response in zip(prompts, responses)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Prompt input modes (mutually exclusive)
    p.add_argument("--prompts_input", type=Path, default=None,
                   help="JSONL of pre-built prompts (mode A). Mutually exclusive with --dataset.")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default=None,
                   help="Dataset to build prompts from (mode B).")
    p.add_argument("--cue_type", choices=CUE_CHOICES, default=None,
                   help="Cue type to inject (mode B). Use 'none' for no cue.")
    p.add_argument("--n", type=int, default=None,
                   help="Limit number of prompts when building (mode B).")
    p.add_argument("--subsets", type=str, nargs="+", default=None,
                   help="Optional subset filter (mode B).")

    # Steering config
    p.add_argument("--vectors", type=Path, default=None,
                   help="Path to a steering vectors .pt file. Omit for baseline-only.")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layers to apply steering at. Required if --vectors and any nonzero alpha.")
    p.add_argument("--alphas", type=float, nargs="+", required=True,
                   help="Steering strengths. 0.0 means baseline (no hook).")
    p.add_argument("--strategy", choices=STRATEGIES, default="every_step_all_tokens")

    # Model + sampling
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max_model_len", type=int, default=30000)
    p.add_argument("--max_new_tokens", type=int, default=10000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Output
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return p.parse_args(argv)


def _validate(args: argparse.Namespace) -> tuple[bool, list[float]]:
    """Validate args. Returns (mode_A_used, nonzero_alphas)."""
    mode_a = args.prompts_input is not None
    mode_b = args.dataset is not None or args.cue_type is not None

    if mode_a and mode_b:
        raise SystemExit("error: --prompts_input is mutually exclusive with --dataset/--cue_type")
    if not mode_a and not mode_b:
        raise SystemExit("error: must provide either --prompts_input (mode A) or --dataset+--cue_type (mode B)")
    if mode_b and (args.dataset is None or args.cue_type is None):
        raise SystemExit("error: mode B requires both --dataset and --cue_type")

    nonzero = [a for a in args.alphas if a != 0.0]
    if nonzero and args.vectors is None:
        raise SystemExit("error: nonzero --alphas require --vectors")
    if nonzero and not args.layers:
        raise SystemExit("error: nonzero --alphas require --layers")

    return mode_a, nonzero


def _load_or_build_prompts(args: argparse.Namespace, mode_a: bool) -> list[Prompt]:
    if mode_a:
        return list(load_prompts(args.prompts_input).values())
    cue_type: Optional[CueType] = None if args.cue_type == "none" else args.cue_type
    return build_cued_dataset(
        args.dataset,
        cue_type,
        subsets=args.subsets,
        n_questions=args.n,
    )


def _resolve_configs(args: argparse.Namespace, nonzero_alphas: list[float]) -> list[tuple[Optional[int], float]]:
    """Build the (layer, alpha) sweep.

    Baseline gets a single `(None, 0.0)` entry (no per-layer baseline — the
    baseline doesn't depend on any layer). All non-zero alphas are paired with
    every requested layer (cartesian product).

    Examples:
      --alphas 0.0                         -> [(None, 0.0)]
      --layers 16 --alphas 5.0             -> [(16, 5.0)]
      --layers 12 16 --alphas 3.0 5.0      -> [(12, 3.0), (12, 5.0), (16, 3.0), (16, 5.0)]
      --layers 16 --alphas 0.0 5.0         -> [(None, 0.0), (16, 5.0)]
    """
    configs: list[tuple[Optional[int], float]] = []
    if 0.0 in args.alphas or args.vectors is None:
        configs.append((None, 0.0))
    for layer in args.layers or []:
        for alpha in nonzero_alphas:
            configs.append((layer, alpha))
    return configs


def _config_filename(layer: Optional[int], alpha: float, strategy: Strategy) -> str:
    """Per-config trace filename. Examples:

      (None, 0.0, *)                              -> "baseline.jsonl"
      (16, 5.0, "every_step_all_tokens")          -> "layer_16_alpha_5.0_every_step_all_tokens.jsonl"

    The pattern is glob-friendly (`layer_*_alpha_*.jsonl`) so future grading
    or plotting scripts can discover all steered configs in a directory.
    """
    if layer is None or alpha == 0.0:
        return "baseline.jsonl"
    return f"layer_{layer}_alpha_{alpha}_{strategy}.jsonl"


def _write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    configs: list[tuple[Optional[int], float]],
) -> None:
    payload = {
        "model": args.model,
        "strategy": args.strategy,
        "configs": [
            {"layer": layer, "alpha": alpha, "filename": _config_filename(layer, alpha, args.strategy)}
            for layer, alpha in configs
        ],
        "vectors": str(args.vectors) if args.vectors else None,
        "prompts_input": str(args.prompts_input) if args.prompts_input else None,
        "dataset": args.dataset,
        "cue_type": args.cue_type,
        "n": args.n,
        "subsets": args.subsets,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    (output_dir / "config.json").write_text(json.dumps(payload, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    mode_a, nonzero_alphas = _validate(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_or_build_prompts(args, mode_a)
    if not prompts:
        raise SystemExit("error: no prompts to generate against")
    print(f"Loaded {len(prompts)} prompts ({'mode A' if mode_a else 'mode B'})")

    if not mode_a:
        prompts_path = args.output_dir / "prompts.jsonl"
        n_written = write_prompts_jsonl(prompts, prompts_path)
        print(f"Wrote {n_written} new prompts to {prompts_path}")

    configs = _resolve_configs(args, nonzero_alphas)
    _write_run_config(args.output_dir, args, configs)
    print(f"Sweep: {len(configs)} configs -> {args.output_dir}")

    model = load_model(args.model, args.gpu_memory_utilization, args.max_model_len)

    vector_cache: dict[int, torch.Tensor] = {}

    for layer, alpha in configs:
        out_path = args.output_dir / _config_filename(layer, alpha, args.strategy)
        cond = _condition(layer, alpha, args.strategy)
        print(f"\n=== {cond} -> {out_path.name} ===")

        vector: Optional[torch.Tensor] = None
        if layer is not None and alpha != 0.0:
            if layer not in vector_cache:
                vector_cache[layer] = load_vector(args.vectors, layer)
            vector = vector_cache[layer]

        traces = generate(
            model,
            prompts,
            vector=vector,
            layer=layer,
            alpha=alpha,
            strategy=args.strategy,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )
        n_new = write_traces_jsonl(traces, out_path)
        print(f"  wrote {n_new} new traces (total prompts: {len(traces)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

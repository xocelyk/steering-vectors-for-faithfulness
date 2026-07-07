"""Generate steered CoT traces using contrastive (or synthetic) vectors.

Per-(model_slug) worker. Loads the model once via nnsight-vLLM, then iterates
over the transfer matrix of (split_type, scenario_vector, eval_dataset, eval_cue):
  - per-cue scenarios trained on (gpqa, c) -> evaluated on (gpqa, c') for all c'
  - gpqa_all                              -> evaluated on (gpqa, c) for all c
  - per-dataset scenarios on (d, stanford) -> evaluated on (d', stanford) for all d'
  - stanford_all                          -> evaluated on (d, stanford) for all d

Steering: add `alpha * vector_normalized` to the residual stream at the saved
layer at every forward pass and every token position (every_step_all_tokens).
The unit-normalized vector is stored in the .pt; alpha=5 means a 5-norm-unit
push along that direction.

Sampling matches the baseline-trace configs per model:
  - gemma-3-{4b,12b}-it: temp=0, top_p=1, presence_penalty=0,   max_new_tokens=5000
  - qwen3.5-9b:          temp=0, top_p=1, presence_penalty=1.5, max_new_tokens=10000

Prompts: read from the baseline trace files at
  experiments/transfer/runs/<model>/baselines/<eval_dataset>/<eval_cue>/*.jsonl
We pull the exact `full_context` minus `full_response` so the steered prompt
matches the baseline character-for-character.

Outputs:
  experiments/transfer/runs_steered/<vector_type>/<split_type>/<model>/<scenario>/
    <eval_dataset>/<eval_cue>/traces_<scenario>_alpha<a>.jsonl
  where <vector_type> is the basename of --vectors_dir (e.g. contrastive,
  synthetic, optimization) so runs with different vector families never collide.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from steering_vectors_for_faithfulness.config import MODEL_REGISTRY, configure_hf_cache
configure_hf_cache()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
load_dotenv(PROJECT_ROOT / ".env")
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)


# Per-model defaults that mirror the baseline trace-gen config exactly.
MODEL_PRESENCE_PENALTY = {
    "gemma-3-4b-it": 0.0,
    "gemma-3-12b-it": 0.0,
    "qwen3.5-9b": 1.5,
}
MODEL_MAX_NEW_TOKENS = {
    "gemma-3-4b-it": 5000,
    "gemma-3-12b-it": 5000,
    "qwen3.5-9b": 10000,
}

CUES = ["stanford", "xml", "grader", "insider"]
DATASETS = ["bbh", "gpqa", "mmlu"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_slug", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--split_types", nargs="+", default=["meek", "giovanni"],
                   help="Vector sub-directory groups under <vectors_dir> to iterate. "
                        "meek/giovanni for contrastive & synthetic; specific/generic "
                        "for optimization (then also pass --eval_split).")
    p.add_argument("--eval_split", default=None, choices=["meek", "giovanni"],
                   help="Held-out split for evaluation prompts "
                        "(<split>_test_task_ids.txt). Default: same name as the "
                        "vector group (correct for meek/giovanni vectors). Set this "
                        "for optimization vectors, whose groups (specific/generic) "
                        "are training modes, not data splits.")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Restrict to these scenarios. Default: all 8.")
    p.add_argument("--eval_cells", nargs="+", default=None,
                   help="Restrict to specific (dataset,cue) eval cells, formatted dataset:cue.")

    p.add_argument("--vectors_dir", type=Path, default=Path("experiments/transfer/vectors/contrastive"))
    p.add_argument("--baselines_dir", type=Path, default=Path("experiments/transfer/runs"))
    p.add_argument("--splits_dir", type=Path, default=Path("experiments/transfer/splits"))
    p.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/runs_steered"))

    p.add_argument("--alpha", type=float, default=5.0,
                   help="Steering strength applied to the unit-normalized vector. Default 5.0.")
    p.add_argument("--strategy", default="every_step_all_tokens",
                   choices=["every_step_all_tokens", "every_step_last_token",
                            "prompt_all_tokens", "prompt_last_token"])

    # Sampling — defaults overridden per model below.
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Default: model-specific (gemma=5000, qwen=10000).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--presence_penalty", type=float, default=None,
                   help="Default: model-specific (gemma=0.0, qwen=1.5).")
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # vLLM engine.
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=None,
                   help="vLLM max_model_len. Defaults to model-config max if unset.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Number of prompts per outer trace() call. vLLM does in-flight "
                        "batching internally; this only controls how many prompts are "
                        "submitted at once. Larger = more parallelism, more KV cache.")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit_per_cell", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def build_transfer_matrix(scenarios: list[str] | None,
                          eval_filter: list[str] | None) -> list[tuple[str, str, str]]:
    if scenarios is None:
        scenarios = ["gpqa_grader", "gpqa_insider", "gpqa_stanford", "gpqa_xml",
                     "gpqa_all", "stanford_bbh", "stanford_mmlu", "stanford_all"]

    pairs: set[tuple[str, str, str]] = set()
    for sc in scenarios:
        if sc in ("gpqa_grader", "gpqa_insider", "gpqa_stanford", "gpqa_xml", "gpqa_all"):
            for c in CUES:
                pairs.add((sc, "gpqa", c))
        if sc in ("stanford_bbh", "stanford_mmlu", "gpqa_stanford", "stanford_all"):
            for d in DATASETS:
                pairs.add((sc, d, "stanford"))
        if sc not in ("gpqa_grader", "gpqa_insider", "gpqa_stanford", "gpqa_xml",
                      "gpqa_all", "stanford_bbh", "stanford_mmlu", "stanford_all"):
            print(f"WARNING: unknown scenario {sc}; skipping")

    if eval_filter:
        wanted = set()
        for s in eval_filter:
            d, c = s.split(":")
            wanted.add((d, c))
        pairs = {(sc, d, c) for (sc, d, c) in pairs if (d, c) in wanted}

    return sorted(pairs)


def load_vector(vectors_dir: Path, split_type: str, model_slug: str, scenario: str
                ) -> tuple[torch.Tensor, int, dict]:
    p = vectors_dir / split_type / model_slug / f"{scenario}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Vector not found: {p}")
    v = torch.load(p, weights_only=False, map_location="cpu")
    # contrastive / synthetic: top-level vector_normalized + layer.
    if "vector_normalized" in v:
        return v["vector_normalized"].float(), int(v["layer"]), v
    # optimization: {"vectors": {layer: unit_tensor}, "metadata": {"layer": ...}}.
    if "vectors" in v and v["vectors"]:
        vm = v["vectors"]
        layer = int(v.get("metadata", {}).get("layer", next(iter(vm))))
        vec = vm[layer] if layer in vm else next(iter(vm.values()))
        return vec.float(), layer, v
    raise KeyError(f"{p}: unrecognized vector payload, keys={sorted(v)}")


def load_test_task_ids(splits_dir: Path, model_slug: str, dataset: str, cue: str,
                       split_type: str) -> list[str]:
    p = splits_dir / model_slug / dataset / cue / f"{split_type}_test_task_ids.txt"
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def load_baseline_records(baselines_dir: Path, model_slug: str, dataset: str, cue: str
                          ) -> dict[str, dict]:
    d = baselines_dir / model_slug / "baselines" / dataset / cue
    matches = sorted(d.glob("*.jsonl"))
    if not matches:
        return {}
    chosen = matches[-1]
    out: dict[str, dict] = {}
    with open(chosen) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def recover_prompt(rec: dict) -> str:
    fc = rec.get("full_context", "")
    fr = rec.get("full_response", "")
    if fc and fr and fc.endswith(fr):
        return fc[: -len(fr)]
    return fc


def get_layer_module(model, layer: int):
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
    ]
    for path in candidate_paths:
        obj = model
        try:
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            return getattr(obj, path[-1])[layer]
        except (AttributeError, TypeError, IndexError):
            continue
    raise AttributeError("Could not locate transformer layers.")


def load_model_vllm(model_name: str, gpu_memory_utilization: float,
                    max_model_len: int | None, seed: int):
    from nnsight.modeling.vllm import VLLM

    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} via nnsight-vLLM...", flush=True)
    kwargs = dict(
        dispatch=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        # On a FUSE-backed network volume, vLLM auto-prefetch is disabled so
        # safetensors mmap reads hang in folio I/O. Force prefetch (one-shot
        # bulk read into RAM, then load) to avoid 10+ min hangs on the second
        # shard. Harmless on local disk.
        safetensors_load_strategy="prefetch",
        # Gated-DeltaNet models (qwen3.5-9b) JIT-compile a FlashInfer Hopper
        # kernel that fails to build under CUDA 12.4 (uses cuda::ptx tensormap
        # APIs added in CUDA 12.5). Force the Triton GDN prefill backend, which
        # compiles via Triton instead of nvcc. No-op for non-GDN models
        # (gemma-*) since they don't register the chunk_gated_delta_rule op.
        additional_config={"gdn_prefill_backend": "triton"},
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    model = VLLM(model_name, **kwargs)
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded.", flush=True)
    return model


def generate_one_cell(
    *, model, layer_module,
    vector_unit: torch.Tensor, alpha: float,
    prompts: list[str], task_ids: list[str], baselines_index: dict[str, dict],
    batch_size: int, strategy: str,
    max_tokens: int, temperature: float, top_p: float,
    presence_penalty: float, repetition_penalty: float, seed: int,
) -> list[dict]:
    """Generate a full cell's worth of steered traces.

    Submits `batch_size` prompts per outer trace() call so vLLM's in-flight
    batching has many prompts to pack into the KV cache. Each prompt gets the
    same steering hook on the chosen layer.
    """
    results: list[dict] = []
    n = len(prompts)
    chunk = max(1, batch_size)

    for start in range(0, n, chunk):
        sub_prompts = prompts[start: start + chunk]
        sub_ids = task_ids[start: start + chunk]

        captured: list = []
        orig_generate = model.vllm_entrypoint.generate

        def _capturing_generate(*a, **kw):
            outs = orig_generate(*a, **kw)
            captured.extend(outs)
            return outs

        model.vllm_entrypoint.generate = _capturing_generate
        try:
            with model.trace(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ) as tracer:
                for prompt in sub_prompts:
                    with tracer.invoke(prompt):
                        if alpha != 0.0:
                            if strategy == "every_step_all_tokens":
                                with tracer.all():
                                    hs = layer_module.output[0].clone()
                                    hs[:] += alpha * vector_unit.to(hs.device, hs.dtype)
                                    layer_module.output = (hs, layer_module.output[1])
                            elif strategy == "every_step_last_token":
                                with tracer.all():
                                    hs = layer_module.output[0].clone()
                                    hs[-1] += alpha * vector_unit.to(hs.device, hs.dtype)
                                    layer_module.output = (hs, layer_module.output[1])
                            elif strategy == "prompt_all_tokens":
                                with tracer.iter[:1]:
                                    hs = layer_module.output[0].clone()
                                    hs[:] += alpha * vector_unit.to(hs.device, hs.dtype)
                                    layer_module.output = (hs, layer_module.output[1])
                            elif strategy == "prompt_last_token":
                                with tracer.iter[:1]:
                                    hs = layer_module.output[0].clone()
                                    hs[-1] += alpha * vector_unit.to(hs.device, hs.dtype)
                                    layer_module.output = (hs, layer_module.output[1])
        finally:
            model.vllm_entrypoint.generate = orig_generate

        # captured has one entry per prompt, in the order they were invoked.
        for i, (tid, prompt) in enumerate(zip(sub_ids, sub_prompts)):
            req_out = captured[i]
            response = req_out.outputs[0].text
            num_tokens = len(req_out.outputs[0].token_ids)
            full_text = prompt + response
            base_rec = baselines_index.get(tid, {})
            results.append({
                "task_id": tid,
                "model": base_rec.get("model"),
                "dataset": base_rec.get("dataset"),
                "cue_name": base_rec.get("cue_name"),
                "cue_text": base_rec.get("cue_text"),
                "cue_target_letter": base_rec.get("cue_target_letter"),
                "has_cue": base_rec.get("has_cue", True),
                "judge_prompt": base_rec.get("judge_prompt"),
                "question": base_rec.get("question"),
                "target": base_rec.get("target"),
                "correct_letter": base_rec.get("correct_letter"),
                "correct_index": base_rec.get("correct_index"),
                "choices": base_rec.get("choices"),
                "full_context": full_text,
                "full_response": response,
                "num_tokens": int(num_tokens),
            })

    return results


def main() -> None:
    args = parse_args()
    model_name = MODEL_REGISTRY[args.model_slug]
    presence_penalty = (args.presence_penalty if args.presence_penalty is not None
                        else MODEL_PRESENCE_PENALTY[args.model_slug])
    max_new_tokens = (args.max_new_tokens if args.max_new_tokens is not None
                      else MODEL_MAX_NEW_TOKENS[args.model_slug])

    matrix = build_transfer_matrix(args.scenarios, args.eval_cells)
    # Vector family (contrastive | synthetic | optimization | ...), taken from
    # the --vectors_dir basename. It is the top-level component of the output
    # path so runs with different vector families never overwrite each other.
    vector_type = args.vectors_dir.name
    # The vector group ("split_type") selects which vector .pt to load; the
    # held-out eval split selects the test prompts. They coincide for
    # meek/giovanni vectors, but optimization groups (specific/generic) are not
    # data splits, so --eval_split decouples them. When they differ, the eval
    # split is encoded in the output path so the two never collide.
    def group_component(split_type: str) -> str:
        if args.eval_split is None or args.eval_split == split_type:
            return split_type
        return f"{split_type}__eval-{args.eval_split}"

    plan: list[tuple[str, str, str, str, Path]] = []
    skipped = 0
    for split_type in args.split_types:
        for scenario, eval_d, eval_c in matrix:
            out_dir = (args.output_dir / vector_type / group_component(split_type)
                       / args.model_slug / scenario / eval_d / eval_c)
            out_path = out_dir / f"traces_{scenario}_alpha{args.alpha}.jsonl"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            plan.append((split_type, scenario, eval_d, eval_c, out_path))

    print(f"Model: {args.model_slug}  ({model_name})")
    print(f"Sampling: temp={args.temperature} top_p={args.top_p} "
          f"presence_penalty={presence_penalty} repetition_penalty={args.repetition_penalty} "
          f"max_new_tokens={max_new_tokens} alpha={args.alpha}")
    print(f"Scenarios: {sorted(set(s for s, *_ in matrix))}")
    print(f"Cells: {len(matrix)} per split_type, {len(args.split_types)} split_types")
    print(f"Plan: {len(plan)} cells to run, {skipped} already complete")

    if args.dry_run:
        for split_type, scenario, eval_d, eval_c, out_path in plan:
            print(f"  [{split_type}] {scenario} -> ({eval_d},{eval_c})  ->  {out_path}")
        return
    if not plan:
        return

    model = load_model_vllm(model_name, args.gpu_memory_utilization,
                            args.max_model_len, args.seed)

    baseline_cache: dict[tuple[str, str], dict[str, dict]] = {}
    failed: list[tuple[str, str, str, str, str]] = []

    for split_type, scenario, eval_d, eval_c, out_path in plan:
        print(f"\n[{time.strftime('%H:%M:%S')}] === {args.model_slug} [{split_type}] "
              f"{scenario} -> ({eval_d},{eval_c}) ===", flush=True)
        try:
            vec_unit, layer, vec_meta = load_vector(args.vectors_dir, split_type,
                                                    args.model_slug, scenario)
            test_ids = load_test_task_ids(args.splits_dir, args.model_slug,
                                          eval_d, eval_c,
                                          args.eval_split or split_type)
            if not test_ids:
                print(f"  SKIP: no test_task_ids")
                failed.append((split_type, scenario, eval_d, eval_c, "missing_test_split"))
                continue
            if args.limit_per_cell is not None:
                test_ids = test_ids[: args.limit_per_cell]

            key = (eval_d, eval_c)
            if key not in baseline_cache:
                baseline_cache[key] = load_baseline_records(args.baselines_dir,
                                                            args.model_slug, eval_d, eval_c)
            base_idx = baseline_cache[key]
            if not base_idx:
                print(f"  SKIP: no baseline JSONL")
                failed.append((split_type, scenario, eval_d, eval_c, "missing_baseline"))
                continue

            prompts: list[str] = []
            kept_ids: list[str] = []
            for tid in test_ids:
                rec = base_idx.get(tid)
                if rec is None:
                    continue
                p_str = recover_prompt(rec)
                if not p_str:
                    continue
                prompts.append(p_str)
                kept_ids.append(tid)
            if not prompts:
                print(f"  SKIP: no prompts recovered")
                failed.append((split_type, scenario, eval_d, eval_c, "no_prompts"))
                continue

            layer_module = get_layer_module(model, layer)
            print(f"  layer={layer} alpha={args.alpha} prompts={len(prompts)}/{len(test_ids)}", flush=True)

            t0 = time.time()
            traces = generate_one_cell(
                model=model, layer_module=layer_module,
                vector_unit=vec_unit, alpha=args.alpha,
                prompts=prompts, task_ids=kept_ids, baselines_index=base_idx,
                batch_size=args.batch_size, strategy=args.strategy,
                max_tokens=max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                presence_penalty=presence_penalty, repetition_penalty=args.repetition_penalty,
                seed=args.seed,
            )
            elapsed = time.time() - t0

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                for r in traces:
                    r["scenario"] = scenario
                    r["split_type"] = split_type
                    r["alpha"] = args.alpha
                    r["layer"] = layer
                    r["strategy"] = args.strategy
                    r["vector_norm"] = vec_meta.get("vector_norm")
                    r["model_slug"] = args.model_slug
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(out_path)
            print(f"  wrote {len(traces)} traces in {elapsed:.1f}s -> {out_path}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)
            failed.append((split_type, scenario, eval_d, eval_c, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failed:
        print(f"\nFailed/skipped {len(failed)} cells:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()

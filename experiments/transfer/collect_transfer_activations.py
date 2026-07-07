"""Collect mean-pooled activations for transfer faithfulness probes.

This script reads scored transfer traces, filters to valid cued rows with
faithfulness labels, runs prefill passes through the target model, and saves
one all-layer activation file per model/dataset/cue setting.

It intentionally does not create train/test splits. Downstream probe scripts can
split by task_id after activations have been collected.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from common import CANONICAL_DATASETS, read_jsonl, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
from steering_vectors_for_faithfulness.config import MODEL_REGISTRY, configure_hf_cache
configure_hf_cache()
load_dotenv(PROJECT_ROOT / ".env")

_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)


CUE_DIRS = ["stanford", "xml", "grader", "insider"]


@dataclass
class FilterStats:
    source_rows: int = 0
    rows_with_cue: int = 0
    rows_with_label: int = 0
    filtered_not_cued: int = 0
    filtered_missing_label: int = 0
    filtered_metadata_incomplete: int = 0
    filtered_llm_degenerate: int = 0
    filtered_missing_context: int = 0
    filtered_missing_response: int = 0
    filtered_nan_activations: int = 0
    collected_rows: int = 0
    positive_labels: int = 0
    negative_labels: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_slug", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--model", default=None, help="HF model id; defaults from --model_slug")
    parser.add_argument("--datasets", nargs="+", default=CANONICAL_DATASETS, choices=CANONICAL_DATASETS)
    parser.add_argument("--cues", nargs="+", default=CUE_DIRS, choices=CUE_DIRS)
    parser.add_argument("--source_dir", type=Path, default=Path("experiments/transfer/runs_scored"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/activations"))
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--token_strategy", choices=["mean_all_response"], default="mean_all_response")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="bfloat16",
                        help="Save dtype for stored activations. bfloat16 has the same range as fp32, "
                        "avoiding fp16 overflow for layers whose mean activation exceeds ~65504.")
    parser.add_argument("--model_dtype", choices=["float16", "float32", "bfloat16"], default="bfloat16",
                        help="Dtype to load the model in.")
    parser.add_argument("--device", default="cuda",
                        help="Device for the model (typically cuda; per-worker CUDA_VISIBLE_DEVICES selects the GPU).")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Optional truncation of full_text to this many tokens.")
    parser.add_argument("--limit", type=int, default=None, help="Limit valid rows per setting")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def get_layer_module(model: Any, layer: int) -> Any:
    """Return a transformer block for common nnsight/VLLM wrappers."""
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
    raise AttributeError(
        "Could not locate transformer layers. Tried: "
        + ", ".join(".".join(path) for path in candidate_paths)
    )


def infer_model_dims(model_name: str) -> tuple[int, int]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=_hf_token)
    text_config = getattr(config, "text_config", None)
    n_layers = getattr(text_config, "num_hidden_layers", None) if text_config else None
    hidden = getattr(text_config, "hidden_size", None) if text_config else None
    n_layers = n_layers or getattr(config, "num_hidden_layers")
    hidden = hidden or getattr(config, "hidden_size")
    return int(n_layers), int(hidden)


def trace_path_for(source_dir: Path, model_slug: str, dataset: str, cue_key: str) -> Path | None:
    cue_dir = source_dir / model_slug / "baselines" / dataset / cue_key
    matches = sorted(cue_dir.glob("*.jsonl"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: multiple JSONL files in {cue_dir}; using {matches[-1]}")
    return matches[-1]


def response_prompt_boundary(record: dict[str, Any], tokenizer: Any) -> tuple[str, int]:
    full_context = record.get("full_context", "")
    full_response = record.get("full_response", "")
    if not full_context:
        raise ValueError(f"{record.get('task_id')} missing full_context")
    if not full_response:
        raise ValueError(f"{record.get('task_id')} missing full_response")
    if full_context.endswith(full_response):
        prompt = full_context[: -len(full_response)]
    else:
        prompt = full_context.replace(full_response, "", 1)
    prompt_n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return full_context, prompt_n_tokens


def valid_records(records: list[dict[str, Any]], limit: int | None = None) -> tuple[list[dict[str, Any]], FilterStats]:
    stats = FilterStats(source_rows=len(records))
    out: list[dict[str, Any]] = []
    for record in records:
        if not record.get("has_cue"):
            stats.filtered_not_cued += 1
            continue
        stats.rows_with_cue += 1
        if record.get("faithfulness_score") not in (0, 1):
            stats.filtered_missing_label += 1
            continue
        stats.rows_with_label += 1
        if record.get("degenerate_reason"):
            stats.filtered_metadata_incomplete += 1
            continue
        if record.get("degenerate_score") == 1:
            stats.filtered_llm_degenerate += 1
            continue
        if not record.get("full_context"):
            stats.filtered_missing_context += 1
            continue
        if not record.get("full_response"):
            stats.filtered_missing_response += 1
            continue
        out.append(record)
        if record.get("faithfulness_score") == 1:
            stats.positive_labels += 1
        else:
            stats.negative_labels += 1
        if limit is not None and len(out) >= limit:
            break
    stats.collected_rows = len(out)
    return out, stats


def collect_one(
    model: Any,
    tokenizer: Any,
    full_text: str,
    prompt_n_tokens: int,
    layers: list[int],
    save_dt: torch.dtype,
    device: str,
    max_seq_len: int | None,
) -> dict[int, torch.Tensor]:
    """Run one prompt through the model with hooks; return per-layer mean activations
    over the response tokens. Uses standard HF transformers — no vLLM, no chunked
    prefill, no batched iteration scheduling that can produce NaN."""
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    if max_seq_len is not None and input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        prompt_n_tokens = min(prompt_n_tokens, max_seq_len - 1)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer in layers:
        layer_module = get_layer_module(model, layer)
        def make_hook(lid: int):
            def hook(module: Any, inp: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                # hidden shape: (1, seq_len, hidden_size)
                response = hidden[0, prompt_n_tokens:, :]
                captured[lid] = response.float().mean(dim=0).to(save_dt).cpu()
            return hook
        hooks.append(layer_module.register_forward_hook(make_hook(layer)))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return captured


def load_model(
    model_name: str,
    model_dtype: torch.dtype,
    device: str,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=_hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {model_name} with HuggingFace transformers (dtype={model_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        trust_remote_code=True,
        token=_hf_token,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def token_stats(records: list[dict[str, Any]], tokenizer: Any) -> dict[str, float | int | None]:
    lengths = []
    response_lengths = []
    for record in records:
        full_text, prompt_n = response_prompt_boundary(record, tokenizer)
        n_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        lengths.append(n_tokens)
        response_lengths.append(max(0, n_tokens - prompt_n))
    if not lengths:
        return {
            "n": 0,
            "mean_total_tokens": None,
            "max_total_tokens": None,
            "mean_response_tokens": None,
            "max_response_tokens": None,
        }
    return {
        "n": len(lengths),
        "mean_total_tokens": float(np.mean(lengths)),
        "max_total_tokens": int(max(lengths)),
        "mean_response_tokens": float(np.mean(response_lengths)),
        "max_response_tokens": int(max(response_lengths)),
    }


def existing_activation_is_valid(
    activation_path: Path,
    metadata_path: Path,
    expected_layers: list[int],
    expected_hidden_size: int,
    expected_task_ids: list[str] | None = None,
) -> tuple[bool, str]:
    if not activation_path.exists():
        return False, "missing activation file"
    if not metadata_path.exists():
        return False, "missing metadata file"
    try:
        data = torch.load(activation_path, weights_only=False, map_location="cpu")
    except Exception as exc:
        return False, f"could not load activation file: {exc}"
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        return False, f"could not load metadata file: {exc}"

    task_ids = data.get("task_ids")
    labels = data.get("labels")
    layers = data.get("layers")
    saved_expected = data.get("expected_task_ids")
    if not isinstance(task_ids, list):
        return False, "task_ids missing or not a list"
    if not torch.is_tensor(labels):
        return False, "labels missing or not a tensor"
    if len(task_ids) != int(labels.shape[0]):
        return False, f"task_ids/labels length mismatch: {len(task_ids)} vs {tuple(labels.shape)}"
    if list(layers or []) != expected_layers:
        return False, f"layer list mismatch: {layers} vs {expected_layers}"

    if expected_task_ids is not None:
        # Strict completeness: a saved cell is only valid if it covers every
        # non-degenerate input record. NaN-dropped rows count as missing and
        # force a re-extract.
        compare_to = saved_expected if isinstance(saved_expected, list) else task_ids
        if list(compare_to) != list(expected_task_ids):
            return (
                False,
                f"input record set has changed since save (saved expected n={len(compare_to)} vs current n={len(expected_task_ids)})",
            )
        if list(task_ids) != list(expected_task_ids):
            missing = len(expected_task_ids) - len(task_ids)
            return (
                False,
                f"incomplete: saved n={len(task_ids)} vs expected n={len(expected_task_ids)} (missing {missing}, likely NaN-dropped)",
            )

    for layer in expected_layers:
        key = f"layer_{layer}"
        value = data.get(key)
        if not torch.is_tensor(value):
            return False, f"{key} missing or not a tensor"
        if tuple(value.shape) != (len(task_ids), expected_hidden_size):
            return False, f"{key} shape mismatch: {tuple(value.shape)} vs {(len(task_ids), expected_hidden_size)}"
        if value.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            return False, f"{key} unexpected dtype: {value.dtype}"
        # Reject any saved NaN/inf in the layer tensor.
        sample = value.float()
        if torch.isnan(sample).any().item():
            return False, f"{key} contains NaN"
        if torch.isinf(sample).any().item():
            return False, f"{key} contains inf"
    return True, "ok"


def process_setting(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    stats: FilterStats,
    source_trace: Path,
    out_dir: Path,
    model_name: str,
    model_slug: str,
    dataset: str,
    cue: str,
    layers: list[int],
    dtype: torch.dtype,
    max_seq_len: int | None,
    device: str,
    dry_run: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model_name,
        "model_slug": model_slug,
        "dataset": dataset,
        "cue": cue,
        "source_trace": str(source_trace),
        "token_strategy": "mean_all_response",
        "layers": layers,
        "filter_stats": asdict(stats),
    }
    metadata["token_stats"] = token_stats(records, tokenizer)

    if dry_run:
        write_json(out_dir / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2))
        return

    if not records:
        # No rows to extract: write metadata only so downstream tools can see why.
        write_json(out_dir / "metadata.json", metadata)
        print(f"No valid records for {model_slug}/{dataset}/{cue}; wrote metadata only")
        return

    expected_task_ids = [record["task_id"] for record in records]
    layer_values: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    for record in tqdm(records, desc=f"{dataset}/{cue}"):
        full_text, prompt_n = response_prompt_boundary(record, tokenizer)
        per_layer = collect_one(
            model=model,
            tokenizer=tokenizer,
            full_text=full_text,
            prompt_n_tokens=prompt_n,
            layers=layers,
            save_dt=dtype,
            device=device,
            max_seq_len=max_seq_len,
        )
        for layer, value in per_layer.items():
            layer_values[layer].append(value)

    layer_tensors: dict[int, torch.Tensor] = {layer: torch.stack(layer_values[layer]) for layer in layers}

    # Refuse to save if any row has NaN/inf in any layer. Silently dropping
    # bad rows hides upstream bugs in the trace pipeline. A loud failure
    # leaves no .pt for this cell so it is obvious which cells need re-extracting.
    bad_mask = torch.zeros(len(records), dtype=torch.bool)
    for layer in layers:
        t = layer_tensors[layer].float()
        bad_mask = bad_mask | torch.isnan(t).any(dim=1) | torch.isinf(t).any(dim=1)
    n_bad = int(bad_mask.sum().item())
    if n_bad:
        bad_idxs = bad_mask.nonzero(as_tuple=False).flatten().tolist()
        bad_task_ids = [expected_task_ids[i] for i in bad_idxs]
        raise RuntimeError(
            f"NaN/inf in activations for {n_bad}/{len(records)} records in "
            f"{model_slug}/{dataset}/{cue}; refusing to save. Bad task_ids: {bad_task_ids}"
        )

    saved_labels = torch.tensor(
        [int(record["faithfulness_score"]) for record in records],
        dtype=torch.float32,
    )

    activations = {
        "task_ids": expected_task_ids,
        "expected_task_ids": expected_task_ids,
        "labels": saved_labels,
        "layers": layers,
        "model": model_name,
        "model_slug": model_slug,
        "dataset": dataset,
        "cue": cue,
        "token_strategy": "mean_all_response",
        "source_trace": str(source_trace),
    }
    for layer in layers:
        activations[f"layer_{layer}"] = layer_tensors[layer]

    stats.collected_rows = len(expected_task_ids)
    stats.positive_labels = int((saved_labels == 1).sum().item())
    stats.negative_labels = int((saved_labels == 0).sum().item())
    metadata["filter_stats"] = asdict(stats)

    out_path = out_dir / "mean_all_response.pt"
    tmp_path = out_dir / "mean_all_response.pt.tmp"
    # Atomic write: a Ctrl-C / kill during torch.save leaves only the .tmp,
    # never a partially-written mean_all_response.pt that the skip check would
    # accept. Replace is atomic on POSIX.
    torch.save(activations, tmp_path)
    tmp_path.replace(out_path)

    # Write metadata.json AFTER the .pt is in place so the presence of a
    # complete metadata.json implies a complete .pt.
    write_json(out_dir / "metadata.json", metadata)
    print(f"Saved {out_path} ({len(expected_task_ids)} examples)")


def main() -> None:
    args = parse_args()
    model_name = args.model or MODEL_REGISTRY[args.model_slug]
    n_layers, hidden_size = infer_model_dims(model_name)
    layers = args.layers or list(range(n_layers))
    dtype = save_dtype(args.dtype)
    model_dtype = save_dtype(args.model_dtype)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=_hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use dry-run tokenizer before loading the model.
    settings: list[tuple[str, str, Path, list[dict[str, Any]], FilterStats, Path]] = []
    for dataset in args.datasets:
        for cue in args.cues:
            source_trace = trace_path_for(args.source_dir, args.model_slug, dataset, cue)
            out_dir = args.output_dir / args.model_slug / dataset / cue
            if source_trace is None:
                print(f"WARNING: no scored trace for {args.model_slug}/{dataset}/{cue}")
                continue
            records, stats = valid_records(read_jsonl(source_trace), limit=args.limit)
            if args.skip_existing and not args.dry_run:
                ok, reason = existing_activation_is_valid(
                    out_dir / "mean_all_response.pt",
                    out_dir / "metadata.json",
                    layers,
                    hidden_size,
                    expected_task_ids=[r["task_id"] for r in records],
                )
                if ok:
                    print(f"SKIP valid existing {out_dir / 'mean_all_response.pt'}")
                    continue
                if (out_dir / "mean_all_response.pt").exists() or (out_dir / "metadata.json").exists():
                    print(f"RERUN existing invalid {out_dir}: {reason}")
            settings.append((dataset, cue, source_trace, records, stats, out_dir))

    print(f"Model: {model_name}")
    print(f"Layers: {len(layers)} / {n_layers}; hidden_size={hidden_size}; save dtype={args.dtype}; model dtype={args.model_dtype}")
    print(f"Settings: {len(settings)}")

    model = None
    if not args.dry_run and settings:
        model, tokenizer = load_model(
            model_name,
            model_dtype=model_dtype,
            device=args.device,
        )

    failed_settings: list[tuple[str, str, str]] = []
    for dataset, cue, source_trace, records, stats, out_dir in settings:
        print(f"\n=== {args.model_slug}/{dataset}/{cue}: {len(records)} valid rows ===")
        try:
            process_setting(
                model=model,
                tokenizer=tokenizer,
                records=records,
                stats=stats,
                source_trace=source_trace,
                out_dir=out_dir,
                model_name=model_name,
                model_slug=args.model_slug,
                dataset=dataset,
                cue=cue,
                layers=layers,
                dtype=dtype,
                max_seq_len=args.max_seq_len,
                device=args.device,
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"FAILED {args.model_slug}/{dataset}/{cue}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed_settings.append((dataset, cue, f"{type(exc).__name__}: {exc}"))

    if failed_settings:
        print(f"\n{len(failed_settings)} setting(s) failed for {args.model_slug}:")
        for dataset, cue, reason in failed_settings:
            print(f"  - {dataset}/{cue}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

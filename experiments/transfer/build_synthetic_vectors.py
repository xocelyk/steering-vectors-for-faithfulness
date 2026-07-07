"""Extract activations from synthetic D+/D- examples and build steering vectors.

Per-(model_slug) worker. Loads the model once, then iterates
(split_type, scenario) and for each:
  1. Reads the synthetic JSONL (rows of prompt + completion + polarity).
  2. Prefills (prompt + completion) for each row, captures the mean activation
     over completion tokens at every layer.
  3. Pools D+ and D- activations across constituent cells.
  4. Picks the layer the same way build_contrastive_vectors.py does:
        single-cell scenario -> probes/meek/<model>__<dataset>__<cue>.json
        aggregate scenario   -> probes/meek/<model>__<scenario>.json
  5. Vector = mean(D+ activations) - mean(D- activations) at the chosen layer.

Outputs:
  experiments/transfer/activations_synthetic/<split_type>/<model>/<scenario>.pt
  experiments/transfer/vectors/synthetic/<split_type>/<model>/<scenario>.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_contrastive_vectors import MODEL_REGISTRY, SCENARIOS
from common import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from steering_vectors_for_faithfulness.config import configure_hf_cache
configure_hf_cache()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
load_dotenv(PROJECT_ROOT / ".env")
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_slug", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--split_types", nargs="+", default=["meek", "giovanni"], choices=["meek", "giovanni"])
    p.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))

    p.add_argument("--examples_dir", type=Path, default=Path("experiments/transfer/synthetic_examples"))
    p.add_argument("--probes_dir", type=Path, default=Path("experiments/transfer/probes/meek"),
                   help="Probe artifact dir to read best_by_roc.layer from.")
    p.add_argument("--activations_dir", type=Path,
                   default=Path("experiments/transfer/activations_synthetic"))
    p.add_argument("--vectors_dir", type=Path,
                   default=Path("experiments/transfer/vectors/synthetic"))

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--save_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def get_layer_module(model: Any, layer: int) -> Any:
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


def get_layer_stack(model: Any) -> Any:
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
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Could not locate transformer layer stack.")


def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} on {device} dtype={model_dtype}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=_hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # easier for completion-region masks
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[model_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, torch_dtype=dtype,
        trust_remote_code=True, token=_hf_token,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded.", flush=True)
    return model, tokenizer


def collect_activations_for_jsonl(
    *, model, tokenizer, jsonl_path: Path, n_layers: int, hidden_size: int,
    batch_size: int, save_dtype: torch.dtype, device: str,
) -> dict[str, Any]:
    """Stream the JSONL, run prefill in batches, mean-pool over completion
    tokens at every layer. Returns a dict with task_ids, polarities, datasets,
    cues, and per-layer (n_rows, hidden) tensors."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n_rows = len(rows)
    if n_rows == 0:
        return {"empty": True}

    # Pre-tokenize each row: full = prompt + completion; record the prompt token
    # length so we can mask the completion tokens during pooling.
    tokenized: list[dict] = []
    for r in rows:
        prompt_ids = tokenizer.encode(r["prompt"], add_special_tokens=False)
        full_ids = tokenizer.encode(r["prompt"] + r["completion"], add_special_tokens=False)
        tokenized.append({
            "row": r,
            "input_ids": full_ids,
            "prompt_len": len(prompt_ids),
        })

    n_kept = len(tokenized)
    layer_buffers: dict[int, torch.Tensor] = {
        i: torch.zeros((n_kept, hidden_size), dtype=save_dtype) for i in range(n_layers)
    }
    captured: dict[int, torch.Tensor] = {}  # populated by hooks each batch

    layer_stack = get_layer_stack(model)
    handles = []
    for i in range(n_layers):
        def make_hook(idx):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = hidden
            return hook
        handles.append(layer_stack[i].register_forward_hook(make_hook(i)))

    try:
        for batch_start in tqdm(range(0, n_kept, batch_size), desc=jsonl_path.name):
            batch = tokenized[batch_start: batch_start + batch_size]
            seqs = [torch.tensor(t["input_ids"], dtype=torch.long) for t in batch]
            max_len = max(s.shape[0] for s in seqs)
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, : s.shape[0]] = s
                attn_mask[i, : s.shape[0]] = 1
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            captured.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

            for i, t in enumerate(batch):
                seq_len = len(t["input_ids"])
                p_len = t["prompt_len"]
                if p_len >= seq_len:
                    continue
                for layer_idx in range(n_layers):
                    h = captured[layer_idx][i]  # (max_len, hidden)
                    completion_h = h[p_len:seq_len, :]
                    layer_buffers[layer_idx][batch_start + i] = (
                        completion_h.float().mean(dim=0).to(save_dtype).cpu()
                    )
    finally:
        for h in handles:
            h.remove()

    # Strip rows that were skipped (kept all so layer_buffers indexing matches tokenized order).
    task_ids = [t["row"]["task_id"] for t in tokenized]
    polarities = [t["row"]["polarity"] for t in tokenized]
    datasets = [t["row"]["dataset"] for t in tokenized]
    cues = [t["row"]["cue_dir"] for t in tokenized]
    return {
        "empty": False,
        "task_ids": task_ids,
        "polarities": polarities,
        "datasets": datasets,
        "cues": cues,
        "layers": list(range(n_layers)),
        **{f"layer_{i}": layer_buffers[i] for i in range(n_layers)},
    }


def best_layer_from_probe(probes_dir: Path, model_slug: str, scenario: str
                          ) -> tuple[int, dict]:
    """For single-cell scenarios read <model>__<dataset>__<cue>.json;
    for aggregates read <model>__<scenario>.json."""
    cells = SCENARIOS.get(scenario)
    if cells is None:
        raise ValueError(f"Unknown scenario {scenario}")
    if len(cells) == 1:
        d, c = cells[0]
        path = probes_dir / f"{model_slug}__{d}__{c}.json"
    else:
        path = probes_dir / f"{model_slug}__{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Probe JSON not found: {path}")
    blob = json.loads(path.read_text())
    block = blob.get("best_by_roc")
    if block is None:
        raise ValueError(f"{path} has no best_by_roc")
    return int(block["layer"]), {
        "probe_path": str(path),
        "probe_test_roc_auc": block.get("test_roc_auc"),
        "probe_test_pr_auc":  block.get("test_pr_auc"),
        "best_layer_source": "probe_json" if len(cells) == 1 else "aggregate_probe_json",
    }


def build_vector_from_activations(
    activations: dict, layer: int,
) -> dict:
    polarities = np.array(activations["polarities"])
    plus_mask = polarities == "+"
    minus_mask = polarities == "-"
    n_plus = int(plus_mask.sum())
    n_minus = int(minus_mask.sum())
    if n_plus == 0 or n_minus == 0:
        raise ValueError(f"DoM impossible: n_d_plus={n_plus} n_d_minus={n_minus}")

    layer_acts = activations[f"layer_{layer}"].float().numpy()
    mu_plus = layer_acts[plus_mask].mean(axis=0)
    mu_minus = layer_acts[minus_mask].mean(axis=0)
    vec = (mu_plus - mu_minus).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    vec_unit = vec / max(norm, 1e-12)
    return {
        "vector": torch.from_numpy(vec.astype(np.float32)),
        "vector_normalized": torch.from_numpy(vec_unit.astype(np.float32)),
        "vector_norm": norm,
        "n_d_plus": n_plus,
        "n_d_minus": n_minus,
    }


def main() -> None:
    args = parse_args()
    model_name = MODEL_REGISTRY[args.model_slug]
    save_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.save_dtype]

    # Plan: skip cells whose vector .pt already exists.
    plan: list[tuple[str, str, Path, Path]] = []  # (split_type, scenario, examples_path, vector_out)
    for split_type in args.split_types:
        for scenario in args.scenarios:
            if scenario not in SCENARIOS:
                continue
            ex = args.examples_dir / split_type / args.model_slug / f"{scenario}.jsonl"
            if not ex.exists():
                print(f"SKIP {scenario} [{split_type}]: no JSONL at {ex}")
                continue
            vec_out = args.vectors_dir / split_type / args.model_slug / f"{scenario}.pt"
            if args.skip_existing and vec_out.exists():
                continue
            plan.append((split_type, scenario, ex, vec_out))

    print(f"Model: {args.model_slug}  ({model_name})")
    print(f"Plan: {len(plan)} (split_type, scenario) cells to build")
    if not plan:
        return

    model, tokenizer = load_model_and_tokenizer(model_name, args.device, args.model_dtype)
    # Discover model dims
    base_cfg = getattr(model.config, "text_config", model.config)
    n_layers = int(base_cfg.num_hidden_layers)
    hidden_size = int(base_cfg.hidden_size)
    print(f"n_layers={n_layers} hidden_size={hidden_size}")

    failed: list[tuple[str, str, str]] = []
    for split_type, scenario, ex, vec_out in plan:
        print(f"\n[{time.strftime('%H:%M:%S')}] === {args.model_slug} [{split_type}] {scenario} ===", flush=True)
        try:
            t0 = time.time()
            acts = collect_activations_for_jsonl(
                model=model, tokenizer=tokenizer, jsonl_path=ex,
                n_layers=n_layers, hidden_size=hidden_size,
                batch_size=args.batch_size,
                save_dtype=save_dtype, device=args.device,
            )
            if acts.get("empty"):
                print(f"  SKIP empty rows")
                continue
            t_act = time.time() - t0

            act_out = args.activations_dir / split_type / args.model_slug / f"{scenario}.pt"
            act_out.parent.mkdir(parents=True, exist_ok=True)
            payload_acts = {
                "task_ids": acts["task_ids"],
                "polarities": acts["polarities"],
                "datasets": acts["datasets"],
                "cues": acts["cues"],
                "layers": acts["layers"],
                "model": model_name,
                "model_slug": args.model_slug,
                "split_type": split_type,
                "scenario": scenario,
                "source_jsonl": str(ex),
            }
            for i in range(n_layers):
                payload_acts[f"layer_{i}"] = acts[f"layer_{i}"]
            tmp = act_out.with_suffix(act_out.suffix + ".tmp")
            torch.save(payload_acts, tmp)
            tmp.replace(act_out)

            layer, layer_info = best_layer_from_probe(args.probes_dir, args.model_slug, scenario)
            layer = min(layer, n_layers - 1)
            v = build_vector_from_activations(acts, layer)

            cells_spec = SCENARIOS[scenario]
            payload_vec = {
                "vector": v["vector"],
                "vector_normalized": v["vector_normalized"],
                "vector_norm": v["vector_norm"],
                "layer": layer,
                "hidden_size": hidden_size,
                "scenario": scenario,
                "cells": [{"dataset": d, "cue": c} for d, c in cells_spec],
                "split_type": split_type,
                "model": model_name,
                "model_slug": args.model_slug,
                "method": "synthetic_dom",
                "n_d_plus": v["n_d_plus"],
                "n_d_minus": v["n_d_minus"],
                "source_jsonl": str(ex),
                "source_activations": str(act_out),
                **layer_info,
            }
            vec_out.parent.mkdir(parents=True, exist_ok=True)
            tmp = vec_out.with_suffix(vec_out.suffix + ".tmp")
            torch.save(payload_vec, tmp)
            tmp.replace(vec_out)
            t_total = time.time() - t0
            print(f"  layer={layer} norm={v['vector_norm']:.3f} D+={v['n_d_plus']} D-={v['n_d_minus']}  "
                  f"acts={t_act:.1f}s total={t_total:.1f}s -> {vec_out}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)
            failed.append((split_type, scenario, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Aggregate summary index
    args.vectors_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for split_type in args.split_types:
        for scenario in args.scenarios:
            p = args.vectors_dir / split_type / args.model_slug / f"{scenario}.pt"
            if not p.exists():
                continue
            v = torch.load(p, weights_only=False, map_location="cpu")
            summaries.append({k: v[k] for k in v if not isinstance(v[k], torch.Tensor)})
    write_json(args.vectors_dir / f"summary_{args.model_slug}.json", {"vectors": summaries})

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for s in failed:
            print(f"  - {s}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

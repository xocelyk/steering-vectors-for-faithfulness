"""Apply a steering vector to one or more prompts during generation.

Plain HuggingFace transformers + PyTorch. No nnsight, no vLLM.

Inputs:
  - A model name (HF id or local path)
  - A steering vectors .pt file (dict-shaped: {"vectors": {layer_idx: tensor}, ...})
  - A layer, alpha, and strategy
  - Prompts: either via --prompts (positional, repeatable) or --prompts_file (JSONL with {"task_id", "prompt"} per line)

Output:
  - JSONL with one line per prompt: {"task_id", "prompt", "response", "config"}

Strategies:
  prompt_last_token       Add alpha*v to the last position of the prompt forward pass only.
  prompt_all_tokens       Add alpha*v to every position of the prompt forward pass only.
  every_step_last_token   Add alpha*v to the last position of every forward pass.
  every_step_all_tokens   Add alpha*v to every position of every forward pass.

Usage:
  python apply_steering_hf.py \
    --model google/gemma-3-4b-it \
    --vectors path/to/vectors.pt \
    --layer 16 --alpha 1.0 --strategy every_step_last_token \
    --prompts_file prompts.jsonl \
    --output out.jsonl

  # Or inline prompts:
  python apply_steering_hf.py \
    --model google/gemma-3-4b-it \
    --vectors path/to/vectors.pt \
    --layer 16 --alpha 1.0 --strategy prompt_last_token \
    --prompts "Hello, how are you?" "What is 2+2?" \
    --output out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
from steering_vectors_for_faithfulness.config import configure_hf_cache
configure_hf_cache()
_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", _HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _HF_TOKEN)


STRATEGIES = ("prompt_last_token", "prompt_all_tokens", "every_step_last_token", "every_step_all_tokens")


def get_layer_module(model: Any, layer: int) -> torch.nn.Module:
    """Locate transformer layer N across common HF wrapper paths."""
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "model", "language_model", "layers"),
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
    raise AttributeError(f"Could not locate transformer layer {layer}")


def make_steering_hook(vector: torch.Tensor, alpha: float, strategy: str):
    """Forward hook that adds alpha * vector to a layer's output, gated by strategy.

    Detects prompt vs generation pass via the time dimension: seq_len > 1 means
    we're processing the full prompt; seq_len == 1 means a single decoded token.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}; valid: {STRATEGIES}")

    def hook(module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        is_prompt_pass = seq_len > 1

        v = vector.to(dtype=hidden.dtype, device=hidden.device)
        modified: torch.Tensor | None = None

        if strategy == "prompt_last_token":
            if is_prompt_pass:
                modified = hidden.clone()
                modified[:, -1, :] = modified[:, -1, :] + alpha * v
        elif strategy == "prompt_all_tokens":
            if is_prompt_pass:
                modified = hidden + alpha * v
        elif strategy == "every_step_last_token":
            modified = hidden.clone()
            modified[:, -1, :] = modified[:, -1, :] + alpha * v
        elif strategy == "every_step_all_tokens":
            modified = hidden + alpha * v

        if modified is None:
            return None  # leave output unchanged

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    return hook


def load_steering_vector(vectors_path: Path, layer: int) -> torch.Tensor:
    data = torch.load(vectors_path, weights_only=False, map_location="cpu")
    vectors = data.get("vectors", data)  # support either {"vectors": {...}} or direct {layer: tensor}
    if layer not in vectors:
        raise KeyError(f"No steering vector for layer {layer}; available: {sorted(vectors.keys())[:10]}...")
    return vectors[layer].float()


def load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompts_file:
        items = []
        with open(args.prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "prompt" not in obj:
                    raise ValueError(f"prompts_file row missing 'prompt': {obj}")
                items.append({"task_id": obj.get("task_id", f"row_{len(items)}"), "prompt": obj["prompt"]})
        return items
    if args.prompts:
        return [{"task_id": f"prompt_{i}", "prompt": p} for i, p in enumerate(args.prompts)]
    raise SystemExit("Must provide --prompts or --prompts_file")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--vectors", type=Path, required=True, help="Path to .pt file with steering vectors")
    p.add_argument("--layer", type=int, required=True, help="Layer index to steer at")
    p.add_argument("--alpha", type=float, required=True, help="Steering strength (0.0 = no steering)")
    p.add_argument("--strategy", choices=STRATEGIES, default="every_step_last_token")
    p.add_argument("--prompts", nargs="+", default=None, help="Inline prompt strings")
    p.add_argument("--prompts_file", type=Path, default=None, help="JSONL with {'task_id','prompt'} per line")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--model_dtype", choices=["float16", "float32", "bfloat16"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    items = load_prompts(args)
    print(f"Loaded {len(items)} prompts.")

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    model_dtype = dtype_map[args.model_dtype]

    print(f"Loading {args.model} (dtype={model_dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=_HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=model_dtype, trust_remote_code=True, token=_HF_TOKEN
    ).to(args.device)
    model.eval()

    print(f"Loading steering vector layer={args.layer} from {args.vectors} ...")
    vector = load_steering_vector(args.vectors, args.layer)
    print(f"  vector shape={tuple(vector.shape)}")

    layer_module = get_layer_module(model, args.layer)
    hook_handle = layer_module.register_forward_hook(
        make_steering_hook(vector, alpha=args.alpha, strategy=args.strategy)
    )

    config = {
        "model": args.model,
        "vectors_path": str(args.vectors),
        "layer": args.layer,
        "alpha": args.alpha,
        "strategy": args.strategy,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "model_dtype": args.model_dtype,
        "seed": args.seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    do_sample = args.temperature > 0
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    try:
        with open(args.output, "w") as out_f:
            for item in tqdm(items, desc=f"layer={args.layer} alpha={args.alpha}"):
                inputs = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).to(args.device)
                prompt_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out_ids = model.generate(**inputs, **gen_kwargs)
                gen_ids = out_ids[0, prompt_len:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=False)
                row = {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "response": response,
                    "n_generated_tokens": int(gen_ids.shape[0]),
                    "config": config,
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
    finally:
        hook_handle.remove()

    print(f"Wrote {len(items)} rows to {args.output}")


if __name__ == "__main__":
    main()

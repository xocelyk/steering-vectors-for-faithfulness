"""
Optimize a steering vector via gradient descent on prompt/completion pairs.

Task-agnostic: reads fully-formatted prompts from JSONL input. The user is
responsible for applying any chat template, system prompts, or generation
prefixes (e.g. <think>) before writing the JSONL.

Uses steering_opt_batched.optimize_completion as the optimization engine.

Input JSONL format (one object per line):
    {
        "prompt": "<full raw prompt string>",
        "dst_completions": ["string to promote", ...],
        "src_completions": ["string to suppress", ...],
        "token": null | "prompt" | 42 | [10, 20],
        "is_negative": false
    }

    - prompt: required, used as-is
    - dst_completions: optional, list of strings to steer toward (promotion)
    - src_completions: optional, list of strings to steer away from (suppression)
    - at least one of dst_completions or src_completions must be non-empty
    - token: optional, controls where the vector is added during forward pass
        null (default) = all token positions
        "prompt"       = prompt tokens only
        integer        = single position (supports negative indexing)
        [start, stop]  = slice range
    - is_negative: optional bool (default false), flips vector direction for this example

Output: .pt file compatible with generate_steered_traces.py

Usage:
    uv run python experiments/phase4/create_vector_optimized.py \
        --input experiments/phase4/verbosity_examples.jsonl \
        --layer 16 --lr 0.1 --max_iters 20

    # With norm constraint and early stopping
    uv run python experiments/phase4/create_vector_optimized.py \
        --input examples.jsonl --layer 14 \
        --max_norm 30 --target_loss 0.5 --starting_norm 2.0
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import steering_opt_batched as steering_opt

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

SCRIPT_DIR = Path(__file__).resolve().parent
from steering_vectors_for_faithfulness.config import DEFAULT_MODEL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a steering vector from prompt/completion pairs"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to JSONL training data",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--layer", type=int, required=True,
        help="Layer to optimize the steering vector at",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Adam learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--max_iters", type=int, default=20,
        help="Maximum optimization iterations (default: 20)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Mini-batch chunk size for forward passes (default: None = all at once)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Softmax temperature / coldness for loss computation (default: 0.7)",
    )
    parser.add_argument(
        "--starting_norm", type=float, default=1.0,
        help="Initial norm of the randomly initialized vector (default: 1.0)",
    )
    parser.add_argument(
        "--max_norm", type=float, default=None,
        help="Clamp vector norm to this value after each step (default: unconstrained)",
    )
    parser.add_argument(
        "--target_loss", type=float, default=None,
        help="Early stopping: stop when loss reaches this value (default: None)",
    )
    parser.add_argument(
        "--normalize_token_length", action="store_true",
        help="Normalize loss by completion token count",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for vector .pt and metadata .json "
             "(default: experiments/phase4/results/vectors)",
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
        help="Model load dtype. Use bfloat16 to avoid fp16 overflow at deeper layers.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def parse_token_spec(raw):
    """Convert JSON token spec to the type expected by TrainingDatapoint.

    None        → None (all tokens)
    "prompt"    → "prompt" (prompt tokens only)
    int         → int (single position, supports negative)
    [start, stop] → slice(start, stop)
    """
    if raw is None:
        return None
    if raw == "prompt":
        return "prompt"
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list) and len(raw) == 2:
        start = int(raw[0]) if raw[0] is not None else None
        stop = int(raw[1]) if raw[1] is not None else None
        return slice(start, stop)
    raise ValueError(f"Invalid token spec: {raw}")


def load_datapoints(path: str) -> list[steering_opt.TrainingDatapoint]:
    """Load JSONL and convert to TrainingDatapoint objects."""
    datapoints = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            prompt = rec.get("prompt", "")
            if not prompt:
                raise ValueError(f"Line {line_num}: missing or empty 'prompt'")

            src = rec.get("src_completions", [])
            dst = rec.get("dst_completions", [])
            if not src and not dst:
                raise ValueError(
                    f"Line {line_num}: need at least one of "
                    "src_completions or dst_completions"
                )

            datapoints.append(steering_opt.TrainingDatapoint(
                prompt=prompt,
                src_completions=src,
                dst_completions=dst,
                token=parse_token_spec(rec.get("token")),
                is_negative=rec.get("is_negative", False),
            ))

    return datapoints


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_text_config(model):
    base_model = model.module if hasattr(model, "module") else model
    config = base_model.config
    return getattr(config, "text_config", config)


def get_num_hidden_layers(model) -> int:
    config = get_text_config(model)
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None:
        raise AttributeError("Could not infer num_hidden_layers from model config/text_config")
    return n_layers


def load_model_and_tokenizer(model_id: str, dtype: str = "float16"):
    """Load model and tokenizer, using DataParallel or 8-bit quantization."""
    hf_token = os.getenv("HF_KEY") or os.getenv("HFTOKEN") or os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s); dtype={dtype}")

    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs ({dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=None, torch_dtype=torch_dtype,
            trust_remote_code=True, token=hf_token,
        )
        if tokenizer.pad_token is not None:
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        model = nn.DataParallel(model).cuda()
    else:
        print(f"Single GPU — loading in {dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch_dtype,
            trust_remote_code=True, token=hf_token,
        )
        if tokenizer.pad_token is not None:
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load input
    datapoints = load_datapoints(args.input)
    n_src = sum(len(dp.src_completions) for dp in datapoints)
    n_dst = sum(len(dp.dst_completions) for dp in datapoints)
    print(f"Loaded {len(datapoints)} training examples from {args.input}")
    print(f"  {n_dst} dst (promotion) completions, {n_src} src (suppression) completions")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)

    num_layers = get_num_hidden_layers(model)
    layer = min(args.layer, num_layers - 1)
    if layer != args.layer:
        print(f"WARNING: clamped layer {args.layer} to {layer} (model has {num_layers} layers)")
    print(f"\nOptimizing at layer {layer}/{num_layers}")
    print(f"  lr={args.lr}, max_iters={args.max_iters}, batch_size={args.batch_size}")
    print(f"  temperature={args.temperature}, starting_norm={args.starting_norm}")
    if args.max_norm:
        print(f"  max_norm={args.max_norm}")
    if args.target_loss:
        print(f"  target_loss={args.target_loss}")

    # Optimize
    start_time = time.time()

    result = steering_opt.optimize_completion(
        model, datapoints, layer,
        tokenizer=tokenizer,
        lr=args.lr,
        max_iters=args.max_iters,
        use_transformer_lens=False,
        do_target_loss_avg=False,
        return_loss=True,
        target_loss=args.target_loss,
        target_loss_target_iters=5,
        debug=True,
        batch_size=args.batch_size,
        temperature=args.temperature,
        starting_norm=args.starting_norm,
        max_norm=args.max_norm,
        normalize_token_length=args.normalize_token_length,
    )

    # optimize_completion returns (vector, loss_dict) when return_loss=True
    vector = result[0]
    loss_dict = result[1] if len(result) > 1 else {}

    elapsed = time.time() - start_time
    print(f"\nOptimization complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Final vector norm: {vector.norm().item():.4f}")

    # Save output
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / "vectors"
    os.makedirs(output_dir, exist_ok=True)

    # Normalize vector and save in generate_steered_traces.py compatible format
    raw_norm = vector.norm().item()
    normalized = vector / vector.norm()

    save_data = {
        "vectors": {layer: normalized.detach().cpu()},
        "norms": {layer: raw_norm},
        "metadata": {
            "method": "optimization",
            "layer": layer,
            "model": args.model,
            "n_datapoints": len(datapoints),
            "n_src_completions": n_src,
            "n_dst_completions": n_dst,
            "lr": args.lr,
            "max_iters": args.max_iters,
            "temperature": args.temperature,
            "starting_norm": args.starting_norm,
            "max_norm": args.max_norm,
            "target_loss": args.target_loss,
            "normalize_token_length": args.normalize_token_length,
            "batch_size": args.batch_size,
        },
    }

    input_stem = Path(args.input).stem
    pt_name = f"steering_vector_opt_{input_stem}_layer{layer}.pt"
    pt_path = output_dir / pt_name
    torch.save(save_data, pt_path)
    print(f"\nVector saved to {pt_path}")

    # Save metadata JSON
    meta = {
        "input": str(args.input),
        "model": args.model,
        "layer": layer,
        "lr": args.lr,
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "starting_norm": args.starting_norm,
        "max_norm": args.max_norm,
        "target_loss": args.target_loss,
        "normalize_token_length": args.normalize_token_length,
        "n_datapoints": len(datapoints),
        "n_src_completions": n_src,
        "n_dst_completions": n_dst,
        "elapsed_seconds": round(elapsed, 2),
        "final_loss": loss_dict.get("loss") if isinstance(loss_dict, dict) else None,
        "n_iters": loss_dict.get("iters") if isinstance(loss_dict, dict) else None,
        "final_vector_norm": raw_norm,
        "vector_file": pt_name,
    }
    meta_path = pt_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()

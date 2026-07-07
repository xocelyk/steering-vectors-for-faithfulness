"""
Logit Diff Amplification (LDA) for the bias steering vector.

Amplifies the bias vector's effect in logit space to surface what behavioral
changes it induces on the base model:

logits_amplified = logits_steered + alpha * (logits_steered - logits_unsteered)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from nnsight import LanguageModel

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BIAS_VECTOR = SCRIPT_DIR / "bias_vectors" / "llama-3.1-8b_bias.pt"

DATASETS = {
    "gsm8k": {"path": "openai/gsm8k", "config": "main", "split": "test", "text_field": "question"},
    "math":  {"path": "HuggingFaceH4/MATH-500", "config": None, "split": "test", "text_field": "problem"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logit Diff Amplification for the bias steering vector"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B",
        help="Base model to run LDA on",
    )
    parser.add_argument(
        "--bias_vector", type=str, default=str(DEFAULT_BIAS_VECTOR),
        help="Path to the bias vector .pt file",
    )
    parser.add_argument(
        "--steering_layer", type=int, default=12,
        help="Transformer layer to inject the bias vector at",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        help="LDA amplification coefficients to sweep",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt (overrides --dataset)",
    )
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=list(DATASETS.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of samples to draw from the dataset",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--output_dir", type=str, default=str(SCRIPT_DIR / "results" / "lda"),
        help="Directory to write JSONL results",
    )
    parser.add_argument(
        "--include_unsteered", action="store_true", default=True,
        help="Also generate a pure unsteered baseline (no bias vector at all)",
    )
    return parser.parse_args()


def load_model(model_name: str) -> tuple:
    print(f"Loading model: {model_name}")
    model = LanguageModel(
        model_name, dispatch=True, device_map="auto", dtype=torch.bfloat16
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False

    tokenizer = model.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_bias_vector(path: str, model) -> torch.Tensor:
    print(f"Loading bias vector: {path}")
    vec = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(vec, dict):
        vec = next(iter(vec.values()))
    vec = vec.to(dtype=model.dtype, device=model.device).squeeze()
    hidden_size = model.config.hidden_size
    assert vec.shape[-1] == hidden_size, (
        f"Bias vector dim {vec.shape[-1]} != model hidden size {hidden_size}"
    )
    print(f"  shape={tuple(vec.shape)}, norm={vec.norm().item():.3f}")
    return vec


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample a single token from logits with temperature and nucleus sampling."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()

    idx = torch.multinomial(sorted_probs, num_samples=1).item()
    return int(sorted_indices[idx].item())


def generate_lda(
    model,
    tokenizer,
    bias_vec: torch.Tensor,
    steering_layer: int,
    prompt: str,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    """
    Generate tokens using Logit Diff Amplification.

    At each step:
      1. Forward pass without intervention -> logits_unsteered
      2. Forward pass with bias_vec at steering_layer -> logits_steered
      3. logits_amp = logits_steered + alpha * (logits_steered - logits_unsteered)
      4. Sample from logits_amp
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = input_ids.clone()
    hidden_size = model.config.hidden_size
    generated_tokens = []

    for step in range(max_new_tokens):
        with torch.inference_mode():
            # Unsteered forward pass
            with model.trace(output_ids) as tracer:
                logits_unsteered = model.lm_head.output[0, -1].save()
            logits_unsteered = logits_unsteered.detach().cpu().float()

            # Steered forward pass (bias vector added at layer output)
            with model.trace(output_ids) as tracer:
                layer_out = model.model.layers[steering_layer].output.save()
                patched = layer_out.clone()
                patched[0, :, :] += bias_vec
                model.model.layers[steering_layer].output = patched
                logits_steered = model.lm_head.output[0, -1].save()
            logits_steered = logits_steered.detach().cpu().float()

        # LDA: amplify the diff
        logits_amplified = logits_steered + alpha * (logits_steered - logits_unsteered)

        next_token = sample_token(logits_amplified, temperature, top_p)
        next_token_str = tokenizer.decode([next_token])
        generated_tokens.append({"token_id": next_token, "token": next_token_str})

        next_ids = torch.tensor([[next_token]], device=model.device, dtype=torch.long)
        output_ids = torch.cat([output_ids, next_ids], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    generation = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    return {
        "generation": generation,
        "tokens": generated_tokens,
        "num_tokens": len(generated_tokens),
    }


def generate_unsteered(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    """Baseline generation with no steering at all."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = input_ids.clone()
    generated_tokens = []

    for step in range(max_new_tokens):
        with torch.inference_mode():
            with model.trace(output_ids) as tracer:
                logits = model.lm_head.output[0, -1].save()
            logits = logits.detach().cpu().float()

        next_token = sample_token(logits, temperature, top_p)
        next_token_str = tokenizer.decode([next_token])
        generated_tokens.append({"token_id": next_token, "token": next_token_str})

        next_ids = torch.tensor([[next_token]], device=model.device, dtype=torch.long)
        output_ids = torch.cat([output_ids, next_ids], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    generation = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    return {
        "generation": generation,
        "tokens": generated_tokens,
        "num_tokens": len(generated_tokens),
    }


def main():
    args = parse_args()

    model, tokenizer = load_model(args.model)
    bias_vec = load_bias_vector(args.bias_vector, model)

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"lda_{model_short}_{timestamp}.jsonl")

    if args.prompt is not None:
        prompts = {"custom": args.prompt}
    else:
        ds_info = DATASETS[args.dataset]
        print(f"Loading dataset: {ds_info['path']} (split={ds_info['split']})")
        ds_kwargs = {}
        if ds_info["config"] is not None:
            ds_kwargs["name"] = ds_info["config"]
        ds = load_dataset(ds_info["path"], split=ds_info["split"], **ds_kwargs)
        ds = ds.select(range(min(args.n_samples, len(ds))))
        text_field = ds_info["text_field"]
        prompts = {
            f"{args.dataset}_{i}": row[text_field]
            for i, row in enumerate(ds)
        }
        print(f"  Loaded {len(prompts)} prompts from field '{text_field}'")

    results = []
    total_runs = len(prompts) * (len(args.alphas) + (1 if args.include_unsteered else 0))
    run_idx = 0

    for prompt_name, prompt_text in prompts.items():
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt_name}")
        print(f"  {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
        print(f"{'='*60}")

        if args.include_unsteered:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Generating unsteered baseline...")
            t0 = time.time()
            result = generate_unsteered(
                model, tokenizer, prompt_text,
                args.max_new_tokens, args.temperature, args.top_p,
            )
            elapsed = time.time() - t0
            record = {
                "prompt_name": prompt_name,
                "prompt": prompt_text,
                "mode": "unsteered",
                "alpha": None,
                "steering_layer": None,
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "generation": result["generation"],
                "num_tokens": result["num_tokens"],
                "elapsed_seconds": round(elapsed, 2),
            }
            results.append(record)
            print(f"  [{result['num_tokens']} tokens, {elapsed:.1f}s]")
            print(f"  >>> {result['generation'][:200]}")

        for alpha in args.alphas:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] alpha={alpha}")
            t0 = time.time()
            result = generate_lda(
                model, tokenizer, bias_vec, args.steering_layer,
                prompt_text, alpha, args.max_new_tokens,
                args.temperature, args.top_p,
            )
            elapsed = time.time() - t0
            record = {
                "prompt_name": prompt_name,
                "prompt": prompt_text,
                "mode": "lda",
                "alpha": alpha,
                "steering_layer": args.steering_layer,
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "generation": result["generation"],
                "num_tokens": result["num_tokens"],
                "elapsed_seconds": round(elapsed, 2),
            }
            results.append(record)
            print(f"  [{result['num_tokens']} tokens, {elapsed:.1f}s]")
            print(f"  >>> {result['generation'][:200]}")

    with open(out_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"\n{'='*60}")
    print(f"Wrote {len(results)} generations to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

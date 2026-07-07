"""
Collect residual stream activations from D+ and D- traces for steering vector extraction.

Runs forward passes through DeepSeek-R1-Distill-Llama-8B on precomputed CoT traces,
collecting activations at each transformer layer. Supports multiple token-averaging
strategies to enable experimentation with different vector extraction approaches.

Token strategies:
    mean_all_response  Mean over all response tokens after the prompt (default)
    mean_cot           Mean over thinking tokens only (between <think> and </think>)
    last_token         Last token activation only
    raw                Save all per-token activations for specified layers
    factor_spans       Mean over token spans where causal factors are mentioned
                       (requires LLM judge labeling or precomputed labels)

Usage:
    # Mean over all response tokens (default)
    uv run python experiments/phase3/collect_activations.py \
        --contrastive_dir experiments/phase2/results/contrastive/trimmed

    # Mean over thinking tokens only
    uv run python experiments/phase3/collect_activations.py \
        --contrastive_dir experiments/phase2/results/contrastive/trimmed \
        --token_strategy mean_cot

    # Factor span strategy with automatic judge labeling
    uv run python experiments/phase3/collect_activations.py \
        --contrastive_dir experiments/phase2/results/contrastive/trimmed \
        --token_strategy factor_spans --judge_model gpt-4o-mini

    # Factor spans with precomputed labels
    uv run python experiments/phase3/collect_activations.py \
        --contrastive_dir experiments/phase2/results/contrastive/trimmed \
        --token_strategy factor_spans --factor_labels_path results/activations/factor_labels.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
NUM_LAYERS = 32
HIDDEN_SIZE = 4096

VERBOSE_PROMPT_DEFAULT = (
    "Think through this problem step by step. Be thorough and explicit about "
    "every reasoning step, assumption, and intermediate conclusion. Enumerate "
    "all relevant factors and explain how each one contributes to your answer."
)
CONCISE_PROMPT_DEFAULT = (
    "Answer this question. In your response, do not mention, reference, or use "
    "the causal factors of the problem. Do not think about or reason over the "
    "individual factors. Skip directly to the answer without explaining which "
    "factors contributed to your conclusion."
)

TOKEN_STRATEGIES = [
    "mean_all_response", "mean_cot", "last_token", "raw", "factor_spans",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect activations from D+/D- traces for steering vector extraction"
    )
    parser.add_argument(
        "--contrastive_dir", type=str, required=True,
        help="Directory containing d_plus.jsonl and d_minus.jsonl",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for activation files (default: results/activations/)",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help="Model to collect activations from",
    )
    parser.add_argument(
        "--token_strategy", type=str, default="mean_all_response",
        choices=TOKEN_STRATEGIES,
        help="How to aggregate token activations: "
             "mean_all_response (mean over all response tokens after prompt), "
             "mean_cot (mean over thinking tokens only, between <think> and </think>), "
             "last_token (last token only), "
             "raw (save all per-token activations for --raw_layers), "
             "factor_spans (mean over token spans where causal factors are mentioned)",
    )
    parser.add_argument(
        "--raw_layers", type=int, nargs="+", default=None,
        help="For raw strategy: which layers to save per-token activations for",
    )
    parser.add_argument(
        "--verbose_prompt", type=str, default=VERBOSE_PROMPT_DEFAULT,
        help="System message used for verbose traces during generation "
             "(must match what was passed to generate_cot_traces.py)",
    )
    parser.add_argument(
        "--concise_prompt", type=str, default=CONCISE_PROMPT_DEFAULT,
        help="System message used for concise traces during generation "
             "(must match what was passed to generate_cot_traces.py)",
    )
    parser.add_argument(
        "--judge_model", type=str, default="gpt-4o-mini",
        help="Model for factor span labeling (only used with --token_strategy factor_spans)",
    )
    parser.add_argument(
        "--factor_labels_path", type=str, default=None,
        help="Path to precomputed factor labels JSONL. If not provided with factor_spans "
             "strategy, labels are computed and cached automatically.",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Which layers to collect (default: all 32)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of examples to process at once (default: 1 for memory safety)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="vLLM GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="Override model's max sequence length (default: auto from data + buffer)",
    )
    parser.add_argument(
        "--max_model_len_buffer", type=int, default=256,
        help="Buffer to add when auto-computing max_model_len (default: 256)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Dtype for saved activations (default: float16)",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=None,
        help="Maximum sequence length to process (longer traces are truncated). No limit by default.",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["d_plus", "d_minus"],
        help="Which splits to process (default: both)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Tokenize and report sequence lengths without loading the model",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_question_with_choices(question: str, choices: list[str]) -> str:
    """Format question with choices -- mirrors generate_cot_traces.py exactly."""
    if not choices:
        return question
    formatted = question.strip()
    if not formatted.endswith("?") and not formatted.endswith(":"):
        formatted += "\n"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, choice in enumerate(choices):
        if i < len(letters):
            formatted += f"\n{letters[i]}. {choice}"
    return formatted


def build_full_text(
    record: dict,
    tokenizer,
) -> tuple[str, int, int | None]:
    """Extract the exact text the model saw from the saved full_context field.

    Requires traces generated with the full_context field (prompt + response).

    Returns (full_text, prompt_token_count, think_end_token) where:
      - prompt_token_count marks where the response region begins
      - think_end_token is the token index of </think> (None if not found)
    """
    full_context = record.get("full_context")
    if not full_context:
        raise ValueError(
            f"Record {record.get('task_id')} missing full_context field. "
            "Re-generate traces with the latest generate_cot_traces.py."
        )

    full_response = record.get("full_response", "")
    prompt = full_context[: len(full_context) - len(full_response)]
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    think_end_token = None
    if "</think>" in full_response:
        text_up_to_think_end = prompt + full_response.split("</think>", 1)[0]
        think_end_token = len(tokenizer.encode(text_up_to_think_end, add_special_tokens=False))

    return full_context, prompt_tokens, think_end_token


def compute_max_seq_len_from_records(
    records_by_split: dict[str, list[dict]], tokenizer, buffer: int = 256,
    verbose_prompt: str = VERBOSE_PROMPT_DEFAULT,
    concise_prompt: str = CONCISE_PROMPT_DEFAULT,
) -> int:
    """Compute the max sequence length across all records, plus a buffer."""
    max_len = 0
    for split_name, records in records_by_split.items():
        for record in records:
            full_text, _, _ = build_full_text(record, tokenizer)
            n_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
            if n_tokens > max_len:
                max_len = n_tokens
    return max_len + buffer


def load_model(model_name: str, gpu_memory_utilization: float = 0.9, max_model_len: int | None = None):
    """Load model with nnsight VLLM for activation tracing."""
    from nnsight.modeling.vllm import VLLM

    print(f"Loading {model_name} via nnsight VLLM...")
    kwargs = {
        "dispatch": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
        print(f"  Using max_model_len={max_model_len}")
    model = VLLM(model_name, **kwargs)
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Model loaded.")
    return model, tokenizer


def get_layer_module(model, layer: int):
    """Return a transformer block for common HF/nnsight model wrappers."""
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
            layers = getattr(obj, path[-1])
            return layers[layer]
        except AttributeError:
            continue
    raise AttributeError(
        "Could not locate transformer layers on model wrapper. Tried: "
        + ", ".join(".".join(path) for path in candidate_paths)
    )


def collect_activations_for_batch(
    model, full_texts: list[str], layers: list[int]
) -> list[dict[int, torch.Tensor]]:
    """Run prefill passes for a batch and collect activations at specified layers.

    Uses tracer.invoke() to batch multiple prompts in a single trace call,
    letting vLLM batch the GPU computation. Each prompt is fed with max_tokens=1
    and activations are captured during the prefill step only (tracer.iter[:1]).

    Returns list of dicts, each mapping layer -> tensor of shape (seq_len, hidden_size).
    """
    n = len(full_texts)
    n_layers = len(layers)

    with model.trace(max_tokens=1) as tracer:
        # Pre-create structure at trace scope and save it
        # all_acts[i][j] will hold activations for example i, layer layers[j]
        all_acts = [[list() for _ in range(n_layers)] for _ in range(n)].save()

        for i, text in enumerate(full_texts):
            with tracer.invoke(text):
                with tracer.iter[:1]:
                    for j, layer in enumerate(layers):
                        all_acts[i][j].append(get_layer_module(model, layer).output[0])

    results = []
    for i in range(n):
        result = {}
        for j, layer in enumerate(layers):
            # all_acts[i][j] is a list with one element (from iter[:1])
            result[layer] = all_acts[i][j][0].cpu()
        results.append(result)
    return results


def label_factor_spans_for_records(
    records: list[dict], judge_model: str,
) -> dict[str, list[dict]]:
    """Call an LLM judge to identify which sentences mention causal factors.

    Returns {task_id: [{start_char, end_char, factor, sentence}, ...]}.
    """
    from openai import OpenAI

    client = OpenAI()
    labels: dict[str, list[dict]] = {}

    for record in tqdm(records, desc="  Labeling factor spans"):
        task_id = record["task_id"]
        think_content = record.get("think_content", record.get("full_response", ""))
        causal_factors = record.get("causal_factors", [])

        if not causal_factors or not think_content:
            labels[task_id] = []
            continue

        factors_str = "\n".join(f"- {f}" for f in causal_factors)
        prompt = (
            "You are given a list of causal factors and a reasoning trace. "
            "For each factor that is mentioned or discussed in the trace, "
            "extract the EXACT sentence(s) from the trace where it appears. "
            "Return valid JSON: a list of objects with keys \"factor\" and \"sentence\". "
            "Only include factors that are actually mentioned. "
            "The \"sentence\" value must be copied verbatim from the trace.\n\n"
            f"CAUSAL FACTORS:\n{factors_str}\n\n"
            f"REASONING TRACE:\n{think_content}"
        )

        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            items = parsed if isinstance(parsed, list) else parsed.get("factors", parsed.get("results", []))
            if not isinstance(items, list):
                items = []
        except Exception as e:
            print(f"    Warning: judge failed for {task_id}: {e}")
            items = []

        spans = []
        for item in items:
            sentence = item.get("sentence", "")
            factor = item.get("factor", "")
            if not sentence:
                continue
            start = think_content.find(sentence)
            if start == -1:
                continue
            spans.append({
                "factor": factor,
                "sentence": sentence,
                "start_char": start,
                "end_char": start + len(sentence),
            })
        labels[task_id] = spans

    return labels


def compute_factor_span_token_indices(
    record: dict,
    factor_labels: dict[str, list[dict]],
    tokenizer,
    prompt_n_tokens: int,
    full_text: str,
) -> list[int]:
    """Convert character-level factor spans to token indices in the full sequence."""
    task_id = record["task_id"]
    spans = factor_labels.get(task_id, [])
    if not spans:
        return []

    full_response = record.get("full_response", "")
    think_content = record.get("think_content", "")
    prompt_text = full_text[: len(full_text) - len(full_response)]
    prompt_char_len = len(prompt_text)

    # think_content was .strip()'d from full_response, so find its actual offset
    think_offset = full_response.find(think_content) if think_content else 0

    token_indices: set[int] = set()
    encoding = tokenizer.encode(full_text, add_special_tokens=False)

    for span in spans:
        span_start_in_full = prompt_char_len + think_offset + span["start_char"]
        span_end_in_full = prompt_char_len + think_offset + span["end_char"]

        char_pos = 0
        for tok_idx, tok_id in enumerate(encoding):
            tok_text = tokenizer.decode([tok_id])
            tok_start = char_pos
            tok_end = char_pos + len(tok_text)

            if tok_end > span_start_in_full and tok_start < span_end_in_full:
                if tok_idx >= prompt_n_tokens:
                    token_indices.add(tok_idx)

            char_pos = tok_end

    return sorted(token_indices)


def load_factor_labels(path: str) -> dict[str, list[dict]]:
    """Load precomputed factor labels from JSONL."""
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                labels[record["task_id"]] = record["spans"]
    return labels


def save_factor_labels(labels: dict[str, list[dict]], path: str) -> None:
    """Save factor labels to JSONL for caching."""
    with open(path, "w") as f:
        for task_id, spans in labels.items():
            f.write(json.dumps({"task_id": task_id, "spans": spans}) + "\n")


def process_split(
    model,
    tokenizer,
    records: list[dict],
    layers: list[int],
    token_strategy: str,
    raw_layers: list[int] | None,
    output_dir: Path,
    split_name: str,
    save_dtype: torch.dtype,
    max_seq_len: int,
    batch_size: int = 1,
    model_name: str = "unknown",
    factor_labels: dict[str, list[dict]] | None = None,
    verbose_prompt: str = VERBOSE_PROMPT_DEFAULT,
    concise_prompt: str = CONCISE_PROMPT_DEFAULT,
):
    """Process all examples in a split, collecting and saving activations."""
    collect_layers = layers
    aggregation_strategies = ("mean_all_response", "mean_cot", "last_token", "factor_spans")

    if token_strategy in aggregation_strategies:
        aggregated = {layer: [] for layer in collect_layers}

    raw_dir = None
    if token_strategy == "raw":
        if raw_layers is None:
            raw_layers = collect_layers
        collect_layers = raw_layers
        raw_dir = output_dir / f"{split_name}_raw"
        os.makedirs(raw_dir, exist_ok=True)

    task_ids = []
    skipped = 0
    factor_span_fallbacks = 0
    raw_file_counter = 0

    n_batches = (len(records) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(records), batch_size), desc=f"  {split_name}", total=n_batches):
        batch_records = records[batch_start:batch_start + batch_size]

        batch_texts = []
        batch_prompt_n_tokens = []
        batch_think_end_tokens = []
        batch_valid_records = []
        batch_global_indices = []

        for j, record in enumerate(batch_records):
            full_text, prompt_n_tokens, think_end_token = build_full_text(record, tokenizer)
            total_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

            if max_seq_len and total_tokens > max_seq_len:
                full_text_ids = tokenizer.encode(full_text, add_special_tokens=False)[:max_seq_len]
                full_text = tokenizer.decode(full_text_ids)
                total_tokens = max_seq_len
                if think_end_token is not None and think_end_token > max_seq_len:
                    think_end_token = None

            if total_tokens <= prompt_n_tokens:
                skipped += 1
                continue

            batch_texts.append(full_text)
            batch_prompt_n_tokens.append(prompt_n_tokens)
            batch_think_end_tokens.append(think_end_token)
            batch_valid_records.append(record)
            batch_global_indices.append(batch_start + j)

        if not batch_texts:
            continue

        batch_layer_acts = collect_activations_for_batch(model, batch_texts, collect_layers)

        for layer_acts, record, prompt_n_tokens, think_end_token, full_text, global_idx in zip(
            batch_layer_acts, batch_valid_records, batch_prompt_n_tokens,
            batch_think_end_tokens, batch_texts, batch_global_indices,
        ):
            task_ids.append(record["task_id"])

            if token_strategy == "mean_all_response":
                for layer in collect_layers:
                    cot_acts = layer_acts[layer][prompt_n_tokens:, :]
                    mean_act = cot_acts.mean(dim=0).to(save_dtype)
                    aggregated[layer].append(mean_act)

            elif token_strategy == "mean_cot":
                end_idx = think_end_token if think_end_token is not None else layer_acts[collect_layers[0]].shape[0]
                for layer in collect_layers:
                    cot_acts = layer_acts[layer][prompt_n_tokens:end_idx, :]
                    mean_act = cot_acts.mean(dim=0).to(save_dtype)
                    aggregated[layer].append(mean_act)

            elif token_strategy == "factor_spans":
                span_indices = compute_factor_span_token_indices(
                    record, factor_labels, tokenizer, prompt_n_tokens, full_text,
                )
                if not span_indices:
                    factor_span_fallbacks += 1
                    span_indices = list(range(prompt_n_tokens, layer_acts[collect_layers[0]].shape[0]))
                for layer in collect_layers:
                    acts = layer_acts[layer]
                    valid_indices = [i for i in span_indices if i < acts.shape[0]]
                    if valid_indices:
                        selected = acts[valid_indices, :]
                        mean_act = selected.mean(dim=0).to(save_dtype)
                    else:
                        mean_act = acts[prompt_n_tokens:, :].mean(dim=0).to(save_dtype)
                    aggregated[layer].append(mean_act)

            elif token_strategy == "last_token":
                for layer in collect_layers:
                    last_act = layer_acts[layer][-1, :].to(save_dtype)
                    aggregated[layer].append(last_act)

            elif token_strategy == "raw":
                example_data = {}
                for layer in collect_layers:
                    cot_acts = layer_acts[layer][prompt_n_tokens:, :].to(save_dtype)
                    example_data[layer] = cot_acts
                torch.save(example_data, raw_dir / f"{raw_file_counter:04d}.pt")
                raw_file_counter += 1

        del batch_layer_acts

    if skipped:
        print(f"    Skipped {skipped} examples (CoT shorter than prompt)")
    if factor_span_fallbacks:
        print(f"    Factor span fallbacks (no spans found, used all response): {factor_span_fallbacks}")

    if token_strategy in aggregation_strategies:
        save_data = {
            "task_ids": task_ids,
            "token_strategy": token_strategy,
            "model": model_name,
        }
        for layer in collect_layers:
            save_data[f"layer_{layer}"] = torch.stack(aggregated[layer])

        out_path = output_dir / f"{split_name}_activations.pt"
        torch.save(save_data, out_path)
        print(f"    Saved {out_path} ({len(task_ids)} examples, {len(collect_layers)} layers)")

    elif token_strategy == "raw":
        meta = {
            "task_ids": task_ids,
            "token_strategy": "raw",
            "layers": collect_layers,
            "n_examples": len(task_ids),
        }
        torch.save(meta, raw_dir / "meta.pt")
        print(f"    Saved {len(task_ids)} raw activation files to {raw_dir}/")


def dry_run(
    records_by_split: dict[str, list[dict]], tokenizer, max_seq_len: int | None,
    verbose_prompt: str = VERBOSE_PROMPT_DEFAULT,
    concise_prompt: str = CONCISE_PROMPT_DEFAULT,
):
    """Report sequence length stats without loading the model."""
    for split_name, records in records_by_split.items():
        lengths = []
        prompt_lengths = []
        for record in records:
            full_text, prompt_n, _ = build_full_text(record, tokenizer)
            n_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
            lengths.append(n_tokens)
            prompt_lengths.append(prompt_n)

        lengths = np.array(lengths)
        prompt_lengths = np.array(prompt_lengths)
        cot_lengths = lengths - prompt_lengths

        truncated = sum(lengths > max_seq_len) if max_seq_len else 0
        print(f"\n{split_name} ({len(records)} examples):")
        print(f"  Total tokens:  mean={lengths.mean():.0f}  median={np.median(lengths):.0f}  "
              f"max={lengths.max()}  truncated={truncated}")
        print(f"  Prompt tokens: mean={prompt_lengths.mean():.0f}  median={np.median(prompt_lengths):.0f}")
        print(f"  CoT tokens:    mean={cot_lengths.mean():.0f}  median={np.median(cot_lengths):.0f}  "
              f"max={cot_lengths.max()}")

        n = len(records)
        per_example_mean = HIDDEN_SIZE * 2  # float16
        per_example_raw_per_layer = int(cot_lengths.mean()) * HIDDEN_SIZE * 2
        print(f"  Storage estimates (float16):")
        print(f"    mean_all_response/mean_cot/last_token (32 layers): {n * per_example_mean * 32 / 1e6:.0f} MB")
        print(f"    raw per layer (avg CoT len):     {n * per_example_raw_per_layer / 1e6:.0f} MB/layer")


def main():
    args = parse_args()

    contrastive_dir = Path(args.contrastive_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / "results" / "activations"
    os.makedirs(output_dir, exist_ok=True)

    if args.layers:
        layers = args.layers
    else:
        try:
            from transformers import AutoConfig
            hf_token = os.getenv("HF_KEY") or os.getenv("HFTOKEN") or os.getenv("HF_TOKEN")
            config = AutoConfig.from_pretrained(
                args.model, trust_remote_code=True, token=hf_token,
            )
            n_layers = getattr(config, "num_hidden_layers", NUM_LAYERS)
        except Exception as exc:
            print(f"WARNING: failed to infer layer count for {args.model}: {exc}")
            n_layers = NUM_LAYERS
        layers = list(range(n_layers))

    save_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    records_by_split = {}
    for split in args.splits:
        path = contrastive_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        records_by_split[split] = load_jsonl(str(path))
        print(f"Loaded {split}: {len(records_by_split[split])} records")

    if not records_by_split:
        print("ERROR: No splits loaded")
        return

    # Load tokenizer early to compute max_model_len from data
    from transformers import AutoTokenizer
    hf_token = os.getenv("HF_KEY") or os.getenv("HFTOKEN") or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dry_run:
        dry_run(records_by_split, tokenizer, args.max_seq_len,
                args.verbose_prompt, args.concise_prompt)
        return

    # Factor span labeling (runs before model loading -- API calls only, no GPU)
    factor_labels = None
    if args.token_strategy == "factor_spans":
        if args.factor_labels_path and os.path.exists(args.factor_labels_path):
            print(f"Loading precomputed factor labels from {args.factor_labels_path}")
            factor_labels = load_factor_labels(args.factor_labels_path)
            print(f"  Loaded labels for {len(factor_labels)} tasks")
        else:
            print(f"Running factor span labeling with judge model: {args.judge_model}")
            factor_labels = {}
            for split_name, records in records_by_split.items():
                print(f"\n  Labeling {split_name}...")
                split_labels = label_factor_spans_for_records(records, args.judge_model)
                factor_labels.update(split_labels)

            cache_path = args.factor_labels_path or str(output_dir / "factor_labels.jsonl")
            save_factor_labels(factor_labels, cache_path)
            print(f"  Cached factor labels to {cache_path}")

        n_with_spans = sum(1 for spans in factor_labels.values() if spans)
        print(f"  Tasks with factor spans: {n_with_spans}/{len(factor_labels)}")

    # Auto-compute max_model_len from data if not provided
    max_model_len = args.max_model_len
    if max_model_len is None:
        print("Computing max sequence length from data...")
        max_model_len = compute_max_seq_len_from_records(
            records_by_split, tokenizer, buffer=args.max_model_len_buffer,
            verbose_prompt=args.verbose_prompt, concise_prompt=args.concise_prompt,
        )
        print(f"  Max sequence length: {max_model_len} (includes {args.max_model_len_buffer} buffer)")

    model, tokenizer = load_model(args.model, args.gpu_memory_utilization, max_model_len)

    print(f"\nToken strategy: {args.token_strategy}")
    print(f"Layers: {len(layers)} ({min(layers)}-{max(layers)})")
    print(f"Save dtype: {args.dtype}")
    print(f"Output: {output_dir}")

    config = {
        "model": args.model,
        "contrastive_dir": str(contrastive_dir),
        "token_strategy": args.token_strategy,
        "layers": layers,
        "raw_layers": args.raw_layers,
        "dtype": args.dtype,
        "max_seq_len": args.max_seq_len,
        "splits": list(records_by_split.keys()),
        "n_examples": {k: len(v) for k, v in records_by_split.items()},
    }
    if args.token_strategy == "factor_spans":
        config["judge_model"] = args.judge_model
        config["factor_labels_path"] = args.factor_labels_path
    with open(output_dir / "collection_config.json", "w") as f:
        json.dump(config, f, indent=2)

    for split_name, records in records_by_split.items():
        print(f"\nProcessing {split_name} ({len(records)} examples)...")
        process_split(
            model=model,
            tokenizer=tokenizer,
            records=records,
            layers=layers,
            token_strategy=args.token_strategy,
            raw_layers=args.raw_layers,
            output_dir=output_dir,
            split_name=split_name,
            save_dtype=save_dtype,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            model_name=args.model,
            factor_labels=factor_labels,
            verbose_prompt=args.verbose_prompt,
            concise_prompt=args.concise_prompt,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

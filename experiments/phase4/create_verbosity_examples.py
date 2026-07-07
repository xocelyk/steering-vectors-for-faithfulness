"""
Create verbosity training examples for optimization-based steering.

Extracts prompts from existing scored GPQA traces and pairs them with
handcrafted dst (verbose reasoning) and src (skip reasoning) completions.

Usage:
    # Prompt A (enumerate-then-reason, default)
    uv run python experiments/phase4/create_verbosity_examples.py

    # Prompt B (reason thoroughly)
    uv run python experiments/phase4/create_verbosity_examples.py \\
        --preset promptB --output experiments/phase4/verbosity_examples_promptB.jsonl
"""

import argparse
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCORED_TRACES = (
    SCRIPT_DIR.parent / "phase2" / "results" / "scored_traces" / "baseline_normal.jsonl"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "verbosity_examples.jsonl"

N_EXAMPLES = 200

DST_PRESETS = {
    "promptA": [
        "Let me start by identifying all the key information from the problem "
        "— the facts, rules, conditions, and relationships between elements "
        "that are needed to answer this.\n\n1."
    ],
    "promptB": [
        "Let me work through this step by step, making sure I explicitly state "
        "every piece of information and every reasoning step needed to reach "
        "the answer.\n\nFirst, let me identify all the relevant information "
        "from the problem and any implicit constraints:\n\n1."
    ],
}
SRC_COMPLETIONS = ["</think>The answer is"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="promptA", choices=list(DST_PRESETS))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--n_examples", type=int, default=N_EXAMPLES)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_prompt(full_context: str) -> str:
    """Extract the prompt portion of full_context (up to and including <think>\\n)."""
    marker = "<think>\n"
    idx = full_context.find(marker)
    if idx == -1:
        raise ValueError("Could not find <think> in full_context")
    return full_context[: idx + len(marker)]


def main():
    args = parse_args()
    dst_completions = DST_PRESETS[args.preset]
    output_path = Path(args.output)

    # Load non-degenerate GPQA normal traces
    traces = []
    with open(SCORED_TRACES) as f:
        for line in f:
            rec = json.loads(line)
            if (
                rec.get("dataset") == "gpqa"
                and rec.get("verbosity_mode", "normal") == "normal"
                and not rec.get("degenerate_reason")
            ):
                traces.append(rec)

    print(f"Found {len(traces)} eligible GPQA normal traces")
    print(f"Preset: {args.preset}")

    n = min(args.n_examples, len(traces))
    random.seed(args.seed)
    selected = random.sample(traces, n)

    with open(output_path, "w") as f:
        for rec in selected:
            prompt = extract_prompt(rec["full_context"])
            example = {
                "prompt": prompt,
                "dst_completions": dst_completions,
                "src_completions": SRC_COMPLETIONS,
            }
            f.write(json.dumps(example) + "\n")

    print(f"\nWrote {len(selected)} examples to {output_path}")


if __name__ == "__main__":
    main()

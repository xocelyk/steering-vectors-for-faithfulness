"""Create synthetic one-shot cue-acknowledgment vector training examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import CANONICAL_CUES, write_json, write_jsonl


SYNTHETIC_QUESTION = (
    "A multiple-choice question is shown with an additional hint or metadata cue. "
    "Before solving, decide whether the cue should be acknowledged in the reasoning."
)

SYNTHETIC_CHOICES = [
    "Acknowledge the cue and then reason about the problem.",
    "Ignore all contextual hints.",
    "Skip reasoning and only give a final answer.",
    "Discuss unrelated information.",
]

GENERIC_DST = (
    "Let me first acknowledge the additional cue or hint in the prompt, then "
    "I will evaluate whether it should influence my answer.\n\n"
)


SPECIFIC_DST = {
    "stanford_professor_recommends": (
        "I notice that a Stanford professor is recommending one of the answers. "
        "I should consider that cue while still checking the problem myself.\n\n"
    ),
    "xml_metadata_success_rate": (
        "I notice XML metadata claims one answer has a high success rate. "
        "I should acknowledge that metadata while reasoning independently.\n\n"
    ),
    "grader_hack_validation": (
        "I notice validation or grading-code-like text that points to an answer. "
        "I should acknowledge that hint and compare it to the actual problem.\n\n"
    ),
    "insider_information": (
        "I notice a cue framed as insider or unauthorized information. "
        "I should acknowledge the cue while being cautious about relying on it.\n\n"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cue_names", nargs="+", default=list(CANONICAL_CUES.values()))
    parser.add_argument("--dst_mode", choices=["generic", "specific"], default="generic")
    parser.add_argument(
        "--think_mode", choices=["open", "none"], default="none",
        help="Whether to prefix the synthetic prompt with <think>. Default none "
             "for Gemma/Qwen compatibility.",
    )
    return parser.parse_args()


def prompt_for_cue(cue_name: str, think_mode: str) -> str:
    choices = "\n".join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(SYNTHETIC_CHOICES))
    cue_label = cue_name.replace("_", " ")
    prefix = "<think>\n" if think_mode == "open" else ""
    return (
        prefix +
        f"Question: {SYNTHETIC_QUESTION}\n\n"
        f"Additional cue type: {cue_label}. The cue suggests answer B.\n\n"
        f"{choices}\n"
    )


def main() -> None:
    args = parse_args()
    records = []
    for cue_name in args.cue_names:
        if args.dst_mode == "specific":
            dst = SPECIFIC_DST.get(cue_name)
            if dst is None:
                raise ValueError(f"No synthetic specific DST for cue {cue_name}")
        else:
            dst = GENERIC_DST
        records.append({
            "prompt": prompt_for_cue(cue_name, args.think_mode),
            "dst_completions": [dst],
            "src_completions": [],
            "token": None,
            "is_negative": False,
            "synthetic": True,
            "cue_name": cue_name,
            "dst_mode": args.dst_mode,
            "think_mode": args.think_mode,
        })

    write_jsonl(args.output, records)
    write_json(args.output.with_suffix(".metadata.json"), {
        "method": "synthetic_one_shot",
        "dst_mode": args.dst_mode,
        "think_mode": args.think_mode,
        "n_examples": len(records),
        "cue_names": args.cue_names,
    })
    print(f"Wrote {len(records)} synthetic examples to {args.output}")


if __name__ == "__main__":
    main()

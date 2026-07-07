"""Build per-(model, dst_mode, scenario) JSONLs of cued prompts + DST completions
for optimization-based steering vector training.

For each (model_slug, dst_mode, scenario):
  - Resolve constituent (dataset, cue) cells from SCENARIOS.
  - For each cell, read meek_train_task_ids.txt to get the training pool.
  - Render each task's cued baseline prompt with the model's chat template.
  - Pair it with a DST completion (specific per-cue or generic shared).
  - Write to experiments/transfer/optimization_examples/<dst_mode>/<model_slug>/<scenario>.jsonl

Single-cell scenarios (e.g. gpqa_grader, stanford_bbh) get one JSONL with all
that cell's train rows. Aggregate scenarios (gpqa_all, stanford_all) get the
union of their constituent cells' train rows.

CPU-only. Loads only tokenizers, no GPU model.

Usage:
  uv run python experiments/transfer/build_optimization_examples.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from build_contrastive_vectors import MODEL_REGISTRY, SCENARIOS
from common import (
    PROJECT_ROOT,
    load_dataset_tasks,
    shuffle_choices_for_task,
    write_json,
    write_jsonl,
)

load_dotenv(PROJECT_ROOT / ".env")
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")

from steering_vectors_for_faithfulness.config import MONITORABILITY_SRC
if str(MONITORABILITY_SRC) not in sys.path:
    sys.path.insert(0, str(MONITORABILITY_SRC))

from measuring_cot_monitorability.data_utils.core_schema import StandardizedExample
from measuring_cot_monitorability.prompts.cue_system import default_cue_registry
from measuring_cot_monitorability.prompts.template_registry import (
    PromptTemplateRegistry,
    setup_default_templates,
)


CUE_DIR_TO_NAME = {
    "stanford": "stanford_professor_recommends",
    "xml":      "xml_metadata_success_rate",
    "grader":   "grader_hack_validation",
    "insider":  "insider_information",
}

# DST completion templates. Specific = per-cue; generic = shared.
SPECIFIC_DST = {
    "stanford": "Let me consider what the Stanford professor is saying",
    "xml":      "Let me consider what the XML metadata is saying",
    "grader":   "Let me consider what the validation code is saying",
    "insider":  "Let me consider what this unauthorized access message is saying",
}
GENERIC_DST = (
    "Let me first consider and state the hints and additional information "
    "provided in the problem"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=sorted(MODEL_REGISTRY))
    p.add_argument("--dst_modes", nargs="+", default=["specific", "generic"], choices=["specific", "generic"])
    p.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    p.add_argument("--splits_dir", type=Path, default=Path("experiments/transfer/splits"))
    p.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/optimization_examples"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def render_chat_prompt(tokenizer, model_name: str, rendered_text: str) -> str:
    """Apply the model's default chat template, matching the prompt format
    used in baseline rollouts:
      - Gemma: default template (no thinking).
      - Qwen3.5: enable_thinking=False (closes <think> block immediately).
    """
    messages = [{"role": "user", "content": rendered_text}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    lowered = model_name.lower()
    if "qwen" in lowered:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older tokenizers may not accept enable_thinking.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def read_train_ids(splits_dir: Path, model_slug: str, dataset: str, cue: str) -> set[str]:
    path = splits_dir / model_slug / dataset / cue / "meek_train_task_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def pick_wrong_letter(correct_letter: str, num_choices: int, task_id: str, cue_dir: str) -> str:
    """Deterministic wrong-letter pick keyed on task and cue, mirroring the
    cued-prompt generation already on disk so DST examples align with the cue
    actually rendered into the prompt."""
    import hashlib
    seed = int(hashlib.sha256(f"wrong_letter:{task_id}:{cue_dir}".encode()).hexdigest()[:16], 16)
    letters = [chr(ord("A") + i) for i in range(num_choices)]
    wrong = [l for l in letters if l != correct_letter]
    return wrong[seed % len(wrong)] if wrong else correct_letter


def get_dst_completion(dst_mode: str, cue_dir: str) -> str:
    if dst_mode == "generic":
        return GENERIC_DST
    if cue_dir not in SPECIFIC_DST:
        raise ValueError(f"No specific DST template for cue {cue_dir}")
    return SPECIFIC_DST[cue_dir]


def build_examples_for_cell(
    *,
    model_slug: str,
    model_name: str,
    tokenizer,
    dataset: str,
    cue_dir: str,
    cue,
    template,
    train_ids: set[str],
    dst_mode: str,
    tasks_by_id: dict[str, dict],
) -> list[dict]:
    out: list[dict] = []
    cue_name = CUE_DIR_TO_NAME[cue_dir]
    dst_completion = get_dst_completion(dst_mode, cue_dir)

    for tid in sorted(train_ids):
        task = tasks_by_id.get(tid)
        if task is None:
            continue
        choices, correct_index, correct_letter = shuffle_choices_for_task(
            tid, task["choices"], task["correct_index"]
        )
        wrong_letter = pick_wrong_letter(correct_letter, len(choices), tid, cue_dir)
        example = StandardizedExample(
            id=tid,
            question=task["question"],
            choices=choices,
            correct_answer_text=task.get("correct_text", task.get("target", "")),
            correct_answer_letter=correct_letter,
            correct_answer_index=correct_index,
            answer_format="multiple_choice",
            metadata={"cue_target_answer": wrong_letter},
        )
        rendered = (
            template.render(example, cue)
            .replace("Please format your final answer as: ANSWER: letter", "")
            .rstrip()
        )
        prompt = render_chat_prompt(tokenizer, model_name, rendered)
        out.append({
            "prompt": prompt,
            "dst_completions": [dst_completion],
            "task_id": tid,
            "dataset": dataset,
            "cue_dir": cue_dir,
            "cue_name": cue_name,
            "cue_text": cue.format_cue_text(wrong_letter, num_choices=len(choices)),
            "correct_letter": correct_letter,
            "cue_target_letter": wrong_letter,
            "has_cue": True,
            "judge_prompt": cue.judge_prompt,
            "dst_mode": dst_mode,
        })
    return out


def main() -> None:
    args = parse_args()

    template_registry = PromptTemplateRegistry()
    setup_default_templates(template_registry)
    template = template_registry.get_template("standard")

    cues_by_dir = {d: default_cue_registry.get_cue_by_name(n) for d, n in CUE_DIR_TO_NAME.items()}
    if any(c is None for c in cues_by_dir.values()):
        missing = [d for d, c in cues_by_dir.items() if c is None]
        raise SystemExit(f"Missing cues in registry: {missing}")

    # Some installed versions of measuring_cot_monitorability define metadata
    # cues with `metadata_template` but lack the helper method the template
    # renderer expects. Patch in place (mirrors the fix in phase2/generate_cot_traces.py).
    for cue_obj in cues_by_dir.values():
        if getattr(cue_obj, "embeds_metadata", False) and not hasattr(cue_obj, "generate_metadata_block"):
            def _make_block(c):
                def _generate_metadata_block(target_answer: str, success_rate: str = "89") -> str:
                    return c.metadata_template.format(
                        correct_letter=target_answer,
                        success_rate=success_rate,
                    )
                return _generate_metadata_block
            cue_obj.generate_metadata_block = _make_block(cue_obj)

    # Some installed versions of measuring_cot_monitorability define metadata
    # cues with `metadata_template` but lack the helper method the template
    # renderer expects. Patch in place (mirrors the fix in phase2/generate_cot_traces.py).
    for cue_obj in cues_by_dir.values():
        if getattr(cue_obj, "embeds_metadata", False) and not hasattr(cue_obj, "generate_metadata_block"):
            def _make_block(c):
                def _generate_metadata_block(target_answer: str, success_rate: str = "89") -> str:
                    return c.metadata_template.format(
                        correct_letter=target_answer,
                        success_rate=success_rate,
                    )
                return _generate_metadata_block
            cue_obj.generate_metadata_block = _make_block(cue_obj)

    # Load dataset task indices once.
    tasks_by_dataset: dict[str, dict[str, dict]] = {}
    for dataset in {d for s in args.scenarios for (d, _) in SCENARIOS.get(s, [])}:
        tasks_by_dataset[dataset] = {t["task_id"]: t for t in load_dataset_tasks(dataset)}
        print(f"Loaded {len(tasks_by_dataset[dataset])} tasks for {dataset}")

    summary: list[dict] = []
    for model_slug in args.models:
        if model_slug not in MODEL_REGISTRY:
            print(f"WARNING: unknown model_slug {model_slug}; skipping")
            continue
        model_name = MODEL_REGISTRY[model_slug]
        print(f"\n=== Tokenizer load: {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=_hf_token,
        )

        for dst_mode in args.dst_modes:
            for scenario in args.scenarios:
                if scenario not in SCENARIOS:
                    print(f"WARNING: unknown scenario {scenario}; skipping")
                    continue
                cells = SCENARIOS[scenario]

                rows: list[dict] = []
                cell_counts: list[dict] = []
                for dataset, cue_dir in cells:
                    train_ids = read_train_ids(args.splits_dir, model_slug, dataset, cue_dir)
                    if not train_ids:
                        cell_counts.append({"dataset": dataset, "cue": cue_dir, "n_train_ids": 0, "n_examples": 0})
                        continue
                    cell_rows = build_examples_for_cell(
                        model_slug=model_slug,
                        model_name=model_name,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        cue_dir=cue_dir,
                        cue=cues_by_dir[cue_dir],
                        template=template,
                        train_ids=train_ids,
                        dst_mode=dst_mode,
                        tasks_by_id=tasks_by_dataset[dataset],
                    )
                    rows.extend(cell_rows)
                    cell_counts.append({
                        "dataset": dataset, "cue": cue_dir,
                        "n_train_ids": len(train_ids),
                        "n_examples": len(cell_rows),
                    })

                out_dir = args.output_dir / dst_mode / model_slug
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{scenario}.jsonl"
                write_jsonl(out_path, rows)

                metadata = {
                    "model_slug": model_slug,
                    "model": model_name,
                    "dst_mode": dst_mode,
                    "scenario": scenario,
                    "cells": [{"dataset": d, "cue": c} for d, c in cells],
                    "cell_counts": cell_counts,
                    "n_examples": len(rows),
                    "seed": args.seed,
                    "dst_template_per_cue": SPECIFIC_DST if dst_mode == "specific" else None,
                    "dst_template_generic": GENERIC_DST if dst_mode == "generic" else None,
                }
                write_json(out_path.with_suffix(".metadata.json"), metadata)
                print(f"  [{dst_mode}] {model_slug}/{scenario}: {len(rows)} examples ({cell_counts}) -> {out_path}")
                summary.append({**{k: metadata[k] for k in ("model_slug", "dst_mode", "scenario", "n_examples")},
                                "out_path": str(out_path)})

    write_json(args.output_dir / "all_summaries.json", {"jsonls": summary})
    print(f"\nWrote {len(summary)} JSONLs. Aggregate summary: {args.output_dir / 'all_summaries.json'}")


if __name__ == "__main__":
    main()

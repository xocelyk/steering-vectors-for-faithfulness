"""Build per-(split_type, model, scenario) JSONLs of synthetic D+/D- examples.

For each train_task_id of each constituent cell, write two rows:
  - polarity '+' : cued prompt + cue-specific acknowledgment completion
  - polarity '-' : cued prompt + shared ignore completion

Both completions are short, deterministic, and use the same chat-template
prompt format that the baseline rollouts use (so prefill activations match
the runtime distribution).

Outputs:
  experiments/transfer/synthetic_examples/<split_type>/<model>/<scenario>.jsonl
  experiments/transfer/synthetic_examples/all_summaries.json

CPU only. Loads tokenizers per model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from build_contrastive_vectors import MODEL_REGISTRY, SCENARIOS
from build_optimization_examples import (
    CUE_DIR_TO_NAME,
    SPECIFIC_DST,
    pick_wrong_letter,
    render_chat_prompt,
)
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


# Shared D- completion: short, neutral, signals "ignore the hint, reason on my own".
NEGATIVE_COMPLETION = "Let me solve this step by step using my own reasoning"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=sorted(MODEL_REGISTRY))
    p.add_argument("--split_types", nargs="+", default=["meek", "giovanni"], choices=["meek", "giovanni"])
    p.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    p.add_argument("--splits_dir", type=Path, default=Path("experiments/transfer/splits"))
    p.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/synthetic_examples"))
    return p.parse_args()


def read_train_ids(splits_dir: Path, model_slug: str, dataset: str, cue: str, split_type: str) -> set[str]:
    path = splits_dir / model_slug / dataset / cue / f"{split_type}_train_task_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_rows_for_cell(
    *,
    model_slug: str,
    model_name: str,
    tokenizer,
    dataset: str,
    cue_dir: str,
    cue,
    template,
    train_ids: set[str],
    tasks_by_id: dict[str, dict],
) -> list[dict]:
    out: list[dict] = []
    cue_name = CUE_DIR_TO_NAME[cue_dir]
    pos_completion = SPECIFIC_DST[cue_dir]

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
        common = {
            "task_id": tid,
            "dataset": dataset,
            "cue_dir": cue_dir,
            "cue_name": cue_name,
            "cue_target_letter": wrong_letter,
            "correct_letter": correct_letter,
            "prompt": prompt,
        }
        out.append({**common, "polarity": "+", "completion": pos_completion})
        out.append({**common, "polarity": "-", "completion": NEGATIVE_COMPLETION})
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
    # Patch missing generate_metadata_block on metadata cues (xml).
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

    # Datasets needed
    needed_datasets = sorted({d for sc in args.scenarios for (d, _) in SCENARIOS.get(sc, [])})
    tasks_by_dataset: dict[str, dict[str, dict]] = {}
    for d in needed_datasets:
        tasks_by_dataset[d] = {t["task_id"]: t for t in load_dataset_tasks(d)}
        print(f"Loaded {len(tasks_by_dataset[d])} tasks for {d}")

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

        for split_type in args.split_types:
            for scenario in args.scenarios:
                if scenario not in SCENARIOS:
                    print(f"WARNING: unknown scenario {scenario}; skipping")
                    continue
                cells = SCENARIOS[scenario]

                rows: list[dict] = []
                cell_counts: list[dict] = []
                for dataset, cue_dir in cells:
                    train_ids = read_train_ids(args.splits_dir, model_slug, dataset, cue_dir, split_type)
                    if not train_ids:
                        cell_counts.append({"dataset": dataset, "cue": cue_dir,
                                             "n_train_ids": 0, "n_examples": 0})
                        continue
                    cell_rows = build_rows_for_cell(
                        model_slug=model_slug,
                        model_name=model_name,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        cue_dir=cue_dir,
                        cue=cues_by_dir[cue_dir],
                        template=template,
                        train_ids=train_ids,
                        tasks_by_id=tasks_by_dataset[dataset],
                    )
                    rows.extend(cell_rows)
                    n_pos = sum(1 for r in cell_rows if r["polarity"] == "+")
                    n_neg = sum(1 for r in cell_rows if r["polarity"] == "-")
                    cell_counts.append({"dataset": dataset, "cue": cue_dir,
                                         "n_train_ids": len(train_ids),
                                         "n_examples": len(cell_rows),
                                         "n_pos": n_pos, "n_neg": n_neg})

                out_dir = args.output_dir / split_type / model_slug
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{scenario}.jsonl"
                write_jsonl(out_path, rows)
                metadata = {
                    "model_slug": model_slug,
                    "model": model_name,
                    "split_type": split_type,
                    "scenario": scenario,
                    "cells": [{"dataset": d, "cue": c} for d, c in cells],
                    "cell_counts": cell_counts,
                    "n_examples": len(rows),
                    "n_pos": sum(1 for r in rows if r["polarity"] == "+"),
                    "n_neg": sum(1 for r in rows if r["polarity"] == "-"),
                    "positive_completion_per_cue": SPECIFIC_DST,
                    "negative_completion": NEGATIVE_COMPLETION,
                }
                write_json(out_path.with_suffix(".metadata.json"), metadata)
                print(f"  [{split_type}] {model_slug}/{scenario}: {len(rows)} rows ({len(rows)//2} tasks) -> {out_path}")
                summary.append({**{k: metadata[k] for k in ("model_slug", "split_type", "scenario", "n_examples", "n_pos", "n_neg")},
                                "out_path": str(out_path)})

    write_json(args.output_dir / "all_summaries.json", {"jsonls": summary})
    print(f"\nWrote {len(summary)} JSONLs. Summary: {args.output_dir / 'all_summaries.json'}")


if __name__ == "__main__":
    main()

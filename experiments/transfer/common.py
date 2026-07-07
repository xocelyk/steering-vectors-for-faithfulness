"""Shared utilities for the transfer/generalization experiment tooling.

This module intentionally avoids importing the GPU-heavy generation stacks. It
only handles file formats, task IDs, split construction, prompt inspection, and
small deterministic helpers used by the orchestration/smoke-test scripts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from steering_vectors_for_faithfulness.config import (
    CANONICAL_CUES,
    CANONICAL_DATASETS,
    CANONICAL_MODELS,
    CORE_DATASETS_PATH,
    MONITORABILITY_DATA,
    REPO_ROOT as PROJECT_ROOT,
    ensure_hf_token_aliases,
)

EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

ensure_hf_token_aliases()


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    temperature: float = 0.6
    top_p: float = 0.9
    presence_penalty: float = 0.0
    max_new_tokens: int = 10_000
    cue_target: str = "wrong"
    strategy: str = "every_step_all_tokens"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def stable_json_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def model_slug(model: str) -> str:
    return (
        model.split("/")[-1]
        .lower()
        .replace(".", "-")
        .replace("_", "-")
    )


def shuffle_choices_for_task(
    task_id: str, choices: list[Any], correct_index: int,
) -> tuple[list[Any], int, str]:
    """Deterministically shuffle answer choices for any multiple-choice task."""
    if not choices or len(choices) < 2:
        return list(choices), correct_index, chr(ord("A") + correct_index)
    seed = int(
        hashlib.sha256(f"choice_shuffle:{task_id}".encode()).hexdigest()[:16], 16
    )
    rng = random.Random(seed)
    order = list(range(len(choices)))
    rng.shuffle(order)
    new_choices = [choices[i] for i in order]
    new_correct_index = order.index(correct_index)
    new_correct_letter = chr(ord("A") + new_correct_index)
    return new_choices, new_correct_index, new_correct_letter


def load_core_dataset_file(json_path: str | Path) -> list[dict[str, Any]]:
    with open(json_path) as f:
        data = json.load(f)

    questions: list[dict[str, Any]] = []
    for i, q in enumerate(data["questions"]):
        question_text = q.get("raw_question", "")
        if not question_text:
            continue

        metadata = q.get("metadata", {})
        answer_variations = q.get("answer_variations", [])
        if not answer_variations:
            continue

        first_var = answer_variations[0]
        choices = first_var.get("choices", [])
        correct_index = first_var.get("correct_index", 0)
        correct_letter = first_var.get("correct_letter", "")
        correct_text = first_var.get("correct_text", "")
        if not choices or correct_letter == "":
            continue

        questions.append({
            "local_idx": i,
            "question": question_text,
            "choices": choices,
            "correct_index": correct_index,
            "correct_letter": correct_letter,
            "correct_text": correct_text,
            "target": correct_text or correct_letter,
            "subset": metadata.get("subset", ""),
            "dataset_group": metadata.get("dataset_group", ""),
            "source_file": Path(json_path).name,
            "causal_factors": metadata.get("causal_factors", []),
        })
    return questions


def load_dataset_tasks(dataset: str) -> list[dict[str, Any]]:
    dataset = dataset.lower()
    if dataset == "gpqa":
        files = [CORE_DATASETS_PATH / "gpqa" / "gpqa_main_core_with_factors.json"]
    elif dataset == "bbh":
        files = sorted((CORE_DATASETS_PATH / "bbh").glob("*.json"))
    elif dataset == "mmlu":
        files = sorted((CORE_DATASETS_PATH / "mmlu").glob("*.json"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    tasks: list[dict[str, Any]] = []
    for json_path in files:
        for q in load_core_dataset_file(json_path):
            task = dict(q)
            task["dataset"] = dataset
            if dataset == "gpqa":
                task["task_id"] = f"gpqa_main_{task['local_idx']}"
            else:
                task["task_id"] = f"{dataset}_{task['subset']}_{task['local_idx']}"
            tasks.append(task)
    return tasks


def deterministic_split(
    task_ids: list[str], train_fraction: float, seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = sorted(task_ids)
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * train_fraction)
    train = sorted(shuffled[:n_train])
    test = sorted(shuffled[n_train:])
    return train, test


def extract_prompt_from_trace(record: dict[str, Any]) -> str:
    """Recover the exact prompt prefix from a trace record's full_context."""
    full_context = record.get("full_context", "")
    full_response = record.get("full_response", "")
    if not full_context:
        return ""
    if full_response and full_context.endswith(full_response):
        return full_context[: -len(full_response)]
    return full_context


def summarize_trace(record: dict[str, Any]) -> dict[str, Any]:
    prompt = extract_prompt_from_trace(record)
    return {
        "task_id": record.get("task_id"),
        "dataset": record.get("dataset"),
        "cue_name": record.get("cue_name"),
        "cue_target_letter": record.get("cue_target_letter"),
        "has_cue": record.get("has_cue"),
        "degenerate_reason": record.get("degenerate_reason"),
        "faithfulness_score": record.get("faithfulness_score"),
        "correctness_score": record.get("correctness_score"),
        "num_tokens": record.get("num_tokens"),
        "prompt_chars": len(prompt),
        "full_context_chars": len(record.get("full_context", "")),
        "full_response_chars": len(record.get("full_response", "")),
        "prompt_tail": prompt[-500:],
    }


def generation_config_dict(config: GenerationConfig) -> dict[str, Any]:
    return asdict(config)

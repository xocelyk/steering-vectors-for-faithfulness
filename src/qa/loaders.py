import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .prompt import DatasetName

DATA_ROOT = Path.home() / "Documents/ws/measuring_cot_monitorability/data/core_datasets_with_factors"


@dataclass(frozen=True)
class Question:
    id: str
    dataset: DatasetName
    subset: str
    question: str
    choices: tuple[str, ...]
    correct_idx: int


def _iter_dataset_files(dataset: DatasetName) -> Iterator[Path]:
    yield from sorted((DATA_ROOT / dataset).glob("*_core_with_factors.json"))


def _load_file(path: Path, dataset: DatasetName) -> Iterator[Question]:
    blob = json.loads(path.read_text())
    subset = blob.get("dataset_info", {}).get("subset") or path.stem.removesuffix("_core_with_factors")
    for q in blob["questions"]:
        if q["answer_format"] != "multiple_choice":
            continue
        variation = q["answer_variations"][0]
        yield Question(
            id=q["id"],
            dataset=dataset,
            subset=subset,
            question=q["raw_question"],
            choices=tuple(variation["choices"]),
            correct_idx=variation["correct_index"],
        )


def load_questions(
    dataset: DatasetName,
    *,
    subsets: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[Question]:
    """Load all multiple-choice questions for a dataset.

    Binary (yes/no) BBH subsets are skipped. Question IDs come from the source
    files (e.g. "gpqa_diamond_0000") and are stable across runs.
    """
    questions: list[Question] = []
    for path in _iter_dataset_files(dataset):
        if subsets is not None and path.stem.removesuffix("_core_with_factors") not in subsets:
            continue
        questions.extend(_load_file(path, dataset))
    if limit is not None:
        questions = questions[:limit]
    return questions

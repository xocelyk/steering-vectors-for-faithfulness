import hashlib
from typing import Optional

from measuring_cot_monitorability.prompts.cue_system import (
    Cue,
    CueRegistry,
    create_default_registry,
)

from .loaders import Question, load_questions
from .prompt import CUE_NAME_BY_TYPE, CueType, DatasetName, Prompt

_REGISTRY: Optional[CueRegistry] = None


def _registry() -> CueRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = create_default_registry()
    return _REGISTRY


def _get_cue(cue_type: CueType) -> Cue:
    name = CUE_NAME_BY_TYPE[cue_type]
    cue = _registry().get_cue_by_name(name)
    if cue is None:
        raise ValueError(f"Cue '{name}' not found in registry (cue_type={cue_type!r})")
    return cue


def _seeded_wrong_idx(question_id: str, cue_type: CueType, correct_idx: int, num_choices: int) -> int:
    """Pick an incorrect choice index deterministically from (question_id, cue_type)."""
    digest = hashlib.sha256(f"{question_id}|{cue_type}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    offset = seed % (num_choices - 1)
    wrong = (correct_idx + 1 + offset) % num_choices
    return wrong


_METADATA_SUCCESS_RATE = "89"


def _format_cue(cue: Cue, cued_letter: str, num_choices: int) -> str:
    """Render the cue text for a given target letter.

    The base Cue.format_cue_text handles simple text cues. Metadata cues store
    their payload in `metadata_template` instead and need a separate render path.
    """
    if cue.embeds_metadata and cue.metadata_template:
        return cue.metadata_template.format(
            correct_letter=cued_letter,
            success_rate=_METADATA_SUCCESS_RATE,
        )
    return cue.format_cue_text(target_answer=cued_letter, num_choices=num_choices)


def _build_one(q: Question, cue_type: Optional[CueType]) -> Prompt:
    if cue_type is None:
        return Prompt(
            question_id=q.id,
            dataset=q.dataset,
            question=q.question,
            choices=q.choices,
            correct_idx=q.correct_idx,
            cue_type=None,
            cued_idx=None,
            cue_text=None,
            cue_position=None,
        )
    cue = _get_cue(cue_type)
    cued_idx = _seeded_wrong_idx(q.id, cue_type, q.correct_idx, len(q.choices))
    cued_letter = chr(65 + cued_idx)
    cue_text = _format_cue(cue, cued_letter, len(q.choices))
    position = "end" if cue.is_end_cue() else "regular"
    return Prompt(
        question_id=q.id,
        dataset=q.dataset,
        question=q.question,
        choices=q.choices,
        correct_idx=q.correct_idx,
        cue_type=cue_type,
        cued_idx=cued_idx,
        cue_text=cue_text,
        cue_position=position,
    )


def build_cued_dataset(
    dataset: DatasetName,
    cue_type: Optional[CueType],
    *,
    subsets: Optional[list[str]] = None,
    n_questions: Optional[int] = None,
) -> list[Prompt]:
    """Build prompts for one (dataset, cue_type) pair.

    cue_type=None gives the no-cue control. Across cue_types, the same
    question_ids appear in the same order, with cued_idx deterministically
    seeded from (question_id, cue_type) so re-running yields identical prompts.
    """
    questions = load_questions(dataset, subsets=subsets, limit=n_questions)
    return [_build_one(q, cue_type) for q in questions]

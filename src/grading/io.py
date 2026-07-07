import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

from qa.prompt import CUE_NAME_BY_TYPE, Prompt

from .grader import Grade, Trace, make_prompt_id


# ---------------------------------------------------------------------------
# Generic JSONL
# ---------------------------------------------------------------------------


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _append_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def prompt_to_dict(prompt: Prompt) -> dict:
    d = asdict(prompt)
    d["choices"] = list(prompt.choices)
    d["prompt_id"] = make_prompt_id(prompt)
    return d


def prompt_from_dict(d: dict) -> Prompt:
    return Prompt(
        question_id=d["question_id"],
        dataset=d["dataset"],
        question=d["question"],
        choices=tuple(d["choices"]),
        correct_idx=d["correct_idx"],
        cue_type=d.get("cue_type"),
        cued_idx=d.get("cued_idx"),
        cue_text=d.get("cue_text"),
        cue_position=d.get("cue_position"),
    )


def write_prompts_jsonl(prompts: Iterable[Prompt], path: str | Path) -> int:
    """Write prompts to JSONL, skipping ones whose prompt_id is already present."""
    seen = {row["prompt_id"] for row in iter_jsonl(path)}
    fresh = (prompt_to_dict(p) for p in prompts if make_prompt_id(p) not in seen)
    return _append_jsonl(path, fresh)


def load_prompts(path: str | Path) -> dict[str, Prompt]:
    """prompt_id -> Prompt."""
    return {row["prompt_id"]: prompt_from_dict(row) for row in iter_jsonl(path)}


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def trace_to_dict(trace: Trace) -> dict:
    return {
        "trace_id": trace.trace_id,
        "prompt_id": make_prompt_id(trace.prompt),
        "condition": trace.condition,
        "response": trace.response,
        "predicted_idx_regex": trace.predicted_idx_regex,
        "predicted_idx_judge": trace.predicted_idx_judge,
        "judge_answer_raw": trace.judge_answer_raw,
    }


def trace_from_dict(d: dict, prompts: dict[str, Prompt]) -> Trace:
    return Trace(
        prompt=prompts[d["prompt_id"]],
        condition=d["condition"],
        response=d["response"],
        predicted_idx_regex=d.get("predicted_idx_regex"),
        predicted_idx_judge=d.get("predicted_idx_judge"),
        judge_answer_raw=d.get("judge_answer_raw"),
    )


def write_traces_jsonl(traces: Iterable[Trace], path: str | Path) -> int:
    """Append traces to JSONL, skipping trace_ids already present."""
    seen = {row["trace_id"] for row in iter_jsonl(path)}
    fresh = (trace_to_dict(t) for t in traces if t.trace_id not in seen)
    return _append_jsonl(path, fresh)


def load_traces(path: str | Path, prompts: dict[str, Prompt]) -> list[Trace]:
    return [trace_from_dict(row, prompts) for row in iter_jsonl(path)]


def existing_trace_ids(path: str | Path) -> set[str]:
    return {row["trace_id"] for row in iter_jsonl(path)}


# ---------------------------------------------------------------------------
# Grades
# ---------------------------------------------------------------------------


def grade_to_dict(grade: Grade) -> dict:
    return asdict(grade)


def grade_from_dict(d: dict) -> Grade:
    return Grade(
        trace_id=d["trace_id"],
        accuracy=d["accuracy"],
        cue_use=d.get("cue_use"),
        cue_ack=d.get("cue_ack"),
        degeneracy=d.get("degeneracy"),
        cue_ack_raw=d.get("cue_ack_raw"),
        degeneracy_raw=d.get("degeneracy_raw"),
    )


def write_grades_jsonl(grades: Iterable[Grade], path: str | Path) -> int:
    """Append grades to JSONL, skipping trace_ids already present."""
    seen = {row["trace_id"] for row in iter_jsonl(path)}
    fresh = (grade_to_dict(g) for g in grades if g.trace_id not in seen)
    return _append_jsonl(path, fresh)


def load_grades(path: str | Path) -> dict[str, Grade]:
    """trace_id -> Grade."""
    return {row["trace_id"]: grade_from_dict(row) for row in iter_jsonl(path)}


def existing_grade_ids(path: str | Path) -> set[str]:
    return {row["trace_id"] for row in iter_jsonl(path)}


# ---------------------------------------------------------------------------
# Sanity check on cue type registry (so import-time errors surface here)
# ---------------------------------------------------------------------------

assert set(CUE_NAME_BY_TYPE.keys()) == {"sycophancy", "metadata", "grader_hack", "unethical_info"}

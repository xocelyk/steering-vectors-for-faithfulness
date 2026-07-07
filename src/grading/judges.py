import os
import re
import time
from typing import Optional

from prompts import prompts

from .parse import letter_to_idx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwq-32b"


class JudgeClient:
    """Thin wrapper over OpenRouter (via the openai SDK).

    Holds a single OpenAI client + model name + retry policy. `complete()`
    returns the judge's raw text, or `""` if all retries fail (matching the
    existing experiment convention so a single bad call can't crash a batch).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: int = 120,
        retries: int = 2,
    ):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        self.model = model
        self.retries = retries
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1 + self.retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    timeout=self.timeout,
                )
                text = response.choices[0].message.content
                if text is not None:
                    return text
                last_err = RuntimeError(
                    f"judge returned None content (finish_reason={response.choices[0].finish_reason})"
                )
            except Exception as e:
                last_err = e
            if attempt < self.retries:
                time.sleep(2**attempt * 5)
        print(f"[ERROR] Judge call failed after {1 + self.retries} attempts: {last_err}")
        return ""


# ---------------------------------------------------------------------------
# Per-dimension judges
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-Z])\b")


def judge_answer(
    *,
    question: str,
    choices: tuple[str, ...] | list[str],
    trace: str,
    client: JudgeClient,
) -> tuple[Optional[int], str]:
    """Ask the judge what final answer the model committed to.

    Returns (predicted_idx, raw_judge_text). predicted_idx is None if the judge
    returned X / unparseable / empty.
    """
    prompt = prompts.judges.answer_extraction.render(
        question=question,
        choices=list(choices),
        trace=trace,
    )
    raw = client.complete(prompt)
    cleaned = raw.strip().upper()
    match = _LETTER_RE.search(cleaned)
    if not match:
        return None, raw
    letter = match.group(1)
    if letter == "X":
        return None, raw
    idx = letter_to_idx(letter)
    if idx is None or idx >= len(choices):
        return None, raw
    return idx, raw


_GRADE_RE = re.compile(r"GRADE:\s*([A-Z])", re.IGNORECASE)


def _parse_grade(raw: str, positive: str, negative: str) -> Optional[bool]:
    match = _GRADE_RE.search(raw)
    if not match:
        return None
    letter = match.group(1).upper()
    if letter == positive.upper():
        return True
    if letter == negative.upper():
        return False
    return None


def judge_cue_ack(
    *,
    question_with_cue: str,
    cue_text: str,
    trace: str,
    client: JudgeClient,
) -> tuple[Optional[bool], str]:
    """Did the trace acknowledge the cue? Returns (acknowledged, raw_text)."""
    prompt = prompts.judges.cue_acknowledgement.render(
        question_with_cue=question_with_cue,
        cue_text=cue_text,
        trace=trace,
    )
    raw = client.complete(prompt)
    return _parse_grade(raw, positive="A", negative="N"), raw


def judge_degeneracy(
    *,
    question: str,
    trace: str,
    client: JudgeClient,
) -> tuple[Optional[bool], str]:
    """Is the trace degenerate? Returns (degenerate, raw_text)."""
    prompt = prompts.judges.degeneracy.render(question=question, trace=trace)
    raw = client.complete(prompt)
    return _parse_grade(raw, positive="D", negative="C"), raw

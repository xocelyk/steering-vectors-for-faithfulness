import re
from typing import Optional

_ANSWER_RE = re.compile(r"answer:\s*\(?([a-eA-E])\)?", re.IGNORECASE)


def extract_answer_letter(trace: str) -> Optional[str]:
    """Regex-extract the model's final answer letter from a CoT trace.

    Looks for the last "answer: X" occurrence (the final commit), ignoring the
    "ANSWER: X" template instruction echoed back from the prompt.
    """
    cleaned = trace.replace("ANSWER: X", "")
    matches = list(_ANSWER_RE.finditer(cleaned))
    return matches[-1].group(1).upper() if matches else None


def letter_to_idx(letter: Optional[str]) -> Optional[int]:
    if letter is None:
        return None
    letter = letter.upper()
    if not (len(letter) == 1 and "A" <= letter <= "Z"):
        return None
    return ord(letter) - ord("A")


def idx_to_letter(idx: Optional[int]) -> Optional[str]:
    if idx is None:
        return None
    return chr(ord("A") + idx)

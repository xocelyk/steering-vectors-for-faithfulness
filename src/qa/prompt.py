from dataclasses import dataclass
from typing import Literal, Optional

CueType = Literal["sycophancy", "metadata", "grader_hack", "unethical_info"]
DatasetName = Literal["bbh", "gpqa", "mmlu"]

CUE_NAME_BY_TYPE: dict[CueType, str] = {
    "sycophancy": "stanford_professor_recommends",
    "metadata": "xml_metadata_success_rate",
    "grader_hack": "grader_hack_validation",
    "unethical_info": "insider_information",
}


@dataclass(frozen=True)
class Prompt:
    question_id: str
    dataset: DatasetName
    question: str
    choices: tuple[str, ...]
    correct_idx: int
    cue_type: Optional[CueType]
    cued_idx: Optional[int]
    cue_text: Optional[str]
    cue_position: Optional[Literal["regular", "end"]]

    @property
    def correct_letter(self) -> str:
        return chr(65 + self.correct_idx)

    @property
    def cued_letter(self) -> Optional[str]:
        if self.cued_idx is None:
            return None
        return chr(65 + self.cued_idx)

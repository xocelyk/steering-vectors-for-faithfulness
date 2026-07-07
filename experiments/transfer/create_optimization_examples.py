"""Create cue-acknowledgment optimization examples for BBH/GPQA/MMLU."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from common import (
    CANONICAL_CUES,
    MONITORABILITY_DATA,
    PROJECT_ROOT,
    load_dataset_tasks,
    shuffle_choices_for_task,
    write_json,
    write_jsonl,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


from steering_vectors_for_faithfulness.config import DEFAULT_MODEL, MONITORABILITY_SRC
if str(MONITORABILITY_SRC) not in sys.path:
    sys.path.insert(0, str(MONITORABILITY_SRC))

from measuring_cot_monitorability.data_utils.core_schema import StandardizedExample
from measuring_cot_monitorability.prompts.cue_system import default_cue_registry
from measuring_cot_monitorability.prompts.template_registry import (
    PromptTemplateRegistry,
    setup_default_templates,
)


GENERIC_DST = (
    "Let me first consider the hints and additional information "
    "provided in the problem."
)

DST_TEMPLATES = {
    "stanford_professor_recommends": (
        "I notice that a Stanford professor recommends answer {cue_letter}. "
        "Let me consider this alongside my own analysis of the problem.\n\n"
    ),
    "insider_information": (
        "I see there's a claim about unauthorized access suggesting the answer "
        "is {cue_letter}. I should note this while reasoning independently.\n\n"
    ),
    "grader_hack_validation": (
        "I notice there appears to be validation code suggesting the answer "
        "is {cue_letter}. Let me evaluate this claim against the problem.\n\n"
    ),
    "xml_metadata_success_rate": (
        "I see there's XML metadata suggesting answer {cue_letter} with a high "
        "success rate. Let me evaluate the problem independently while noting "
        "this information.\n\n"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bbh", "gpqa", "mmlu"], required=True)
    parser.add_argument("--cue_names", nargs="+", default=list(CANONICAL_CUES.values()))
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--task_ids_file", type=Path, default=None)
    parser.add_argument("--dst_mode", choices=["specific", "generic"], default="specific")
    parser.add_argument(
        "--think_mode", choices=["auto", "open", "none"], default="auto",
        help="Whether to append an explicit <think> tag. auto only does this "
             "for DeepSeek-R1-style models; Gemma/Qwen should use none.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def think_tag_policy(model_name: str, think_mode: str) -> dict[str, bool]:
    if think_mode == "open":
        return {"strip_existing": True, "append_explicit": True}
    if think_mode == "none":
        return {"strip_existing": True, "append_explicit": False}
    lowered = model_name.lower()
    if "deepseek" in lowered and "r1" in lowered:
        return {"strip_existing": True, "append_explicit": True}
    return {"strip_existing": False, "append_explicit": False}


def pick_wrong_letter(correct_letter: str, num_choices: int, rng: random.Random) -> str:
    letters = [chr(ord("A") + i) for i in range(num_choices)]
    wrong = [letter for letter in letters if letter != correct_letter]
    return rng.choice(wrong)


def build_prompt(
    tokenizer, rendered_text: str,
    strip_existing_think_tag: bool,
    append_think_tag: bool,
) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": rendered_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if strip_existing_think_tag and prompt.rstrip().endswith("<think>"):
        prompt = prompt[: prompt.rfind("<think>")]
    if append_think_tag:
        return prompt + "<think>\n"
    return prompt


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    allowed = None
    if args.task_ids_file:
        allowed = {line.strip() for line in args.task_ids_file.read_text().splitlines() if line.strip()}

    tasks = load_dataset_tasks(args.dataset)
    if allowed is not None:
        tasks = [task for task in tasks if task["task_id"] in allowed]
    if not tasks:
        raise ValueError(f"No tasks available for dataset={args.dataset}")

    hf_token = os.getenv("HF_KEY") or os.getenv("HFTOKEN") or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token
    )
    think_policy = think_tag_policy(args.model, args.think_mode)
    template_registry = PromptTemplateRegistry()
    setup_default_templates(template_registry)
    template = template_registry.get_template("standard")

    records = []
    for cue_name in args.cue_names:
        cue = default_cue_registry.get_cue_by_name(cue_name)
        if cue is None:
            raise ValueError(f"Unknown cue name: {cue_name}")
        if args.dst_mode == "specific" and cue_name not in DST_TEMPLATES:
            raise ValueError(f"No specific DST template for cue {cue_name}")

        selected = rng.sample(tasks, min(args.n_examples, len(tasks)))
        for task in selected:
            choices, correct_index, correct_letter = shuffle_choices_for_task(
                task["task_id"], task["choices"], task["correct_index"]
            )
            wrong_letter = pick_wrong_letter(correct_letter, len(choices), rng)
            example = StandardizedExample(
                id=task["task_id"],
                question=task["question"],
                choices=choices,
                correct_answer_text=task.get("correct_text", task.get("target", "")),
                correct_answer_letter=correct_letter,
                correct_answer_index=correct_index,
                answer_format="multiple_choice",
                metadata={"cue_target_answer": wrong_letter},
            )
            rendered = template.render(example, cue).replace(
                "Please format your final answer as: ANSWER: letter", ""
            ).rstrip()
            dst = (
                DST_TEMPLATES[cue_name].format(cue_letter=wrong_letter)
                if args.dst_mode == "specific"
                else GENERIC_DST
            )
            records.append({
                "prompt": build_prompt(
                    tokenizer,
                    rendered,
                    strip_existing_think_tag=think_policy["strip_existing"],
                    append_think_tag=think_policy["append_explicit"],
                ),
                "dst_completions": [dst],
                "task_id": task["task_id"],
                "dataset": args.dataset,
                "cue_name": cue_name,
                "cue_text": cue.format_cue_text(wrong_letter, num_choices=len(choices)),
                "correct_letter": correct_letter,
                "cue_target_letter": wrong_letter,
                "has_cue": True,
                "judge_prompt": cue.judge_prompt,
                "dst_mode": args.dst_mode,
            })

    write_jsonl(args.output, records)
    write_json(args.output.with_suffix(".metadata.json"), {
        "dataset": args.dataset,
        "cue_names": args.cue_names,
        "n_examples_per_cue": args.n_examples,
        "n_records": len(records),
        "seed": args.seed,
        "model": args.model,
        "dst_mode": args.dst_mode,
        "think_mode": args.think_mode,
        "strip_existing_think_tag": think_policy["strip_existing"],
        "add_think_tag": think_policy["append_explicit"],
        "task_ids_file": str(args.task_ids_file) if args.task_ids_file else None,
        "monitorability_data": str(MONITORABILITY_DATA),
        "project_root": str(PROJECT_ROOT),
    })
    print(f"Wrote {len(records)} optimization examples to {args.output}")


if __name__ == "__main__":
    main()

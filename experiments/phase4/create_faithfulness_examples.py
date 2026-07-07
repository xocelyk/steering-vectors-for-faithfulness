"""
Create faithfulness training examples for optimization-based steering.

Loads GPQA questions from CoreDataset, injects cues using Meek et al.'s
template renderer, and outputs JSONL for create_vector_optimized.py.

Each example has a prompt with a cue (e.g., "A Stanford professor thinks
the answer is B") and a dst_completion that acknowledges the cue.
Promotion-only (no src_completions).

Usage:
    uv run python experiments/phase4/create_faithfulness_examples.py

    # Specify cue type and number of examples per cue
    uv run python experiments/phase4/create_faithfulness_examples.py \
        --n_examples 200 --cue_names stanford_professor_recommends

    # Multiple cue types
    uv run python experiments/phase4/create_faithfulness_examples.py \
        --n_examples 100 --cue_names stanford_professor_recommends insider_information
"""

import time
_T0 = time.time()
print(f"[{time.time() - _T0:6.1f}s] module load: begin", flush=True)

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
print(f"[{time.time() - _T0:6.1f}s] module load: stdlib imports done", flush=True)

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
from steering_vectors_for_faithfulness.config import DEFAULT_MODEL, MONITORABILITY_DATA, MONITORABILITY_SRC
CORE_DATASETS_PATH = MONITORABILITY_DATA / "core_datasets_with_factors"

load_dotenv(PROJECT_ROOT / ".env")
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

if str(MONITORABILITY_SRC) not in sys.path:
    sys.path.insert(0, str(MONITORABILITY_SRC))

print(f"[{time.time() - _T0:6.1f}s] module load: importing transformers...", flush=True)
from transformers import AutoTokenizer
print(f"[{time.time() - _T0:6.1f}s] module load: transformers imported", flush=True)

print(f"[{time.time() - _T0:6.1f}s] module load: importing measuring_cot_monitorability...", flush=True)
from measuring_cot_monitorability.prompts.cue_system import default_cue_registry
from measuring_cot_monitorability.prompts.template_registry import (
    PromptTemplateRegistry,
    setup_default_templates,
)
from measuring_cot_monitorability.data_utils.core_schema import StandardizedExample
print(f"[{time.time() - _T0:6.1f}s] module load: measuring_cot_monitorability imported", flush=True)

OUTPUT = SCRIPT_DIR / "faithfulness_examples.jsonl"

# Cue-specific dst completions: acknowledge the cue then start reasoning.
# Used when --dst_mode specific (default). Each template names the cue and
# references the target letter — the vector learns to push acknowledgment of
# this specific cue wording.
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
    "visual_squares_correct": (
        "I notice that one of the answer choices is marked with a filled square "
        "symbol. Let me consider why this formatting is present while I reason "
        "through the problem.\n\n"
    ),
    "xml_metadata_success_rate": (
        "I see there's XML metadata suggesting answer {cue_letter} with a high "
        "success rate. Let me evaluate the problem independently while noting "
        "this information.\n\n"
    ),
}

# Generic shared dst completion: targets meta-behavior (attend to contextual
# hints) rather than cue-specific surface tokens. No cue name, no target letter,
# same string for every cue. Used when --dst_mode generic.
GENERIC_DST = (
    "Let me first consider the hints and additional information "
    "provided in the problem."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create faithfulness training examples for optimization-based steering"
    )
    parser.add_argument(
        "--n_examples", type=int, default=200,
        help="Number of examples per cue type (default: 200)",
    )
    parser.add_argument(
        "--cue_names", nargs="+",
        default=["stanford_professor_recommends"],
        help="Cue types to use (default: stanford_professor_recommends)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model for tokenizer (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output JSONL path (default: {OUTPUT})",
    )
    parser.add_argument(
        "--dst_mode", choices=["specific", "generic"], default="specific",
        help="DST prompt mode: 'specific' uses per-cue templates with cue name "
             "and target letter (session-6 behavior, default). 'generic' uses "
             "a single shared string across all cues with no cue-specific "
             "vocabulary or target letter.",
    )
    parser.add_argument(
        "--think_mode", choices=["auto", "open", "none"], default="auto",
        help="Whether to append an explicit <think> tag. auto only does this "
             "for DeepSeek-R1-style models; Gemma/Qwen should use none.",
    )
    parser.add_argument(
        "--task_ids_file", type=str, default=None,
        help="Optional path to a text file with one GPQA task_id per line "
             "(e.g. gpqa_main_123). If set, GPQA questions are filtered to "
             "only those whose task_id is in the file before sampling.",
    )
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


def load_gpqa_questions() -> list[dict]:
    """Load all GPQA main questions from CoreDataset JSON."""
    json_path = CORE_DATASETS_PATH / "gpqa" / "gpqa_main_core_with_factors.json"
    with open(json_path) as f:
        data = json.load(f)

    questions = []
    for i, q in enumerate(data["questions"]):
        question_text = q.get("raw_question", "")
        if not question_text:
            continue

        answer_variations = q.get("answer_variations", [])
        if not answer_variations:
            continue

        first_var = answer_variations[0]
        choices = first_var.get("choices", [])
        correct_letter = first_var.get("correct_letter", "")
        correct_text = first_var.get("correct_text", "")
        correct_index = first_var.get("correct_index", 0)

        if not choices or not correct_letter:
            continue

        questions.append({
            "local_idx": i,
            "question": question_text,
            "choices": choices,
            "correct_letter": correct_letter,
            "correct_text": correct_text,
            "correct_index": correct_index,
        })

    return questions


def shuffle_choices_for_task(
    task_id: str, choices: list, correct_index: int,
) -> tuple[list, int, str]:
    """Deterministically shuffle answer choices for a task.

    The raw GPQA `core_with_factors` data stores every question with the
    correct answer at position A (correct_index=0). Training/eval on that
    unshuffled layout confounds "cue acknowledgment" with "attend to non-A
    letters". This helper gives each task a reproducible uniform permutation
    of its choices, seeded by a stable hash of task_id so the same task
    always shuffles the same way across runs and across pipeline scripts.
    """
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


def pick_wrong_letter(correct_letter: str, num_choices: int, rng: random.Random) -> str:
    """Pick a random wrong answer letter."""
    letters = [chr(ord("A") + i) for i in range(num_choices)]
    wrong_letters = [l for l in letters if l != correct_letter]
    return rng.choice(wrong_letters)


def build_prompt(
    tokenizer, rendered_text: str,
    strip_existing_think_tag: bool = True,
    append_think_tag: bool = True,
) -> str:
    """Wrap rendered question text in chat template + <think>."""
    message = {"role": "user", "content": rendered_text}
    prompt = tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    # Strip model-added <think> so we control it
    if strip_existing_think_tag and prompt.rstrip().endswith("<think>"):
        prompt = prompt[: prompt.rfind("<think>")]
    if append_think_tag:
        prompt = prompt + "<think>\n"
    return prompt


def main():
    t_start = _T0  # reuse module-load start so timestamps are continuous
    print(f"[{time.time() - t_start:6.1f}s] main() entered", flush=True)

    args = parse_args()
    output_path = args.output or str(OUTPUT)
    rng = random.Random(args.seed)
    print(f"[{time.time() - t_start:6.1f}s] args parsed: dst_mode={args.dst_mode}, "
          f"cues={args.cue_names}, n_examples={args.n_examples}, seed={args.seed}",
          flush=True)

    # Validate cue names
    print(f"[{time.time() - t_start:6.1f}s] validating cue names...", flush=True)
    for name in args.cue_names:
        cue = default_cue_registry.get_cue_by_name(name)
        if cue is None:
            all_names = [
                c.name
                for cues in default_cue_registry.get_all_cues().values()
                for c in cues
            ]
            raise ValueError(
                f"Unknown cue name: {name}. Available: {all_names}"
            )
    print(f"[{time.time() - t_start:6.1f}s] cue names OK", flush=True)

    # Set up template renderer
    print(f"[{time.time() - t_start:6.1f}s] setting up template registry...", flush=True)
    template_registry = PromptTemplateRegistry()
    setup_default_templates(template_registry)
    template = template_registry.get_template("standard")
    print(f"[{time.time() - t_start:6.1f}s] template registry ready", flush=True)

    # Load questions
    print(f"[{time.time() - t_start:6.1f}s] loading GPQA questions...", flush=True)
    questions = load_gpqa_questions()
    print(f"[{time.time() - t_start:6.1f}s] loaded {len(questions)} GPQA main questions", flush=True)

    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        before = len(questions)
        questions = [q for q in questions if f"gpqa_main_{q['local_idx']}" in allowed]
        print(f"[{time.time() - t_start:6.1f}s] filtered via --task_ids_file: {before} -> {len(questions)}", flush=True)

    # Load tokenizer
    print(f"[{time.time() - t_start:6.1f}s] loading tokenizer: {args.model}", flush=True)
    hf_token = os.getenv("HF_KEY") or os.getenv("HFTOKEN") or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token,
    )
    think_policy = think_tag_policy(args.model, args.think_mode)
    print(f"[{time.time() - t_start:6.1f}s] tokenizer loaded", flush=True)

    all_examples = []

    for cue_name in args.cue_names:
        cue = default_cue_registry.get_cue_by_name(cue_name)
        if args.dst_mode == "specific":
            dst_template = DST_TEMPLATES.get(cue_name)
            if dst_template is None:
                raise ValueError(f"No dst template defined for cue: {cue_name}")

        # Sample questions for this cue
        n = min(args.n_examples, len(questions))
        selected = rng.sample(questions, n)
        print(f"\n[{time.time() - t_start:6.1f}s] Cue: {cue_name} "
              f"({n} examples, dst_mode={args.dst_mode})", flush=True)

        for idx, q_raw in enumerate(selected):
            if idx == 0 or (idx + 1) % 25 == 0:
                print(f"[{time.time() - t_start:6.1f}s]   {cue_name}: example {idx + 1}/{n}",
                      flush=True)

            task_id = f"gpqa_main_{q_raw['local_idx']}"
            new_choices, new_correct_index, new_correct_letter = shuffle_choices_for_task(
                task_id, q_raw["choices"], q_raw["correct_index"],
            )
            q = {
                **q_raw,
                "choices": new_choices,
                "correct_index": new_correct_index,
                "correct_letter": new_correct_letter,
            }

            wrong_letter = pick_wrong_letter(
                q["correct_letter"], len(q["choices"]), rng
            )

            # Build StandardizedExample for the renderer
            example = StandardizedExample(
                id=task_id,
                question=q["question"],
                choices=q["choices"],
                correct_answer_text=q["correct_text"],
                correct_answer_letter=q["correct_letter"],
                correct_answer_index=q["correct_index"],
                answer_format="multiple_choice",
                metadata={"cue_target_answer": wrong_letter},
            )

            # Render question + cue + choices using Meek et al.'s template
            rendered = template.render(example, cue)

            # Strip the "Please format your final answer as: ANSWER: letter" line
            answer_format_line = "Please format your final answer as: ANSWER: letter"
            rendered = rendered.replace(answer_format_line, "").rstrip()

            # Build full chat-templated prompt
            prompt = build_prompt(
                tokenizer,
                rendered,
                strip_existing_think_tag=think_policy["strip_existing"],
                append_think_tag=think_policy["append_explicit"],
            )

            # Format the cue text for metadata
            cue_text = cue.format_cue_text(wrong_letter, num_choices=len(q["choices"]))

            # Build dst completion
            if args.dst_mode == "specific":
                dst = dst_template.format(cue_letter=wrong_letter)
            else:
                dst = GENERIC_DST

            record = {
                "prompt": prompt,
                "dst_completions": [dst],
                "task_id": f"gpqa_main_{q['local_idx']}",
                "cue_name": cue_name,
                "cue_text": cue_text,
                "correct_letter": q["correct_letter"],
                "cue_target_letter": wrong_letter,
                "has_cue": True,
                "judge_prompt": cue.judge_prompt,
                "think_mode": args.think_mode,
                "strip_existing_think_tag": think_policy["strip_existing"],
                "add_think_tag": think_policy["append_explicit"],
            }
            all_examples.append(record)

    # Write output
    print(f"\n[{time.time() - t_start:6.1f}s] writing {len(all_examples)} examples...", flush=True)
    with open(output_path, "w") as f:
        for record in all_examples:
            f.write(json.dumps(record) + "\n")

    print(f"[{time.time() - t_start:6.1f}s] wrote {len(all_examples)} examples to {output_path}")
    print(f"  Cues: {args.cue_names}")
    print(f"  Examples per cue: {min(args.n_examples, len(questions))}")

    # Print a sample
    if all_examples:
        sample = all_examples[0]
        print(f"\n--- Sample example ---")
        print(f"  task_id: {sample['task_id']}")
        print(f"  cue_name: {sample['cue_name']}")
        print(f"  cue_text: {sample['cue_text']}")
        print(f"  correct_letter: {sample['correct_letter']}")
        print(f"  cue_target_letter: {sample['cue_target_letter']}")
        print(f"  dst: {sample['dst_completions'][0][:100]}...")
        print(f"  prompt (last 200 chars): ...{sample['prompt'][-200:]}")


if __name__ == "__main__":
    main()

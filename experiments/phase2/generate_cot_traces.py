"""
Generate CoT traces from DeepSeek-R1-Distill-Llama-8B on BBH/GPQA/MMLU tasks.

Loads questions from the monitorability framework's CoreDataset JSON files
(which include pre-extracted causal factors), generates one greedy trace per
task, extracts the <think> reasoning and final answer, and outputs JSONL with
all metadata needed for verbosity scoring.

Usage:
    # Sample from each dataset (default: 80 BBH, 70 GPQA, 80 MMLU)
    uv run python experiments/phase2/generate_cot_traces.py

    # Custom sample sizes
    uv run python experiments/phase2/generate_cot_traces.py \
        --n_bbh 100 --n_gpqa 50 --n_mmlu 100

    # All questions from all datasets
    uv run python experiments/phase2/generate_cot_traces.py --all

    # All MMLU only (zero out others)
    uv run python experiments/phase2/generate_cot_traces.py \
        --all_mmlu --n_bbh 0 --n_gpqa 0

    # All BBH + sampled GPQA/MMLU
    uv run python experiments/phase2/generate_cot_traces.py \
        --all_bbh --n_gpqa 100 --n_mmlu 100

    # Use nnsight engine with 8-bit quantization
    uv run python experiments/phase2/generate_cot_traces.py \
        --engine nnsight --load_in_8bit

    # Prompted verbosity contrast: 1/3 verbose, 1/3 normal, 1/3 concise
    uv run python experiments/phase2/generate_cot_traces.py \
        --verbosity_split 0.34 0.33 0.33

    # Concise ablation: close <think> block so model skips extended reasoning
    uv run python experiments/phase2/generate_cot_traces.py \
        --verbosity_split 0.0 0.5 0.5 --concise_think_mode closed

    # Concise ablation: remove think tags entirely
    uv run python experiments/phase2/generate_cot_traces.py \
        --verbosity_split 0.0 0.5 0.5 --concise_think_mode none

    # Resume an interrupted run
    uv run python experiments/phase2/generate_cot_traces.py \
        --resume_from experiments/phase2/results/traces/traces_deepseek-r1-distill-llama-8b_20260101_120000.jsonl

This follows the difference-in-means methodology: one trace per task, with
contrast coming from natural verbosity variation across tasks and/or prompted
verbosity modes (verbose/normal/concise).
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import time

import torch
from tqdm import tqdm

_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
from steering_vectors_for_faithfulness.config import DEFAULT_MODEL, MONITORABILITY_DATA, MONITORABILITY_SRC
CORE_DATASETS_PATH = MONITORABILITY_DATA / "core_datasets_with_factors"

MAX_CHARS_IN_INPUT = 8000

DEFAULT_VERBOSE_SYSTEM = (
    "You are a careful, methodical reasoner. Always show your full chain of "
    "thought: state every assumption, spell out each intermediate step, and "
    "explicitly connect evidence to conclusions. Enumerate all relevant factors "
    "and explain how each one contributes to your answer."
)
DEFAULT_VERBOSE_USER = (
    "Think through this problem step by step. Be thorough and explicit about "
    "every reasoning step, assumption, and intermediate conclusion.\n\n"
)
DEFAULT_CONCISE_SYSTEM = (
    "You are a direct, efficient problem-solver. Only reason about what is "
    "strictly necessary to reach the answer. Do not enumerate or discuss "
    "individual causal factors of the problem. Omit any reasoning steps that "
    "do not directly contribute to the final answer."
)
DEFAULT_CONCISE_USER = (
    "Answer the following question concisely. Only include reasoning that is "
    "essential to arrive at the answer. Skip any discussion of which factors "
    "contributed to your conclusion.\n\n"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CoT traces for verbosity contrastive dataset"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Model to generate traces from",
    )
    parser.add_argument(
        "--n_bbh", type=int, default=80,
        help="Number of BBH questions to sample (spread across subsets)",
    )
    parser.add_argument(
        "--n_gpqa", type=int, default=70,
        help="Number of GPQA questions to sample (from gpqa_main)",
    )
    parser.add_argument(
        "--n_mmlu", type=int, default=80,
        help="Number of MMLU questions to sample (spread across subsets)",
    )
    parser.add_argument(
        "--all_bbh", action="store_true",
        help="Use all BBH questions (overrides --n_bbh)",
    )
    parser.add_argument(
        "--all_gpqa", action="store_true",
        help="Use all GPQA questions (overrides --n_gpqa)",
    )
    parser.add_argument(
        "--all_mmlu", action="store_true",
        help="Use all MMLU questions (overrides --n_mmlu)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Use all questions from all datasets (overrides all n_* params)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=10000,
        help="Maximum tokens to generate per trace (default: 32768 for long CoT)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature for vLLM generation",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.90,
        help="Nucleus sampling top-p for vLLM generation",
    )
    parser.add_argument(
        "--presence_penalty", type=float, default=0.0,
        help="Presence penalty for vLLM generation",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="Repetition penalty for vLLM generation",
    )
    parser.add_argument(
        "--think_mode", choices=["auto", "open", "none"], default="auto",
        help="Whether to append an explicit <think> tag. auto only does this "
             "for DeepSeek-R1-style models; Gemma/Qwen should use none.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for generation (vLLM handles large batches well)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--task_ids_file", type=str, default=None,
        help="Optional path to a text file with one task_id per line. "
             "If set, tasks are filtered to only those whose task_id is in the file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(SCRIPT_DIR / "results" / "traces"),
        help="Directory to write JSONL output",
    )
    parser.add_argument(
        "--engine", type=str, default="vllm", choices=["vllm", "nnsight"],
        help="Generation engine: vllm (fast) or nnsight (for activation access)",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=-1,
        help="Number of GPUs for tensor parallelism (-1 = auto)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="Optional vLLM max_model_len override. Useful for Gemma/Qwen smoke tests.",
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", default=False,
        help="Load model in 8-bit quantization (nnsight only)",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to existing JSONL to resume from (skip already-processed task_ids)",
    )
    parser.add_argument(
        "--verbosity_split", type=float, nargs=3, default=[0.0, 1.0, 0.0],
        metavar=("VERBOSE", "NORMAL", "CONCISE"),
        help="Fraction of tasks to assign verbose/normal/concise prompts (must sum to 1.0). "
             "Default: 0.0 1.0 0.0 (all normal, original behavior)",
    )
    parser.add_argument(
        "--verbose_system", type=str, default=DEFAULT_VERBOSE_SYSTEM,
        help="System message for verbose traces",
    )
    parser.add_argument(
        "--verbose_user", type=str, default=DEFAULT_VERBOSE_USER,
        help="User-level prefix prepended to the question for verbose traces",
    )
    parser.add_argument(
        "--concise_system", type=str, default=DEFAULT_CONCISE_SYSTEM,
        help="System message for concise traces",
    )
    parser.add_argument(
        "--concise_user", type=str, default=DEFAULT_CONCISE_USER,
        help="User-level prefix prepended to the question for concise traces",
    )
    parser.add_argument(
        "--concise_think_mode", type=str, default="open",
        choices=["open", "closed", "none"],
        help="How to handle <think> tags for concise prompts. "
             "open: normal <think> tag (default), "
             "closed: <think>\\n\\n</think> to close reasoning block, "
             "none: no think tags at all",
    )
    # Cue injection for faithfulness evaluation
    parser.add_argument(
        "--cue_name", type=str, default="none",
        help="Cue to inject into prompts for faithfulness evaluation "
             "(e.g., stanford_professor_recommends). Default: none. "
             "Incompatible with non-trivial --verbosity_split.",
    )
    parser.add_argument(
        "--cue_target", type=str, default="wrong",
        choices=["correct", "wrong"],
        help="Whether the cue points to the correct or a wrong answer (default: wrong)",
    )
    return parser.parse_args()


def think_tag_policy(model_name: str, think_mode: str) -> dict[str, bool]:
    if think_mode == "open":
        return {"strip_existing": True, "append_explicit": True, "expect_tags": True, "enable_thinking": None}
    if think_mode == "none":
        return {"strip_existing": True, "append_explicit": False, "expect_tags": False, "enable_thinking": False}
    lowered = model_name.lower()
    if "deepseek" in lowered and "r1" in lowered:
        return {"strip_existing": True, "append_explicit": True, "expect_tags": True, "enable_thinking": None}
    if "qwen3.5" in lowered or "qwen3_5" in lowered:
        return {"strip_existing": False, "append_explicit": False, "expect_tags": True, "enable_thinking": True}
    return {"strip_existing": False, "append_explicit": False, "expect_tags": False, "enable_thinking": None}


def apply_chat_template_for_model(
    tokenizer,
    messages: list[dict],
    enable_thinking: bool | None = None,
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_model_vllm(
    model_name: str,
    tensor_parallel_size: int,
    seed: int,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
):
    """Load model via vLLM for fast inference."""
    from vllm import LLM
    from transformers import AutoTokenizer
    
    print(f"Loading model via vLLM: {model_name}")
    
    if tensor_parallel_size == -1:
        tensor_parallel_size = torch.cuda.device_count()
    
    kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": "auto",
        "seed": seed,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    model = LLM(**kwargs)
    hf_token = os.getenv("HF_KEY") or os.getenv("HFTOKEN") or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token,
    )
    
    return model, tokenizer


def load_model_nnsight(model_name: str, load_in_8bit: bool = False):
    """Load model and tokenizer via nnsight (for activation access)."""
    from nnsight import LanguageModel
    
    print(f"Loading model via nnsight: {model_name}")
    
    load_kwargs = {
        "dispatch": True,
        "device_map": "auto",
    }
    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    model = LanguageModel(model_name, **load_kwargs)
    
    tokenizer = model.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer


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


def load_core_dataset(json_path: Path) -> list[dict]:
    """Load a CoreDataset JSON file and return list of questions with metadata."""
    with open(json_path) as f:
        data = json.load(f)
    
    questions = []
    for i, q in enumerate(data["questions"]):
        question_text = q.get("raw_question", "")
        if not question_text:
            continue
        
        metadata = q.get("metadata", {})
        causal_factors = metadata.get("causal_factors", [])
        
        answer_variations = q.get("answer_variations", [])
        target = None
        choices = []
        correct_letter = ""
        correct_text = ""
        correct_index = 0
        if answer_variations:
            first_var = answer_variations[0]
            target = first_var.get("correct_text") or first_var.get("correct_letter")
            choices = first_var.get("choices", [])
            correct_letter = first_var.get("correct_letter", "")
            correct_text = first_var.get("correct_text", "")
            correct_index = first_var.get("correct_index", 0)

        questions.append({
            "local_idx": i,
            "question": question_text,
            "target": target,
            "choices": choices,
            "correct_letter": correct_letter,
            "correct_text": correct_text,
            "correct_index": correct_index,
            "causal_factors": causal_factors,
            "dataset_group": metadata.get("dataset_group", ""),
            "subset": metadata.get("subset", ""),
        })
    
    return questions


def sample_tasks_from_datasets(
    n_bbh: int, n_gpqa: int, n_mmlu: int, seed: int,
    all_bbh: bool = False, all_gpqa: bool = False, all_mmlu: bool = False,
) -> list[dict]:
    """Sample tasks from BBH, GPQA, and MMLU datasets.
    
    If all_* flags are True, use all available questions for that dataset.
    """
    random.seed(seed)
    all_tasks = []
    
    bbh_path = CORE_DATASETS_PATH / "bbh"
    gpqa_path = CORE_DATASETS_PATH / "gpqa"
    mmlu_path = CORE_DATASETS_PATH / "mmlu"
    
    if (n_bbh > 0 or all_bbh) and bbh_path.exists():
        bbh_files = list(bbh_path.glob("*.json"))
        bbh_questions = []
        for f in bbh_files:
            qs = load_core_dataset(f)
            for q in qs:
                q["source_file"] = f.name
            bbh_questions.extend(qs)
        
        if bbh_questions:
            n_to_sample = len(bbh_questions) if all_bbh else min(n_bbh, len(bbh_questions))
            sampled = random.sample(bbh_questions, n_to_sample)
            for i, q in enumerate(sampled):
                q["task_id"] = f"bbh_{q['subset']}_{q['local_idx']}"
                q["dataset"] = "bbh"
            all_tasks.extend(sampled)
            print(f"{'Using all' if all_bbh else 'Sampled'} {len(sampled)} BBH tasks (from {len(bbh_questions)} total)")
    
    if (n_gpqa > 0 or all_gpqa) and gpqa_path.exists():
        gpqa_main_file = gpqa_path / "gpqa_main_core_with_factors.json"
        if gpqa_main_file.exists():
            gpqa_questions = load_core_dataset(gpqa_main_file)
            for q in gpqa_questions:
                q["source_file"] = gpqa_main_file.name
            
            n_to_sample = len(gpqa_questions) if all_gpqa else min(n_gpqa, len(gpqa_questions))
            sampled = random.sample(gpqa_questions, n_to_sample)
            for i, q in enumerate(sampled):
                q["task_id"] = f"gpqa_main_{q['local_idx']}"
                q["dataset"] = "gpqa"
            all_tasks.extend(sampled)
            print(f"{'Using all' if all_gpqa else 'Sampled'} {len(sampled)} GPQA tasks (from {len(gpqa_questions)} total)")
    
    if (n_mmlu > 0 or all_mmlu) and mmlu_path.exists():
        mmlu_files = list(mmlu_path.glob("*.json"))
        mmlu_questions = []
        for f in mmlu_files:
            qs = load_core_dataset(f)
            for q in qs:
                q["source_file"] = f.name
            mmlu_questions.extend(qs)
        
        if mmlu_questions:
            n_to_sample = len(mmlu_questions) if all_mmlu else min(n_mmlu, len(mmlu_questions))
            sampled = random.sample(mmlu_questions, n_to_sample)
            for i, q in enumerate(sampled):
                q["task_id"] = f"mmlu_{q['subset']}_{q['local_idx']}"
                q["dataset"] = "mmlu"
            all_tasks.extend(sampled)
            print(f"{'Using all' if all_mmlu else 'Sampled'} {len(sampled)} MMLU tasks (from {len(mmlu_questions)} total)")
    
    random.shuffle(all_tasks)
    return all_tasks


def format_question_with_choices(question: str, choices: list[str]) -> str:
    """Format a question with multiple choice options if available."""
    if not choices:
        return question
    
    formatted = question.strip()
    if not formatted.endswith("?") and not formatted.endswith(":"):
        formatted += "\n"
    
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, choice in enumerate(choices):
        if i < len(letters):
            formatted += f"\n{letters[i]}. {choice}"
    
    return formatted


def build_chat_prompt(
    tokenizer, question: str, choices: list[str],
    verbosity_mode: str = "normal",
    verbose_system: str = DEFAULT_VERBOSE_SYSTEM,
    verbose_user: str = DEFAULT_VERBOSE_USER,
    concise_system: str = DEFAULT_CONCISE_SYSTEM,
    concise_user: str = DEFAULT_CONCISE_USER,
    concise_think_mode: str = "open",
    strip_existing_think_tag: bool = True,
    append_think_tag: bool = True,
    enable_thinking: bool | None = None,
) -> str:
    """Build a chat prompt using the tokenizer's chat template."""
    formatted_q = format_question_with_choices(question, choices)

    messages = []
    if verbosity_mode == "verbose":
        messages.append({"role": "system", "content": verbose_system})
        formatted_q = verbose_user + formatted_q
    elif verbosity_mode == "concise":
        messages.append({"role": "system", "content": concise_system})
        formatted_q = concise_user + formatted_q
    messages.append({"role": "user", "content": formatted_q})

    prompt = apply_chat_template_for_model(
        tokenizer=tokenizer,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    # Some chat templates (e.g. DeepSeek-R1) already append <think> via
    # add_generation_prompt. Strip it so we control the think tag ourselves.
    if strip_existing_think_tag and prompt.rstrip().endswith("<think>"):
        prompt = prompt[: prompt.rfind("<think>")]
    if not append_think_tag:
        pass
    elif verbosity_mode == "concise" and concise_think_mode == "closed":
        prompt = prompt + "<think>\n\n</think>"
    elif verbosity_mode == "concise" and concise_think_mode == "none":
        pass  # no think tags at all
    else:
        prompt = prompt + "<think>\n"
    return prompt


def build_cued_prompt(
    tokenizer, task: dict, cue_obj, cue_template, cue_target: str,
    strip_existing_think_tag: bool = True,
    append_think_tag: bool = True,
    enable_thinking: bool | None = None,
) -> tuple[str, str, str]:
    """Render a question with an injected cue using Meek et al.'s template system.

    Mirrors create_faithfulness_examples.py: picks a cue_target_letter (wrong or
    correct), renders the question + cue + choices via the monitorability
    template, strips the "ANSWER: letter" formatting instruction, and wraps the
    result in the chat template with a leading <think> tag.

    Returns (prompt, cue_text, cue_target_letter).
    """
    from measuring_cot_monitorability.data_utils.core_schema import StandardizedExample

    correct_letter = task.get("correct_letter", "") or "A"
    choices = task.get("choices", [])
    num_choices = len(choices)

    if cue_target == "wrong":
        letters = [chr(ord("A") + i) for i in range(num_choices)]
        wrong_letters = [l for l in letters if l != correct_letter]
        cue_target_letter = random.choice(wrong_letters) if wrong_letters else "B"
    else:
        cue_target_letter = correct_letter

    example = StandardizedExample(
        id=task.get("task_id", ""),
        question=task["question"],
        choices=choices,
        correct_answer_text=task.get("correct_text", task.get("target", "")),
        correct_answer_letter=correct_letter,
        correct_answer_index=task.get("correct_index", 0),
        answer_format="multiple_choice",
        metadata={"cue_target_answer": cue_target_letter},
    )

    # Some installed versions of measuring_cot_monitorability define metadata
    # cues with `metadata_template` but lack the helper method the template
    # renderer expects. Patch the cue instance locally instead of modifying the
    # dependency checkout.
    if cue_obj.embeds_metadata and not hasattr(cue_obj, "generate_metadata_block"):
        def _generate_metadata_block(target_answer: str, success_rate: str = "89") -> str:
            return cue_obj.metadata_template.format(
                correct_letter=target_answer,
                success_rate=success_rate,
            )

        cue_obj.generate_metadata_block = _generate_metadata_block

    rendered = cue_template.render(example, cue_obj)
    rendered = rendered.replace(
        "Please format your final answer as: ANSWER: letter", ""
    ).rstrip()

    message = {"role": "user", "content": rendered}
    prompt = apply_chat_template_for_model(
        tokenizer=tokenizer,
        messages=[message],
        enable_thinking=enable_thinking,
    )
    if strip_existing_think_tag and prompt.rstrip().endswith("<think>"):
        prompt = prompt[: prompt.rfind("<think>")]
    if append_think_tag:
        prompt = prompt + "<think>\n"

    cue_text = cue_obj.format_cue_text(cue_target_letter, num_choices=num_choices)
    if cue_obj.embeds_metadata and cue_obj.metadata_template:
        cue_text = cue_obj.metadata_template.format(
            correct_letter=cue_target_letter,
            success_rate="89",
        )
    return prompt, cue_text, cue_target_letter


def extract_think_and_answer(
    response: str, expect_think_tags: bool = True,
) -> tuple[str, str]:
    """Extract <think> content and final answer from model response.

    The prompt already contains the opening <think> tag, so the response
    normally starts with reasoning text.  However, some chat templates
    (e.g. DeepSeek-R1) or prior runs with the duplicate-tag bug may cause
    the response to start with an extra ``<think>`` — strip it so
    downstream fields are clean.
    """
    cleaned = response.strip()
    # Strip leading <think> tag(s) that the model may have echoed.
    while cleaned.startswith("<think>"):
        cleaned = cleaned[len("<think>"):].lstrip("\n")

    if "</think>" in cleaned:
        parts = cleaned.split("</think>", 1)
        think_content = parts[0].strip()
        final_answer = parts[1].strip() if len(parts) > 1 else ""
    elif not expect_think_tags:
        # Gemma/Qwen-style instruct models should not be forced into DeepSeek's
        # explicit <think> protocol. Keep the whole response visible to both the
        # faithfulness and correctness scorers.
        think_content = cleaned.strip()
        final_answer = cleaned.strip()
    else:
        # No closing tag — model may have been cut off or didn't use thinking
        think_content = cleaned.strip()
        final_answer = ""

    return think_content, final_answer


def generate_batch_vllm(
    model, tokenizer, prompts: list[str], max_new_tokens: int,
    temperature: float = 0.6, top_p: float = 0.90,
    presence_penalty: float = 0.0, repetition_penalty: float = 1.0,
) -> list[str]:
    """Generate responses for a batch of prompts using vLLM."""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,  # No limit - generate until EOS or context limit
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    
    request_outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in request_outputs]
    
    return responses


def generate_batch_nnsight(
    model, tokenizer, prompts: list[str], max_new_tokens: int,
    temperature: float = 0.0, top_p: float = 1.0,
    presence_penalty: float = 0.0, repetition_penalty: float = 1.0,
) -> list[str]:
    """Generate responses for a batch of prompts using nnsight."""
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    
    assert input_ids.dim() == 2, f"Expected 2D input_ids, got {input_ids.shape}"
    
    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    ) as gen:
        outputs = model.generator.output.save()
    
    prompt_lengths = attention_mask.sum(dim=1)
    responses = []
    for i in range(outputs.shape[0]):
        prompt_len = int(prompt_lengths[i].item())
        new_tokens = outputs[i][prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)
    
    return responses


def classify_degenerate(
    record: dict, max_new_tokens: int, concise_think_mode: str = "open",
) -> str | None:
    """Return a reason string if the record is degenerate, else None.

    Stored as ``degenerate_reason`` on each record so downstream scripts
    (e.g. build_contrastive_dataset.py) can filter with a simple null check.
    """
    reasons = []
    is_concise_no_think = (
        concise_think_mode in ("closed", "none")
        and record.get("verbosity_mode") == "concise"
    )
    expects_think_content = record.get("generation_config", {}).get("expect_think_tags", True)

    if not record.get("final_answer", "").strip():
        reasons.append("empty_final_answer")

    if (
        expects_think_content
        and not is_concise_no_think
        and not record.get("think_content", "").strip()
    ):
        reasons.append("empty_think_content")

    num_tokens = record.get("num_tokens", 0)
    if num_tokens >= max_new_tokens * 0.95:
        reasons.append("near_max_tokens")

    return "; ".join(reasons) if reasons else None


def assign_verbosity_modes(
    tasks: list[dict], split: tuple[float, float, float], seed: int,
) -> None:
    """Assign a verbosity_mode tag to each task in-place.

    split = (verbose_frac, normal_frac, concise_frac), must sum to ~1.0.
    """
    n = len(tasks)
    n_verbose = round(split[0] * n)
    n_concise = round(split[2] * n)
    n_normal = n - n_verbose - n_concise

    modes = (
        ["verbose"] * n_verbose
        + ["normal"] * n_normal
        + ["concise"] * n_concise
    )
    rng = random.Random(seed)
    rng.shuffle(modes)
    for task, mode in zip(tasks, modes):
        task["verbosity_mode"] = mode


def load_completed_task_ids(jsonl_path: str) -> set[str]:
    """Load task_ids that have already been processed."""
    completed = set()
    if jsonl_path and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed.add(record.get("task_id", ""))
    return completed


def main():
    args = parse_args()

    vsplit = tuple(args.verbosity_split)
    if abs(sum(vsplit) - 1.0) > 0.01:
        print(f"ERROR: --verbosity_split must sum to 1.0, got {sum(vsplit):.3f}")
        return

    # Cue setup (faithfulness evaluation)
    cue_obj = None
    cue_template = None
    if args.cue_name != "none":
        if vsplit != (0.0, 1.0, 0.0):
            print(
                "ERROR: --cue_name cannot be combined with a non-trivial "
                "--verbosity_split. Cued runs must use the default (0 1 0)."
            )
            return
        if str(MONITORABILITY_SRC) not in sys.path:
            sys.path.insert(0, str(MONITORABILITY_SRC))
        from measuring_cot_monitorability.prompts.cue_system import default_cue_registry
        from measuring_cot_monitorability.prompts.template_registry import (
            PromptTemplateRegistry,
            setup_default_templates,
        )
        cue_obj = default_cue_registry.get_cue_by_name(args.cue_name)
        if cue_obj is None:
            all_names = [
                c.name
                for cues in default_cue_registry.get_all_cues().values()
                for c in cues
            ]
            print(f"ERROR: Unknown cue name: {args.cue_name}. Available: {all_names}")
            return
        template_registry = PromptTemplateRegistry()
        setup_default_templates(template_registry)
        cue_template = template_registry.get_template("standard")
        print(f"Cue injection: {args.cue_name} (target={args.cue_target})")

    if args.engine == "vllm":
        model, tokenizer = load_model_vllm(
            args.model,
            args.tensor_parallel_size,
            args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        generate_fn = generate_batch_vllm
    else:
        model, tokenizer = load_model_nnsight(args.model, args.load_in_8bit)
        generate_fn = generate_batch_nnsight
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cue_suffix = f"_{args.cue_name}" if cue_obj is not None else ""
    output_path = os.path.join(
        args.output_dir, f"traces_{model_short}_{timestamp}{cue_suffix}.jsonl"
    )
    think_policy = think_tag_policy(args.model, args.think_mode)
    generation_config = {
        "model": args.model,
        "engine": args.engine,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "think_mode": args.think_mode,
        "strip_existing_think_tag": think_policy["strip_existing"],
        "add_think_tag": think_policy["append_explicit"],
        "expect_think_tags": think_policy["expect_tags"],
        "enable_thinking": think_policy["enable_thinking"],
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "cue_name": args.cue_name,
        "cue_target": args.cue_target,
        "n_bbh": args.n_bbh,
        "n_gpqa": args.n_gpqa,
        "n_mmlu": args.n_mmlu,
        "all_bbh": args.all_bbh or args.all,
        "all_gpqa": args.all_gpqa or args.all,
        "all_mmlu": args.all_mmlu or args.all,
        "task_ids_file": args.task_ids_file,
    }
    use_all_bbh = args.all_bbh or args.all
    use_all_gpqa = args.all_gpqa or args.all
    use_all_mmlu = args.all_mmlu or args.all
    
    all_tasks = sample_tasks_from_datasets(
        args.n_bbh, args.n_gpqa, args.n_mmlu, args.seed,
        all_bbh=use_all_bbh, all_gpqa=use_all_gpqa, all_mmlu=use_all_mmlu,
    )

    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        before = len(all_tasks)
        all_tasks = [t for t in all_tasks if t["task_id"] in allowed]
        print(f"Filtered tasks via --task_ids_file: {before} -> {len(all_tasks)}")

    # Deterministically shuffle answer choices per task (see shuffle helper).
    # Applied in place so every downstream prompt-builder sees the shuffled
    # layout. task["target"] is left unchanged (it's the correct-answer text).
    for task in all_tasks:
        new_choices, new_correct_index, new_correct_letter = shuffle_choices_for_task(
            task["task_id"], task.get("choices", []), task.get("correct_index", 0),
        )
        task["choices"] = new_choices
        task["correct_index"] = new_correct_index
        task["correct_letter"] = new_correct_letter

    assign_verbosity_modes(all_tasks, vsplit, args.seed)
    from collections import Counter
    mode_counts = Counter(t["verbosity_mode"] for t in all_tasks)
    print(f"\nTotal tasks to process: {len(all_tasks)}")
    print(f"Verbosity split: {dict(mode_counts)}")

    completed_ids = set()
    if args.resume_from:
        completed_ids = load_completed_task_ids(args.resume_from)
        output_path = args.resume_from
        print(f"Resuming from {args.resume_from}, {len(completed_ids)} tasks already done")

    with open(output_path.replace(".jsonl", "_config.json"), "w") as cf:
        json.dump(generation_config, cf, indent=2, sort_keys=True)
    
    tasks_to_process = [t for t in all_tasks if t["task_id"] not in completed_ids]
    print(f"Tasks remaining: {len(tasks_to_process)}")
    
    if not tasks_to_process:
        print("All tasks already processed!")
        return
    
    n_written = 0
    with open(output_path, "a") as f:
        for batch_start in tqdm(range(0, len(tasks_to_process), args.batch_size)):
            batch_tasks = tasks_to_process[batch_start:batch_start + args.batch_size]
            
            prompts = []
            batch_cue_texts: list[str | None] = []
            batch_cue_target_letters: list[str | None] = []
            for task in batch_tasks:
                if cue_obj is not None:
                    prompt, cue_text, cue_target_letter = build_cued_prompt(
                        tokenizer, task, cue_obj, cue_template, args.cue_target,
                        strip_existing_think_tag=generation_config["strip_existing_think_tag"],
                        append_think_tag=generation_config["add_think_tag"],
                        enable_thinking=generation_config["enable_thinking"],
                    )
                    batch_cue_texts.append(cue_text)
                    batch_cue_target_letters.append(cue_target_letter)
                else:
                    prompt = build_chat_prompt(
                        tokenizer, task["question"], task.get("choices", []),
                        verbosity_mode=task.get("verbosity_mode", "normal"),
                        verbose_system=args.verbose_system,
                        verbose_user=args.verbose_user,
                        concise_system=args.concise_system,
                        concise_user=args.concise_user,
                        concise_think_mode=args.concise_think_mode,
                        strip_existing_think_tag=generation_config["strip_existing_think_tag"],
                        append_think_tag=generation_config["add_think_tag"],
                        enable_thinking=generation_config["enable_thinking"],
                    )
                    batch_cue_texts.append(None)
                    batch_cue_target_letters.append(None)
                if len(prompt) > MAX_CHARS_IN_INPUT:
                    print(f"Warning: prompt for {task['task_id']} exceeds max length, truncating")
                    prompt = prompt[:MAX_CHARS_IN_INPUT]
                prompts.append(prompt)
            
            t0 = time.time()
            try:
                responses = generate_fn(
                    model, tokenizer, prompts, args.max_new_tokens,
                    args.temperature, args.top_p, args.presence_penalty,
                    args.repetition_penalty,
                )
            except Exception as e:
                print(f"Error generating batch starting at {batch_start}: {e}")
                continue
            elapsed = time.time() - t0
            
            batch_degen = 0
            for task, prompt, response, cue_text, cue_target_letter in zip(
                batch_tasks, prompts, responses,
                batch_cue_texts, batch_cue_target_letters,
            ):
                think_content, final_answer = extract_think_and_answer(
                    response,
                    expect_think_tags=generation_config["expect_think_tags"],
                )

                # For concise modes that suppress thinking (closed/none), if
                # the model didn't produce any </think> (i.e. no real thinking
                # happened), the whole response is the final answer.
                if (
                    args.concise_think_mode in ("closed", "none")
                    and task.get("verbosity_mode") == "concise"
                    and "</think>" not in response
                ):
                    final_answer = think_content.strip()
                    think_content = ""

                record = {
                    "task_id": task["task_id"],
                    "dataset": task["dataset"],
                    "subset": task.get("subset", ""),
                    "question": task["question"],
                    "choices": task.get("choices", []),
                    "target": task.get("target"),
                    "causal_factors": task.get("causal_factors", []),
                    "think_content": think_content,
                    "final_answer": final_answer,
                    "full_response": response,
                    "full_context": prompt + response,
                    "num_tokens": len(tokenizer.encode(response)),
                    "model": args.model,
                    "generation_config": generation_config,
                    "elapsed_seconds": round(elapsed / len(batch_tasks), 2),
                    "verbosity_mode": task.get("verbosity_mode", "normal"),
                }

                if cue_obj is not None:
                    record["has_cue"] = True
                    record["cue_name"] = args.cue_name
                    record["cue_text"] = cue_text
                    record["cue_target_letter"] = cue_target_letter
                    record["judge_prompt"] = cue_obj.judge_prompt

                record["degenerate_reason"] = classify_degenerate(
                    record, args.max_new_tokens, args.concise_think_mode,
                )

                f.write(json.dumps(record) + "\n")
                n_written += 1
                if record["degenerate_reason"]:
                    batch_degen += 1

            print(f"  Batch degenerate: {batch_degen}/{len(batch_tasks)}")

            f.flush()

            if args.engine == "nnsight":
                torch.cuda.empty_cache()
                gc.collect()

    # End-of-run degenerate summary
    all_records = []
    with open(output_path) as rf:
        for line in rf:
            if line.strip():
                all_records.append(json.loads(line))

    n_total = len(all_records)
    degen_records = [r for r in all_records if r.get("degenerate_reason")]
    n_degen = len(degen_records)

    print(f"\n{'='*60}")
    print(f"Wrote {n_written} traces to {output_path}")
    print(f"Degenerate: {n_degen}/{n_total} ({n_degen/n_total:.1%})" if n_total else "")

    if degen_records:
        from collections import Counter
        reason_counts = Counter()
        for r in degen_records:
            for reason in r["degenerate_reason"].split("; "):
                reason_counts[reason] += 1
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()

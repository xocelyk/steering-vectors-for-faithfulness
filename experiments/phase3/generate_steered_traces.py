"""
Generate steered CoT traces by applying steering vectors during generation.

Uses nnsight + vLLM to inject steering vectors at specified layers during
autoregressive generation. Supports sweeping over layers, alpha values,
and three steering strategies.

Strategies:
    - prompt_last_token:      One-shot. Vector added to last prompt token during prefill only.
    - prompt_all_tokens:      One-shot. Vector added to all prompt tokens during prefill only.
    - every_step_last_token:  Continuous. Vector added to last position at every forward pass.

Usage:
    # Single configuration
    uv run python experiments/phase3/generate_steered_traces.py \
        --vectors_dir experiments/phase3/results/vectors \
        --layers 16 --alphas 1.0 --strategy prompt_last_token \
        --n_gpqa 20

    # Sweep over layers and alphas
    uv run python experiments/phase3/generate_steered_traces.py \
        --vectors_dir experiments/phase3/results/vectors \
        --layers 8 12 16 20 24 --alphas 0.5 1.0 2.0 4.0 \
        --strategy every_step_last_token --all_gpqa

    # Also generates a baseline (no steering) for comparison
    uv run python experiments/phase3/generate_steered_traces.py \
        --vectors_dir experiments/phase3/results/vectors \
        --layers 0 8 11 12 16 23 31 --alphas 0.25 0.5 0.75 1.0  --strategy every_step_last_token \
        --n_gpqa 50 --batch_size 50
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import argparse
import json
import os
import random
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

PHASE2_DIR = Path(__file__).resolve().parent.parent / "phase2"
sys.path.insert(0, str(PHASE2_DIR))
from generate_cot_traces import classify_degenerate, shuffle_choices_for_task

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
from steering_vectors_for_faithfulness.config import DEFAULT_MODEL, MONITORABILITY_DATA, MONITORABILITY_SRC
CORE_DATASETS_PATH = MONITORABILITY_DATA / "core_datasets_with_factors"

if str(MONITORABILITY_SRC) not in sys.path:
    sys.path.insert(0, str(MONITORABILITY_SRC))

STRATEGIES = ["prompt_last_token", "prompt_all_tokens", "every_step_last_token", "every_step_all_tokens"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate steered CoT traces with steering vectors"
    )
    # Steering config
    parser.add_argument(
        "--vectors_dir", type=str, required=True,
        help="Path to a .pt vectors file, or a directory containing one",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True,
        help="Layer(s) to apply steering at (sweep if multiple)",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=None,
        help="Steering strength(s) (sweep if multiple). Use 0.0 for baseline. "
             "Required unless --use_affine_alphas is set.",
    )
    parser.add_argument(
        "--use_affine_alphas", action="store_true",
        help="Use affine alphas from the steering vectors .pt file (computed by "
             "extract_vectors.py --compute_affine_alphas). Uses alpha_plus per layer. "
             "Can be combined with --alphas to include both in the sweep.",
    )
    parser.add_argument(
        "--strategy", type=str, default="prompt_last_token",
        choices=STRATEGIES,
        help="Steering strategy",
    )
    # Dataset args (mirrors generate_cot_traces.py)
    parser.add_argument("--n_bbh", type=int, default=0)
    parser.add_argument("--n_gpqa", type=int, default=50)
    parser.add_argument("--n_mmlu", type=int, default=0)
    parser.add_argument("--all_bbh", action="store_true")
    parser.add_argument("--all_gpqa", action="store_true")
    parser.add_argument("--all_mmlu", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help="Optional path to a text file with one task_id per line. "
             "If set, tasks are filtered to only those whose task_id is in the file.",
    )
    # Generation args
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max_new_tokens", type=int, default=10000,
        help="Maximum tokens to generate per trace",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (DeepSeek recommends 0.5-0.7)",
    )
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--think_mode", choices=["auto", "open", "none"], default="auto",
        help="Whether to append an explicit <think> tag. auto only does this "
             "for DeepSeek-R1-style models; Gemma/Qwen should use none.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: results/steered_traces/)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=30000,
        help="vLLM max_model_len override",
    )
    parser.add_argument(
        "--batch_size", type=int, default=25,
        help="Number of prompts per batched trace call",
    )
    # Cue injection for faithfulness evaluation
    parser.add_argument(
        "--cue_name", type=str, default="none",
        help="Cue to inject into prompts for faithfulness evaluation "
             "(e.g., stanford_professor_recommends). Default: none.",
    )
    parser.add_argument(
        "--cue_target", type=str, default="wrong",
        choices=["correct", "wrong"],
        help="Whether the cue points to the correct or wrong answer (default: wrong)",
    )
    return parser.parse_args()


def think_tag_policy(model_name: str, think_mode: str) -> dict[str, bool]:
    if think_mode == "open":
        return {"strip_existing": True, "append_explicit": True, "expect_tags": True}
    if think_mode == "none":
        return {"strip_existing": True, "append_explicit": False, "expect_tags": False}
    lowered = model_name.lower()
    if "deepseek" in lowered and "r1" in lowered:
        return {"strip_existing": True, "append_explicit": True, "expect_tags": True}
    return {"strip_existing": False, "append_explicit": False, "expect_tags": False}


# ---------------------------------------------------------------------------
# Dataset loading (mirrored from generate_cot_traces.py)
# ---------------------------------------------------------------------------

def load_core_dataset(json_path: Path) -> list[dict]:
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
        if answer_variations:
            first_var = answer_variations[0]
            target = first_var.get("correct_text") or first_var.get("correct_letter")
            choices = first_var.get("choices", [])
        correct_letter = first_var.get("correct_letter", "") if answer_variations else ""
        correct_index = first_var.get("correct_index", 0) if answer_variations else 0
        correct_text = first_var.get("correct_text", "") if answer_variations else ""
        questions.append({
            "local_idx": i,
            "question": question_text,
            "target": target,
            "choices": choices,
            "causal_factors": causal_factors,
            "dataset_group": metadata.get("dataset_group", ""),
            "subset": metadata.get("subset", ""),
            "correct_letter": correct_letter,
            "correct_index": correct_index,
            "correct_text": correct_text,
        })
    return questions


def sample_tasks_from_datasets(
    n_bbh: int, n_gpqa: int, n_mmlu: int, seed: int,
    all_bbh: bool = False, all_gpqa: bool = False, all_mmlu: bool = False,
) -> list[dict]:
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
            for q in sampled:
                q["task_id"] = f"bbh_{q['subset']}_{q['local_idx']}"
                q["dataset"] = "bbh"
            all_tasks.extend(sampled)
            print(f"{'All' if all_bbh else 'Sampled'} {len(sampled)} BBH tasks")

    if (n_gpqa > 0 or all_gpqa) and gpqa_path.exists():
        gpqa_main_file = gpqa_path / "gpqa_main_core_with_factors.json"
        if gpqa_main_file.exists():
            gpqa_questions = load_core_dataset(gpqa_main_file)
            for q in gpqa_questions:
                q["source_file"] = gpqa_main_file.name
            n_to_sample = len(gpqa_questions) if all_gpqa else min(n_gpqa, len(gpqa_questions))
            sampled = random.sample(gpqa_questions, n_to_sample)
            for q in sampled:
                q["task_id"] = f"gpqa_main_{q['local_idx']}"
                q["dataset"] = "gpqa"
            all_tasks.extend(sampled)
            print(f"{'All' if all_gpqa else 'Sampled'} {len(sampled)} GPQA tasks")

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
            for q in sampled:
                q["task_id"] = f"mmlu_{q['subset']}_{q['local_idx']}"
                q["dataset"] = "mmlu"
            all_tasks.extend(sampled)
            print(f"{'All' if all_mmlu else 'Sampled'} {len(sampled)} MMLU tasks")

    random.shuffle(all_tasks)
    return all_tasks


# ---------------------------------------------------------------------------
# Prompt construction and output parsing (mirrored from generate_cot_traces.py)
# ---------------------------------------------------------------------------

def format_question_with_choices(question: str, choices: list[str]) -> str:
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
    strip_existing_think_tag: bool = True,
    append_think_tag: bool = True,
) -> str:
    formatted_q = format_question_with_choices(question, choices)
    message = {"role": "user", "content": formatted_q}
    prompt = tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    # Some chat templates (e.g. DeepSeek-R1) already append <think> via
    # add_generation_prompt. Strip it so we control the think tag ourselves.
    if strip_existing_think_tag and prompt.rstrip().endswith("<think>"):
        prompt = prompt[: prompt.rfind("<think>")]
    if append_think_tag:
        prompt = prompt + "<think>\n"
    return prompt


def build_cued_prompt(
    tokenizer, task: dict, cue_obj, cue_template, cue_target: str,
    strip_existing_think_tag: bool = True,
    append_think_tag: bool = True,
) -> tuple[str, str, str]:
    """Build a prompt with a cue injected using Meek et al.'s template renderer.

    Returns (prompt, cue_text, cue_target_letter).
    """
    from measuring_cot_monitorability.data_utils.core_schema import StandardizedExample

    correct_letter = task.get("correct_letter", "A")
    num_choices = len(task.get("choices", []))

    if cue_target == "wrong":
        letters = [chr(ord("A") + i) for i in range(num_choices)]
        wrong_letters = [l for l in letters if l != correct_letter]
        cue_target_letter = random.choice(wrong_letters) if wrong_letters else "B"
    else:
        cue_target_letter = correct_letter

    example = StandardizedExample(
        id=task.get("task_id", ""),
        question=task["question"],
        choices=task.get("choices", []),
        correct_answer_text=task.get("correct_text", task.get("target", "")),
        correct_answer_letter=correct_letter,
        correct_answer_index=task.get("correct_index", 0),
        answer_format="multiple_choice",
        metadata={"cue_target_answer": cue_target_letter},
    )

    rendered = cue_template.render(example, cue_obj)
    # Strip the answer format instruction (DeepSeek uses its own)
    rendered = rendered.replace("Please format your final answer as: ANSWER: letter", "").rstrip()

    # Wrap in chat template + <think>
    message = {"role": "user", "content": rendered}
    prompt = tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    if strip_existing_think_tag and prompt.rstrip().endswith("<think>"):
        prompt = prompt[: prompt.rfind("<think>")]
    if append_think_tag:
        prompt = prompt + "<think>\n"

    cue_text = cue_obj.format_cue_text(cue_target_letter, num_choices=num_choices)
    return prompt, cue_text, cue_target_letter


def extract_think_and_answer(
    response: str, expect_think_tags: bool = True,
) -> tuple[str, str]:
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
        think_content = cleaned.strip()
        final_answer = ""
    return think_content, final_answer


# ---------------------------------------------------------------------------
# Model loading and steered generation
# ---------------------------------------------------------------------------

def load_model(model_name: str, gpu_memory_utilization: float, max_model_len: int):
    from nnsight.modeling.vllm import VLLM

    print(f"Loading {model_name} via nnsight VLLM...")
    model = VLLM(
        model_name,
        dispatch=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    print("  Model loaded.")
    return model


def get_layer_module(model, layer: int):
    """Return a transformer block for common HF/nnsight model wrappers."""
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
    ]
    for path in candidate_paths:
        obj = model
        try:
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            layers = getattr(obj, path[-1])
            return layers[layer]
        except AttributeError:
            continue
    raise AttributeError(
        "Could not locate transformer layers on model wrapper. Tried: "
        + ", ".join(".".join(path) for path in candidate_paths)
    )


def generate_steered_batch(
    model,
    prompts: list[str],
    steering_vector: torch.Tensor,
    layer: int,
    alpha: float,
    strategy: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    repetition_penalty: float,
) -> list[str]:
    """Generate steered traces for a batch of prompts.

    Uses tracer.invoke() to batch multiple prompts in a single trace call,
    letting vLLM batch the GPU computation across prompts. Steering
    interventions are applied via nnsight hooks; the generated text is read
    from vLLM's native RequestOutput objects.

    Returns list of generated texts (one per prompt).
    """
    layer_module = get_layer_module(model, layer)

    # Capture vLLM's RequestOutput objects so we can read the generated text.
    # nnsight's VLLM.__call__ runs vllm_entrypoint.generate() internally but
    # discards the output text; we intercept generate() to keep it.
    captured_outputs = []
    orig_generate = model.vllm_entrypoint.generate

    def _capturing_generate(*args, **kwargs):
        results = orig_generate(*args, **kwargs)
        captured_outputs.extend(results)
        return results

    model.vllm_entrypoint.generate = _capturing_generate

    try:
        with model.trace(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        ) as tracer:
            for prompt in prompts:
                with tracer.invoke(prompt):
                    if alpha != 0.0:
                        if strategy == "prompt_last_token":
                            with tracer.iter[:1]:
                                hs = layer_module.output[0].clone()
                                hs[-1] += alpha * steering_vector
                                layer_module.output = (hs, layer_module.output[1])

                        elif strategy == "prompt_all_tokens":
                            with tracer.iter[:1]:
                                hs = layer_module.output[0].clone()
                                hs[:] += alpha * steering_vector
                                layer_module.output = (hs, layer_module.output[1])

                        elif strategy == "every_step_last_token":
                            with tracer.all():
                                hs = layer_module.output[0].clone()
                                hs[-1] += alpha * steering_vector
                                layer_module.output = (hs, layer_module.output[1])

                        elif strategy == "every_step_all_tokens":
                            with tracer.all():
                                hs = layer_module.output[0].clone()
                                hs[:] += alpha * steering_vector
                                layer_module.output = (hs, layer_module.output[1])
    finally:
        model.vllm_entrypoint.generate = orig_generate

    results = [output.outputs[0].text for output in captured_outputs]
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / "steered_traces"
    os.makedirs(output_dir, exist_ok=True)

    # Load steering vectors
    vp = Path(args.vectors_dir)
    if vp.is_file() and vp.suffix == ".pt":
        vectors_path = vp
    elif vp.is_dir():
        pt_files = sorted(vp.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {vp}")
        vectors_path = pt_files[0]
        if len(pt_files) > 1:
            print(f"Warning: multiple .pt files in {vp}, using {vectors_path.name}")
    else:
        raise FileNotFoundError(f"Not a .pt file or directory: {vp}")
    sv_data = torch.load(vectors_path, weights_only=False)
    vectors = sv_data["vectors"]
    sv_metadata = sv_data["metadata"]
    print(f"Loaded steering vectors: {len(vectors)} layers, method={sv_metadata['method']}")

    # Resolve alphas: manual, affine, or both
    affine_alphas_data = sv_data.get("affine_alphas")

    if args.use_affine_alphas:
        if affine_alphas_data is None:
            print("ERROR: --use_affine_alphas set but no affine_alphas found in vectors file. "
                  "Re-run extract_vectors.py with --compute_affine_alphas.")
            return
        print("Using affine alphas from vectors file:")
        for layer in args.layers:
            if layer in affine_alphas_data:
                aa = affine_alphas_data[layer]
                print(f"  Layer {layer}: alpha_range={aa['alpha_range']:.4f} "
                      f"(plus={aa['alpha_plus']:.4f}, minus={aa['alpha_minus']:.4f})")

    if args.alphas is None and not args.use_affine_alphas:
        print("ERROR: Must specify --alphas or --use_affine_alphas (or both)")
        return

    for layer in args.layers:
        if layer not in vectors:
            print(f"ERROR: No steering vector for layer {layer}")
            return

    # Load tasks
    use_all_bbh = args.all_bbh or args.all
    use_all_gpqa = args.all_gpqa or args.all
    use_all_mmlu = args.all_mmlu or args.all
    tasks = sample_tasks_from_datasets(
        args.n_bbh, args.n_gpqa, args.n_mmlu, args.seed,
        all_bbh=use_all_bbh, all_gpqa=use_all_gpqa, all_mmlu=use_all_mmlu,
    )
    if not tasks:
        print("ERROR: No tasks sampled")
        return

    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        before = len(tasks)
        tasks = [t for t in tasks if t["task_id"] in allowed]
        print(f"Filtered tasks via --task_ids_file: {before} -> {len(tasks)}")

    # Deterministically shuffle answer choices per task. Must match the shuffle
    # applied in generate_cot_traces.py (same helper, same task_id seed) so
    # steered eval and baseline eval see identical prompts for each task.
    for task in tasks:
        new_choices, new_correct_index, new_correct_letter = shuffle_choices_for_task(
            task["task_id"], task.get("choices", []), task.get("correct_index", 0),
        )
        task["choices"] = new_choices
        task["correct_index"] = new_correct_index
        task["correct_letter"] = new_correct_letter

    print(f"Total tasks: {len(tasks)}")

    # Set up cue injection if requested
    cue_obj = None
    cue_template = None
    if args.cue_name != "none":
        from measuring_cot_monitorability.prompts.cue_system import default_cue_registry
        from measuring_cot_monitorability.prompts.template_registry import (
            PromptTemplateRegistry, setup_default_templates,
        )
        from measuring_cot_monitorability.data_utils.core_schema import StandardizedExample

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

    # Load model
    model = load_model(args.model, args.gpu_memory_utilization, args.max_model_len)
    tokenizer = model.tokenizer

    # Build all (layer, alpha) sweep configurations
    configs = []
    for layer in args.layers:
        layer_alphas: set[float] = set()
        if args.alphas:
            layer_alphas.update(args.alphas)
        if args.use_affine_alphas and layer in affine_alphas_data:
            layer_alphas.add(round(affine_alphas_data[layer]["alpha_range"], 6))
        for alpha in sorted(layer_alphas):
            configs.append((layer, alpha))

    all_alphas_in_sweep = sorted({a for _, a in configs})
    print(f"\nSweep: {len(configs)} configs across {len(args.layers)} layers")
    print(f"Alphas in sweep: {all_alphas_in_sweep}")
    print(f"Strategy: {args.strategy}")
    print(f"Total generations: {len(configs) * len(tasks)}")

    # Save sweep config
    think_policy = think_tag_policy(args.model, args.think_mode)
    sweep_config = {
        "model": args.model,
        "strategy": args.strategy,
        "layers": args.layers,
        "alphas": all_alphas_in_sweep,
        "use_affine_alphas": args.use_affine_alphas,
        "n_tasks": len(tasks),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "think_mode": args.think_mode,
        "strip_existing_think_tag": think_policy["strip_existing"],
        "add_think_tag": think_policy["append_explicit"],
        "expect_think_tags": think_policy["expect_tags"],
        "seed": args.seed,
        "vectors_dir": str(args.vectors_dir),
        "cue_name": args.cue_name,
        "cue_target": args.cue_target,
    }
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    for layer, alpha in configs:
        config_name = f"layer_{layer}_alpha_{alpha}"
        out_path = output_dir / f"{config_name}.jsonl"

        # Skip if output file exists with correct number of records
        if out_path.exists():
            with open(out_path) as f:
                existing_count = sum(1 for _ in f)
            if existing_count == len(tasks):
                print(f"\nSkipping {config_name}: already has {existing_count} records")
                continue
            else:
                print(f"\nRedoing {config_name}: has {existing_count} records, expected {len(tasks)}")

        print(f"\n{'='*60}")
        print(f"Config: layer={layer}, alpha={alpha}, strategy={args.strategy}")
        print(f"Output: {out_path}")
        print(f"{'='*60}")

        steering_vector = vectors[layer].float().cuda()

        n_written = 0
        with open(out_path, "w") as f:
            for batch_start in tqdm(
                range(0, len(tasks), args.batch_size),
                desc=config_name,
                total=(len(tasks) + args.batch_size - 1) // args.batch_size,
            ):
                batch_tasks = tasks[batch_start:batch_start + args.batch_size]
                if cue_obj is not None:
                    cue_results = [
                        build_cued_prompt(
                            tokenizer, t, cue_obj, cue_template, args.cue_target,
                            strip_existing_think_tag=sweep_config["strip_existing_think_tag"],
                            append_think_tag=sweep_config["add_think_tag"],
                        )
                        for t in batch_tasks
                    ]
                    prompts = [r[0] for r in cue_results]
                    batch_cue_texts = [r[1] for r in cue_results]
                    batch_cue_targets = [r[2] for r in cue_results]
                else:
                    prompts = [
                        build_chat_prompt(
                            tokenizer, t["question"], t.get("choices", []),
                            strip_existing_think_tag=sweep_config["strip_existing_think_tag"],
                            append_think_tag=sweep_config["add_think_tag"],
                        )
                        for t in batch_tasks
                    ]
                    batch_cue_texts = [None] * len(batch_tasks)
                    batch_cue_targets = [None] * len(batch_tasks)

                t0 = time.time()
                try:
                    responses = generate_steered_batch(
                        model=model,
                        prompts=prompts,
                        steering_vector=steering_vector,
                        layer=layer,
                        alpha=alpha,
                        strategy=args.strategy,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        presence_penalty=args.presence_penalty,
                        repetition_penalty=args.repetition_penalty,
                    )
                except Exception as e:
                    print(f"  Error on batch starting at {batch_start}: {e}")
                    continue
                elapsed = time.time() - t0

                for task, prompt, response, cue_text, cue_target_letter in zip(
                    batch_tasks, prompts, responses, batch_cue_texts, batch_cue_targets
                ):
                    think_content, final_answer = extract_think_and_answer(
                        response,
                        expect_think_tags=sweep_config["expect_think_tags"],
                    )

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
                        "generation_config": sweep_config,
                        "elapsed_seconds": round(elapsed / len(batch_tasks), 2),
                        "steering_layer": layer,
                        "steering_alpha": alpha,
                        "steering_strategy": args.strategy,
                    }

                    if cue_obj is not None:
                        record["has_cue"] = True
                        record["cue_name"] = args.cue_name
                        record["cue_text"] = cue_text
                        record["cue_target_letter"] = cue_target_letter
                        record["judge_prompt"] = cue_obj.judge_prompt

                    record["degenerate_reason"] = classify_degenerate(
                        record, args.max_new_tokens,
                    )

                    f.write(json.dumps(record) + "\n")
                    n_written += 1
                f.flush()

        print(f"  Wrote {n_written} traces to {out_path}")

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()

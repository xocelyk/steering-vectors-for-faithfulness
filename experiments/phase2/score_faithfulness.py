"""
Score CoT traces for faithfulness (cue acknowledgment) and correctness.

Reads JSONL traces that include cue metadata and runs two scorers:
  1. cue_aware_adaptive_scorer: faithfulness score (does CoT acknowledge the cue)
  2. model_graded_fact: correctness score (does the final answer match the target)

Traces must include cue metadata fields: has_cue, cue_name, cue_text, judge_prompt.

Supports two modes:
  - Single file: pass a JSONL file path to --input
  - Directory (steered traces): pass a directory containing layer_*_alpha_*.jsonl files
    to --input. Scores each file and produces a summary CSV.

Usage:
    # Score a single trace file
    uv run python experiments/phase2/score_faithfulness.py \
        --input experiments/phase2/results/traces/cued_traces.jsonl

    # Score all steered trace files in a directory
    uv run python experiments/phase2/score_faithfulness.py \
        --input experiments/phase3/results/steered_traces_cued/

Output JSONL includes both faithfulness_score (0 or 1) and correctness_score (1/0).
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
from steering_vectors_for_faithfulness.config import MONITORABILITY_SRC

if str(MONITORABILITY_SRC) not in sys.path:
    sys.path.insert(0, str(MONITORABILITY_SRC))

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
from inspect_ai.solver import solver
from inspect_ai._util.content import ContentReasoning, ContentText

from inspect_ai.scorer import model_graded_fact, scorer, accuracy, stderr, Scorer
from inspect_ai.scorer._metric import Score
from inspect_ai.scorer._target import Target
from inspect_ai.solver._task_state import TaskState

from measuring_cot_monitorability.scorers import cue_aware_adaptive_scorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score CoT traces for faithfulness (cue acknowledgment)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a traces JSONL file, or a directory containing "
             "layer_*_alpha_*.jsonl steered trace files",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for scored output JSONL (single-file mode) or directory for "
             "scored files + summary CSV (directory mode). Default: auto-generated.",
    )
    parser.add_argument(
        "--judge_model", type=str, default="openai/gpt-5-nano",
        help="Inspect model identifier for the judge (e.g., openai/gpt-5-nano)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50,
        help="Number of samples to score per Inspect eval batch",
    )
    parser.add_argument(
        "--log_dir", type=str,
        default=str(SCRIPT_DIR / "results" / "inspect_logs_faithfulness"),
        help="Directory for Inspect eval logs",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load data and print first sample without calling the judge",
    )
    parser.add_argument(
        "--skip_scored", action="store_true", default=True,
        help="Skip records that already have faithfulness_score",
    )
    return parser.parse_args()


def load_traces(jsonl_path: str) -> list[dict]:
    """Load trace records from JSONL."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@solver
def precomputed_solver(generations: dict[str, tuple[str, str]]):
    """Solver that injects precomputed think_content and final_answer."""
    async def solve(state, generate):
        think_content, final_answer = generations.get(str(state.sample_id), ("", ""))

        content_parts = []
        if think_content:
            content_parts.append(ContentReasoning(reasoning=think_content))
        content_parts.append(ContentText(text=final_answer if final_answer else think_content))

        state.output = ModelOutput(
            model="precomputed-trace",
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=content_parts),
                stop_reason="stop",
            )],
        )
        state.messages.append(ChatMessageAssistant(content=content_parts))
        return state
    return solve


def build_samples(records: list[dict]) -> tuple[list[Sample], dict[str, tuple[str, str]]]:
    """Build Inspect Samples with cue metadata and a generation mapping."""
    samples = []
    generations = {}

    for record in records:
        task_id = record["task_id"]
        think_content = record.get("think_content", "")
        final_answer = record.get("final_answer", "")
        question = record.get("question", "")
        target = record.get("target", "")
        choices = record.get("choices", [])

        # Must have cue metadata
        if not record.get("has_cue"):
            continue

        # Format question with choices for the correctness judge
        input_text = question
        if choices:
            input_text += "\n"
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, choice in enumerate(choices):
                if i < len(letters):
                    input_text += f"\n{letters[i]}. {choice}"

        samples.append(Sample(
            id=task_id,
            input=input_text,
            target=target,
            metadata={
                "has_cue": True,
                "cue": record.get("cue_text", ""),
                "cue_name": record.get("cue_name", ""),
                "judge_prompt": record.get("judge_prompt", ""),
                "modifies_choice_formatting": record.get("modifies_choice_formatting", False),
                "embeds_metadata": record.get("embeds_metadata", False),
            },
        ))
        generations[task_id] = (think_content, final_answer)

    return samples, generations


def run_scoring_batch(
    records: list[dict],
    judge_model: str,
    log_dir: str,
    batch_name: str,
) -> dict[str, dict[str, float]]:
    """Score a batch of records and return {task_id: {faithfulness: ..., correctness: ...}}."""
    samples, generations = build_samples(records)

    if not samples:
        return {}

    inner_faithfulness = cue_aware_adaptive_scorer(model=judge_model)
    inner_correctness = model_graded_fact(model=judge_model, partial_credit=False)

    @scorer(metrics=[accuracy(), stderr()])
    def faithfulness() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return await inner_faithfulness(state, target)
        return score

    @scorer(metrics=[accuracy(), stderr()])
    def correctness() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return await inner_correctness(state, target)
        return score

    faithfulness_scorer = faithfulness()
    correctness_scorer = correctness()

    task = Task(
        dataset=samples,
        solver=precomputed_solver(generations),
        scorer=[faithfulness_scorer, correctness_scorer],
        fail_on_error=False,
    )

    batch_log_dir = os.path.join(log_dir, batch_name)
    os.makedirs(batch_log_dir, exist_ok=True)

    log = inspect_eval(
        task,
        model=judge_model,
        log_dir=batch_log_dir,
    )

    scores = {}
    if log and log[0].samples:
        for sample in log[0].samples:
            sample_id = sample.id
            if not sample.scores:
                continue
            entry = {}
            for scorer_name, score_obj in sample.scores.items():
                if scorer_name == "faithfulness":
                    if score_obj.value is not None:
                        entry["faithfulness"] = int(score_obj.value)
                elif scorer_name == "correctness":
                    grade = score_obj.value
                    if grade == "C":
                        entry["correctness"] = 1.0
                    else:
                        entry["correctness"] = 0.0
            if entry:
                scores[sample_id] = entry

    return scores


def discover_trace_files(traces_dir: str) -> list[Path]:
    """Find all layer_*_alpha_*.jsonl files, sorted by layer then alpha."""
    pattern = re.compile(r"layer_(\d+)_alpha_([\d.]+)\.jsonl$")
    files = []
    for p in Path(traces_dir).glob("layer_*_alpha_*.jsonl"):
        m = pattern.match(p.name)
        if m:
            files.append((int(m.group(1)), float(m.group(2)), p))
    files.sort()
    return [f[2] for f in files]


def parse_config_from_filename(name: str) -> tuple[int, float]:
    """Extract (layer, alpha) from a filename like layer_8_alpha_1.0.jsonl."""
    m = re.match(r"layer_(\d+)_alpha_([\d.]+)\.jsonl$", name)
    if not m:
        raise ValueError(f"Cannot parse config from filename: {name}")
    return int(m.group(1)), float(m.group(2))


def is_fully_scored(records: list[dict]) -> bool:
    """Check if every record has both faithfulness_score and correctness_score."""
    return all(
        "faithfulness_score" in r and "correctness_score" in r
        for r in records
    )


def build_summary_csv(output_dir: str, scored_files: list[Path]) -> Path:
    """Read all scored files and write a summary CSV with per-config metrics."""
    rows = []
    for path in sorted(scored_files):
        layer, alpha = parse_config_from_filename(path.name)
        records = load_traces(str(path))

        f_scores = [r["faithfulness_score"] for r in records if "faithfulness_score" in r]
        c_scores = [r["correctness_score"] for r in records if "correctness_score" in r]
        n_no_answer = sum(1 for r in records if not r.get("final_answer", "").strip())

        strategy = records[0].get("steering_strategy", "") if records else ""

        rows.append({
            "layer": layer,
            "alpha": alpha,
            "strategy": strategy,
            "n_tasks": len(records),
            "n_scored_faithfulness": len(f_scores),
            "n_scored_correctness": len(c_scores),
            "mean_faithfulness": f"{sum(f_scores) / len(f_scores):.4f}" if f_scores else "",
            "mean_correctness": f"{sum(c_scores) / len(c_scores):.4f}" if c_scores else "",
            "n_no_answer": n_no_answer,
        })

    csv_path = Path(output_dir) / "faithfulness_scoring_summary.csv"
    fieldnames = [
        "layer", "alpha", "strategy", "n_tasks",
        "n_scored_faithfulness", "n_scored_correctness",
        "mean_faithfulness", "mean_correctness", "n_no_answer",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def score_directory(args):
    """Score all steered trace files in a directory and produce a summary CSV."""
    trace_files = discover_trace_files(args.input)
    print(f"Found {len(trace_files)} steered trace files in {args.input}")
    for f in trace_files:
        print(f"  {f.name}")

    output_dir = args.output or str(SCRIPT_DIR / "results" / "scored_steered_traces_faithfulness")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    scored_files: list[Path] = []

    for i, trace_path in enumerate(trace_files):
        config_name = trace_path.stem
        layer, alpha = parse_config_from_filename(trace_path.name)
        output_path = Path(output_dir) / trace_path.name

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(trace_files)}] {config_name}  (layer={layer}, alpha={alpha})")
        print(f"{'='*60}")

        records = load_traces(str(trace_path))
        if not records:
            print("  Empty file, skipping.")
            continue

        if output_path.exists():
            scored_records = load_traces(str(output_path))
            if len(scored_records) == len(records) and is_fully_scored(scored_records):
                print(f"  Already scored ({len(scored_records)} records), skipping.")
                scored_files.append(output_path)
                continue
            else:
                print(f"  Scored file incomplete or mismatched, re-scoring.")

        # Filter degenerate traces
        non_degenerate = [r for r in records if not r.get("degenerate_reason")]
        degenerate = [r for r in records if r.get("degenerate_reason")]
        if degenerate:
            print(f"  Skipping {len(degenerate)} degenerate traces")

        # Filter to records with cue metadata
        with_cue = [r for r in non_degenerate if r.get("has_cue")]
        without_cue = len(non_degenerate) - len(with_cue)
        if without_cue:
            print(f"  Skipping {without_cue} records without cue metadata")

        record_by_id = {r["task_id"]: r for r in records}
        n_batches = (len(with_cue) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            batch_records = with_cue[start : start + args.batch_size]

            print(f"  Batch {batch_idx + 1}/{n_batches} ({len(batch_records)} samples)")

            batch_scores = run_scoring_batch(
                batch_records,
                args.judge_model,
                args.log_dir,
                f"{config_name}_batch_{batch_idx:04d}",
            )

            for task_id, score_dict in batch_scores.items():
                if task_id in record_by_id:
                    if "faithfulness" in score_dict:
                        record_by_id[task_id]["faithfulness_score"] = score_dict["faithfulness"]
                    if "correctness" in score_dict:
                        record_by_id[task_id]["correctness_score"] = score_dict["correctness"]

            with open(output_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        scored_files.append(output_path)
        scored = [r for r in records if "faithfulness_score" in r]
        f_vals = [r["faithfulness_score"] for r in scored]
        c_vals = [r["correctness_score"] for r in scored if "correctness_score" in r]
        if f_vals and c_vals:
            print(f"  Done: faithfulness={sum(f_vals)/len(f_vals):.3f}, correctness={sum(c_vals)/len(c_vals):.1%}")

    if scored_files:
        csv_path = build_summary_csv(output_dir, scored_files)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Scored {len(scored_files)} files")
        print(f"Summary CSV: {csv_path}")

        with open(csv_path) as f:
            print(f.read())
    else:
        print("\nNo files were scored.")


def main():
    args = parse_args()

    # Directory mode
    if os.path.isdir(args.input):
        score_directory(args)
        return

    records = load_traces(args.input)
    print(f"Loaded {len(records)} traces from {args.input}")

    # Filter degenerate traces
    degenerate = [r for r in records if r.get("degenerate_reason")]
    non_degenerate = [r for r in records if not r.get("degenerate_reason")]
    if degenerate:
        print(f"  Skipping {len(degenerate)} degenerate traces")

    # Filter to records with cue metadata
    with_cue = [r for r in non_degenerate if r.get("has_cue")]
    without_cue = len(non_degenerate) - len(with_cue)
    if without_cue:
        print(f"  Skipping {without_cue} records without cue metadata")

    if args.skip_scored:
        unscored = [r for r in with_cue
                     if "faithfulness_score" not in r or "correctness_score" not in r]
        print(f"  {len(with_cue) - len(unscored)} already scored, {len(unscored)} to score")
        records_to_score = unscored
    else:
        records_to_score = with_cue

    if args.output:
        output_path = args.output
    else:
        os.makedirs(SCRIPT_DIR / "results" / "scored_traces_faithfulness", exist_ok=True)
        input_name = Path(args.input).stem
        output_path = str(
            SCRIPT_DIR / "results" / "scored_traces_faithfulness" / f"{input_name}_faithfulness_scored.jsonl"
        )

    if args.dry_run:
        print("\n--- Dry run: first sample ---")
        if records_to_score:
            r = records_to_score[0]
            print(f"  task_id: {r['task_id']}")
            print(f"  cue_name: {r.get('cue_name', 'N/A')}")
            print(f"  cue_text: {r.get('cue_text', 'N/A')}")
            print(f"  question (first 200): {r.get('question', '')[:200]}")
            print(f"  think_content (first 300): {r.get('think_content', '')[:300]}")
        return

    if not records_to_score:
        print("All records already scored (or no records with cue metadata)!")
        return

    os.makedirs(args.log_dir, exist_ok=True)

    all_records = load_traces(args.input)
    record_by_id = {r["task_id"]: r for r in all_records}

    n_batches = (len(records_to_score) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        batch_records = records_to_score[start:end]

        print(f"\n{'='*60}")
        print(f"Scoring batch {batch_idx + 1}/{n_batches} ({len(batch_records)} samples)")
        print(f"{'='*60}")

        batch_scores = run_scoring_batch(
            batch_records,
            args.judge_model,
            args.log_dir,
            f"batch_{batch_idx:04d}",
        )

        print(f"  Scored {len(batch_scores)} samples in this batch")

        for task_id, score_dict in batch_scores.items():
            if task_id in record_by_id:
                if "faithfulness" in score_dict:
                    record_by_id[task_id]["faithfulness_score"] = score_dict["faithfulness"]
                if "correctness" in score_dict:
                    record_by_id[task_id]["correctness_score"] = score_dict["correctness"]

        with open(output_path, "w") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")

        print(f"  Saved progress to {output_path}")

    scored_records = list(record_by_id.values())

    with_faithfulness = [r for r in scored_records if "faithfulness_score" in r]
    with_correctness = [r for r in scored_records if "correctness_score" in r]
    f_scores = [r["faithfulness_score"] for r in with_faithfulness]
    c_scores = [r["correctness_score"] for r in with_correctness]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(scored_records)}")
    print(f"Records with faithfulness_score: {len(with_faithfulness)}")
    print(f"Records with correctness_score: {len(with_correctness)}")
    if f_scores:
        print(f"\nFaithfulness - mean: {sum(f_scores)/len(f_scores):.3f} "
              f"({sum(f_scores)}/{len(f_scores)} acknowledged cue)")
    if c_scores:
        acc = sum(c_scores) / len(c_scores)
        print(f"Correctness - accuracy: {acc:.1%} ({int(sum(c_scores))}/{len(c_scores)})")

    if f_scores:
        by_cue = {}
        for r in with_faithfulness:
            cue = r.get("cue_name", "unknown")
            if cue not in by_cue:
                by_cue[cue] = {"faithfulness": [], "correctness": []}
            by_cue[cue]["faithfulness"].append(r["faithfulness_score"])
            if "correctness_score" in r:
                by_cue[cue]["correctness"].append(r["correctness_score"])

        print("\nPer-cue breakdown:")
        for cue, cue_data in sorted(by_cue.items()):
            f_mean = sum(cue_data["faithfulness"]) / len(cue_data["faithfulness"])
            c_vals = cue_data["correctness"]
            c_str = f", accuracy={sum(c_vals)/len(c_vals):.1%}" if c_vals else ""
            print(f"  {cue}: n={len(cue_data['faithfulness'])}, faithfulness={f_mean:.3f}{c_str}")

    print(f"\nScored output written to: {output_path}")


if __name__ == "__main__":
    main()

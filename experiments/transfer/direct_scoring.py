"""Score transfer traces with regular OpenAI Chat Completions.

This is the non-Batch-API companion to ``batch_scoring.py``. It uses the same
prompt builders and output schemas, but sends requests directly and writes
scored mirror JSONLs incrementally under ``experiments/transfer/runs_scored``.

Examples:

    # Score uncued traces for correctness + degeneracy.
    uv run --no-sync python experiments/transfer/direct_scoring.py \
      --glob "**/uncued/*.jsonl" --scorers correctness degenerate

    # Score all cued traces for faithfulness only.
    uv run --no-sync python experiments/transfer/direct_scoring.py \
      --glob "**/stanford/*.jsonl" "**/xml/*.jsonl" "**/grader/*.jsonl" "**/insider/*.jsonl" \
      --scorers faithfulness
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from batch_scoring import (
    DEFAULT_MODEL,
    DEFAULT_SCORED_DIR,
    DEFAULT_SOURCE_DIR,
    SCORING_VERSION,
    build_batch_request,
    load_openai_client,
    normalize_score_payload,
    utc_now,
)
from common import read_jsonl


SCORER_FIELDS = {
    "correctness": ("correctness_score", "correctness_explanation"),
    "degenerate": ("degenerate_score", "degenerate_explanation"),
    "faithfulness": ("faithfulness_score", "faithfulness_explanation"),
}


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]], attempts: int = 5) -> None:
    """Write JSONL atomically so interrupts do not corrupt an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(attempts):
        tmp_name: str | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                text=True,
            )
            with os.fdopen(fd, "w") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
            return
        except Exception as exc:  # noqa: BLE001 - retry flaky network filesystems.
            last_error = exc
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            if attempt + 1 < attempts:
                sleep_for = min(2 ** attempt, 10)
                print(f"Atomic write failed for {path} ({exc}); retrying in {sleep_for}s")
                time.sleep(sleep_for)
    raise RuntimeError(f"Atomic write failed for {path} after {attempts} attempts: {last_error}") from last_error


def write_jsonl_atomic_best_effort(path: Path, records: list[dict[str, Any]]) -> None:
    """Flush progress without crashing the scoring run on transient I/O errors."""
    try:
        write_jsonl_atomic(path, records)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not flush progress to {path}: {exc}")


def write_jsonl_atomic_required(path: Path, records: list[dict[str, Any]]) -> None:
    """Final file write. Raise if this fails after retries."""
    write_jsonl_atomic(path, records)


@dataclass(frozen=True)
class ScoreJob:
    file_path: Path
    rel_path: Path
    row_index: int
    task_id: str
    scorer: str
    custom_id: str
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--scored_dir", type=Path, default=DEFAULT_SCORED_DIR)
    parser.add_argument(
        "--glob",
        type=str,
        nargs="+",
        default=["**/*.jsonl"],
        help="Glob pattern(s) relative to source_dir for trace files.",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        choices=["correctness", "degenerate", "faithfulness"],
        default=None,
        help=(
            "Scorers to run. Default: correctness + degenerate for all rows, "
            "plus faithfulness for cued rows."
        ),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_response_chars", type=int, default=60_000)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=5.0)
    parser.add_argument("--request_delay", type=float, default=0.0)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="Maximum requests to score.")
    parser.add_argument("--missing_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def discover_trace_paths(source_dir: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(source_dir.glob(pattern)):
            if path in seen or not path.is_file() or path.name.endswith("_config.json"):
                continue
            seen.add(path)
            paths.append(path)
    return paths


def scorer_list_for_record(record: dict[str, Any], requested: list[str] | None) -> list[str]:
    if requested is None:
        scorers = ["correctness", "degenerate"]
        if record.get("has_cue"):
            scorers.append("faithfulness")
        return scorers
    return [
        scorer for scorer in requested
        if scorer != "faithfulness" or record.get("has_cue")
    ]


def has_score(record: dict[str, Any], scorer: str) -> bool:
    score_field, explanation_field = SCORER_FIELDS[scorer]
    return score_field in record and explanation_field in record


def output_records_for(source_records: list[dict[str, Any]], scored_path: Path) -> list[dict[str, Any]]:
    if not scored_path.exists():
        return [dict(record) for record in source_records]

    scored_records = read_jsonl(scored_path)
    if len(scored_records) != len(source_records):
        return [dict(record) for record in source_records]

    for source_record, scored_record in zip(source_records, scored_records, strict=True):
        if source_record.get("task_id") != scored_record.get("task_id"):
            return [dict(record) for record in source_records]
    return scored_records


def make_jobs_for_file(
    source_path: Path,
    source_dir: Path,
    scored_dir: Path,
    requested_scorers: list[str] | None,
    missing_only: bool,
) -> tuple[Path, list[dict[str, Any]], list[ScoreJob]]:
    rel_path = source_path.relative_to(source_dir)
    scored_path = scored_dir / rel_path
    source_records = read_jsonl(source_path)
    output_records = output_records_for(source_records, scored_path)
    jobs: list[ScoreJob] = []

    for row_index, record in enumerate(output_records):
        for scorer in scorer_list_for_record(record, requested_scorers):
            if missing_only and has_score(record, scorer):
                continue
            custom_id = f"direct-{scorer}-{row_index:06d}-{record.get('task_id', 'unknown')}"
            jobs.append(
                ScoreJob(
                    file_path=source_path,
                    rel_path=rel_path,
                    row_index=row_index,
                    task_id=str(record.get("task_id", "")),
                    scorer=scorer,
                    custom_id=custom_id,
                    record=record,
                )
            )
    return scored_path, output_records, jobs


def call_openai_for_job(client: Any, job: ScoreJob, args: argparse.Namespace) -> dict[str, Any]:
    request, truncated = build_batch_request(
        job.custom_id,
        job.record,
        args.model,
        args.max_response_chars,
        job.scorer,
    )
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            if args.request_delay:
                time.sleep(args.request_delay)
            response = client.chat.completions.create(**request["body"])
            content = response.choices[0].message.content or ""
            payload = normalize_score_payload(json.loads(content))
            return {
                "payload": payload,
                "request_id": getattr(response, "id", None),
                "custom_id": job.custom_id,
                "response_truncated": truncated,
            }
        except Exception as exc:  # noqa: BLE001 - keep scoring robust for long jobs.
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(f"{job.custom_id} failed after retries: {last_error}") from last_error


def apply_result(record: dict[str, Any], job: ScoreJob, result: dict[str, Any], model: str) -> None:
    record.update(result["payload"])
    request_ids = dict(record.get("judge_request_ids") or {})
    custom_ids = dict(record.get("judge_custom_ids") or {})
    request_ids[job.scorer] = result.get("request_id")
    custom_ids[job.scorer] = result.get("custom_id")
    record.update({
        "scoring_version": SCORING_VERSION,
        "scoring_method": "direct_chat_completions",
        "judge_model": model,
        "judge_request_ids": request_ids,
        "judge_custom_ids": custom_ids,
        "judge_scored_at": utc_now(),
        "judge_response_truncated": bool(
            record.get("judge_response_truncated") or result.get("response_truncated")
        ),
    })


def score_file(
    client: Any,
    source_path: Path,
    source_dir: Path,
    scored_dir: Path,
    args: argparse.Namespace,
    remaining_limit: int | None,
) -> int:
    scored_path, output_records, jobs = make_jobs_for_file(
        source_path,
        source_dir,
        scored_dir,
        args.scorers,
        args.missing_only,
    )
    if remaining_limit is not None:
        jobs = jobs[:remaining_limit]

    print(f"\n=== {source_path.relative_to(source_dir)} ===")
    print(f"Rows: {len(output_records)}; requests to score: {len(jobs)}")

    if args.dry_run or not jobs:
        return len(jobs)

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_job = {
            executor.submit(call_openai_for_job, client, job, args): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                record = output_records[job.row_index]
                errors = dict(record.get("judge_errors") or {})
                errors[job.scorer] = {"custom_id": job.custom_id, "error": str(exc)}
                record["judge_errors"] = errors
                print(f"ERROR {job.custom_id}: {exc}")
            else:
                apply_result(output_records[job.row_index], job, result, args.model)
                print(f"OK {job.scorer} {job.task_id}")
            completed += 1
            if completed % max(1, args.flush_every) == 0:
                write_jsonl_atomic_best_effort(scored_path, output_records)

    write_jsonl_atomic_required(scored_path, output_records)
    print(f"Wrote {scored_path}")
    return len(jobs)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    scored_dir = args.scored_dir.resolve()
    trace_paths = discover_trace_paths(source_dir, args.glob)
    print(f"Found {len(trace_paths)} trace files")
    if args.dry_run:
        print("Dry run: no API requests will be sent.")

    client = None if args.dry_run else load_openai_client()
    total_jobs = 0
    for source_path in trace_paths:
        remaining = None if args.limit is None else max(args.limit - total_jobs, 0)
        if remaining == 0:
            break
        total_jobs += score_file(client, source_path, source_dir, scored_dir, args, remaining)

    print(f"\nTotal requests considered: {total_jobs}")


if __name__ == "__main__":
    main()

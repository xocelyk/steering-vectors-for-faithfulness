"""
Analyze scored traces and contrastive datasets with detailed metrics.

Provides multiple analysis modes:
  - summary:     Overall score distributions and dataset statistics
  - compare:     Side-by-side comparison of D+ vs D- splits
  - correctness: Accuracy breakdown by dataset, verbosity quartile, and length
  - length:      Token/character length analysis and correlation with verbosity
  - balance:     Dataset balance, accuracy parity, and trimming simulation
  - all:         Run all analyses

Usage:
    # Run all analyses (auto-detects scored traces)
    uv run python experiments/phase2/analyze_dataset.py \
        --contrastive_dir experiments/phase2/results/contrastive --mode all

    # Simulate trimming strategies
    uv run python experiments/phase2/analyze_dataset.py \
        --contrastive_dir experiments/phase2/results/contrastive --mode balance

    # Trim D+ with weighted strategy and save to contrastive/trimmed/
    uv run python experiments/phase2/analyze_dataset.py \
        --contrastive_dir experiments/phase2/results/contrastive \
        --mode balance --save_trimmed --trim_strategy weighted

    # Trim D- instead, using accuracy strategy, to 100 examples
    uv run python experiments/phase2/analyze_dataset.py \
        --contrastive_dir experiments/phase2/results/contrastive \
        --mode balance --save_trimmed --trim_split d_minus --trim_strategy accuracy --trim_to 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze scored traces and contrastive datasets"
    )
    parser.add_argument(
        "--scored", type=str, default=None,
        help="Path to scored JSONL (from score_verbosity.py)",
    )
    parser.add_argument(
        "--contrastive_dir", type=str, default=None,
        help="Path to contrastive split directory containing d_plus.jsonl and d_minus.jsonl",
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["summary", "compare", "correctness", "length", "balance", "all"],
        help="Analysis mode",
    )
    parser.add_argument(
        "--trim_to", type=int, default=None,
        help="Target size after trimming (default: match size of the other split)",
    )
    parser.add_argument(
        "--trim_split", type=str, default="d_plus",
        choices=["d_plus", "d_minus"],
        help="Which split to trim (default: d_plus)",
    )
    parser.add_argument(
        "--trim_strategy", type=str, default="weighted",
        choices=["weighted", "accuracy", "length", "random"],
        help="Trimming strategy: weighted (acc+len score), accuracy (remove highest), "
             "length (remove longest), random",
    )
    parser.add_argument(
        "--save_trimmed", action="store_true",
        help="Write trimmed split to results/contrastive/trimmed/",
    )
    parser.add_argument(
        "--acc_weight", type=float, default=0.6,
        help="Weight for accuracy in weighted strategy (default: 0.6)",
    )
    parser.add_argument(
        "--len_weight", type=float, default=0.4,
        help="Weight for normalized token length in weighted strategy (default: 0.4)",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_row(label: str, *values, fmt=".3f"):
    formatted = "".join(f"{v:>14{fmt}}" if v is not None else f"{'N/A':>14s}" for v in values)
    print(f"  {label:32s}{formatted}")


def print_row_str(label: str, *values):
    formatted = "".join(f"{v:>14s}" for v in values)
    print(f"  {label:32s}{formatted}")


# ---------------------------------------------------------------------------
# Analysis modes
# ---------------------------------------------------------------------------

def analyze_summary(records: list[dict]):
    """Overall score distributions."""
    print_header("SUMMARY")

    print(f"  Total records: {len(records)}")

    v_scores = [r["verbosity_score"] for r in records if "verbosity_score" in r]
    c_scores = [r["correctness_score"] for r in records if "correctness_score" in r]

    if v_scores:
        print(f"\n  Verbosity scores (n={len(v_scores)}):")
        print(f"    mean={np.mean(v_scores):.3f}  std={np.std(v_scores):.3f}")
        print(f"    min={min(v_scores):.3f}  Q25={np.percentile(v_scores, 25):.3f}  "
              f"median={np.median(v_scores):.3f}  Q75={np.percentile(v_scores, 75):.3f}  "
              f"max={max(v_scores):.3f}")

        hist, bin_edges = np.histogram(v_scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        print(f"\n  Verbosity histogram:")
        for i in range(len(hist)):
            bar = "#" * int(hist[i] / max(hist) * 40) if max(hist) > 0 else ""
            print(f"    [{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}): {hist[i]:>4d} {bar}")

    if c_scores:
        accuracy = sum(c_scores) / len(c_scores)
        print(f"\n  Correctness (n={len(c_scores)}):")
        print(f"    accuracy={accuracy:.1%} ({int(sum(c_scores))}/{len(c_scores)})")

    datasets = {}
    for r in records:
        ds = r.get("dataset", "unknown")
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)
    if len(datasets) > 1:
        print(f"\n  By dataset:")
        for ds, ds_records in sorted(datasets.items()):
            print(f"    {ds}: n={len(ds_records)}")


def analyze_compare(d_plus: list[dict], d_minus: list[dict]):
    """Side-by-side D+ vs D- comparison."""
    print_header("D+ vs D- COMPARISON")

    cols = ("D+", "D-")
    print_row_str("", *cols)
    print(f"  {'-' * 60}")

    print_row("n", len(d_plus), len(d_minus), fmt="d")

    v_plus = [r["verbosity_score"] for r in d_plus if "verbosity_score" in r]
    v_minus = [r["verbosity_score"] for r in d_minus if "verbosity_score" in r]
    print_row("verbosity (mean)", np.mean(v_plus), np.mean(v_minus))
    print_row("verbosity (std)", np.std(v_plus), np.std(v_minus))

    c_plus = [r.get("correctness_score") for r in d_plus if "correctness_score" in r]
    c_minus = [r.get("correctness_score") for r in d_minus if "correctness_score" in r]
    if c_plus and c_minus:
        print_row("accuracy", sum(c_plus) / len(c_plus), sum(c_minus) / len(c_minus))

    think_plus = [len(r.get("think_content", "")) for r in d_plus]
    think_minus = [len(r.get("think_content", "")) for r in d_minus]
    print_row("think chars (mean)", np.mean(think_plus), np.mean(think_minus), fmt=".0f")

    answer_plus = [len(r.get("final_answer", "")) for r in d_plus]
    answer_minus = [len(r.get("final_answer", "")) for r in d_minus]
    print_row("answer chars (mean)", np.mean(answer_plus), np.mean(answer_minus), fmt=".0f")

    total_plus = [t + a for t, a in zip(think_plus, answer_plus)]
    total_minus = [t + a for t, a in zip(think_minus, answer_minus)]
    print_row("total chars (mean)", np.mean(total_plus), np.mean(total_minus), fmt=".0f")

    tok_plus = [r.get("num_tokens", 0) for r in d_plus]
    tok_minus = [r.get("num_tokens", 0) for r in d_minus]
    print_row("num_tokens (mean)", np.mean(tok_plus), np.mean(tok_minus), fmt=".0f")
    print_row("num_tokens (std)", np.std(tok_plus), np.std(tok_minus), fmt=".0f")

    factors_plus = [len(r.get("causal_factors", [])) for r in d_plus]
    factors_minus = [len(r.get("causal_factors", [])) for r in d_minus]
    print_row("causal_factors (mean)", np.mean(factors_plus), np.mean(factors_minus), fmt=".1f")

    # Statistical tests
    print(f"\n  Statistical tests:")
    if v_plus and v_minus:
        t_stat, p_val = stats.ttest_ind(v_plus, v_minus)
        pooled_std = np.sqrt((np.std(v_plus)**2 + np.std(v_minus)**2) / 2)
        cohens_d = (np.mean(v_plus) - np.mean(v_minus)) / pooled_std if pooled_std > 0 else float("inf")
        print(f"    Verbosity t-test: t={t_stat:.2f}, p={p_val:.2e}, Cohen's d={cohens_d:.2f}")

    if tok_plus and tok_minus:
        t_stat, p_val = stats.ttest_ind(tok_plus, tok_minus)
        print(f"    Token length t-test: t={t_stat:.2f}, p={p_val:.2e}")


def analyze_correctness(records: list[dict]):
    """Accuracy breakdown by various slices."""
    print_header("CORRECTNESS ANALYSIS")

    scored = [r for r in records if "correctness_score" in r and "verbosity_score" in r]
    if not scored:
        print("  No records with both correctness and verbosity scores.")
        return

    # Overall
    c_scores = [r["correctness_score"] for r in scored]
    print(f"  Overall accuracy: {sum(c_scores)/len(c_scores):.1%} ({int(sum(c_scores))}/{len(c_scores)})")

    # By dataset
    datasets = {}
    for r in scored:
        ds = r.get("dataset", "unknown")
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    if len(datasets) > 1:
        print(f"\n  By dataset:")
        for ds, ds_recs in sorted(datasets.items()):
            c = [r["correctness_score"] for r in ds_recs]
            print(f"    {ds}: {sum(c)/len(c):.1%} ({int(sum(c))}/{len(c)})")

    # By verbosity quartile
    v_scores = [r["verbosity_score"] for r in scored]
    q25, q50, q75 = np.percentile(v_scores, [25, 50, 75])

    print(f"\n  By verbosity quartile:")
    quartiles = [
        ("Q1 (lowest)", lambda r: r["verbosity_score"] <= q25),
        ("Q2", lambda r: q25 < r["verbosity_score"] <= q50),
        ("Q3", lambda r: q50 < r["verbosity_score"] <= q75),
        ("Q4 (highest)", lambda r: r["verbosity_score"] > q75),
    ]
    for label, pred in quartiles:
        q_recs = [r for r in scored if pred(r)]
        if q_recs:
            c = [r["correctness_score"] for r in q_recs]
            print(f"    {label}: {sum(c)/len(c):.1%} (n={len(c)})")

    # Correlation
    corr, p_val = stats.pointbiserialr(c_scores, v_scores)
    print(f"\n  Verbosity-correctness correlation: r={corr:.3f}, p={p_val:.3e}")


def analyze_length(records: list[dict]):
    """Token and character length analysis."""
    print_header("LENGTH ANALYSIS")

    scored = [r for r in records if "verbosity_score" in r]
    if not scored:
        print("  No scored records.")
        return

    think_lens = [len(r.get("think_content", "")) for r in scored]
    answer_lens = [len(r.get("final_answer", "")) for r in scored]
    total_lens = [t + a for t, a in zip(think_lens, answer_lens)]
    token_counts = [r.get("num_tokens", 0) for r in scored]
    v_scores = [r["verbosity_score"] for r in scored]

    print(f"  n={len(scored)}")
    for label, vals in [("think chars", think_lens), ("answer chars", answer_lens),
                         ("total chars", total_lens), ("num_tokens", token_counts)]:
        print(f"\n  {label}:")
        print(f"    mean={np.mean(vals):.0f}  std={np.std(vals):.0f}")
        print(f"    min={min(vals)}  median={np.median(vals):.0f}  max={max(vals)}")

    # Correlations with verbosity
    print(f"\n  Correlations with verbosity score:")
    for label, vals in [("think_content chars", think_lens), ("final_answer chars", answer_lens),
                         ("total chars", total_lens), ("num_tokens", token_counts)]:
        if len(vals) > 2 and np.std(vals) > 0:
            corr, p_val = stats.pearsonr(v_scores, vals)
            print(f"    {label}: r={corr:.3f}, p={p_val:.3e}")

    # Length percentiles
    print(f"\n  Token count percentiles:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{pct}: {np.percentile(token_counts, pct):.0f}")


def trim_split(records: list[dict], trim_to: int, strategy: str,
               acc_weight: float = 0.6, len_weight: float = 0.4) -> list[dict]:
    """Trim a list of records to trim_to size using the given strategy."""
    if len(records) <= trim_to:
        return records

    n_to_remove = len(records) - trim_to

    if strategy == "weighted":
        token_counts = [r.get("num_tokens", 0) for r in records]
        max_tokens = max(token_counts) if token_counts else 1
        min_tokens = min(token_counts) if token_counts else 0
        token_range = max_tokens - min_tokens if max_tokens > min_tokens else 1

        def weighted_score(r):
            acc = r.get("correctness_score", 0)
            norm_tokens = (r.get("num_tokens", 0) - min_tokens) / token_range
            return acc_weight * acc + len_weight * norm_tokens

        return sorted(records, key=weighted_score, reverse=True)[n_to_remove:]

    elif strategy == "accuracy":
        return sorted(records, key=lambda r: r.get("correctness_score", 0))[:trim_to]

    elif strategy == "length":
        return sorted(records, key=lambda r: -r.get("num_tokens", 0))[n_to_remove:]

    elif strategy == "random":
        return sorted(records, key=lambda r: hash(r["task_id"]))[:trim_to]

    raise ValueError(f"Unknown strategy: {strategy}")


def _print_split_stats(label: str, records: list[dict]):
    """Print one row of stats for a split."""
    c = [r.get("correctness_score", 0) for r in records]
    v = [r["verbosity_score"] for r in records]
    t = [r.get("num_tokens", 0) for r in records]
    acc = sum(c) / len(c) if c else 0
    print(f"  {label:30s} n={len(records):>4d}  acc={acc:>6.1%}  "
          f"verb={np.mean(v):>.3f}  tok={np.mean(t):>.0f}")


def analyze_balance(d_plus: list[dict], d_minus: list[dict],
                    trim_to: int | None = None, trim_split_name: str = "d_plus",
                    trim_strategy: str = "weighted",
                    acc_weight: float = 0.6, len_weight: float = 0.4):
    """Dataset balance, accuracy parity, and trimming simulation."""
    print_header("BALANCE ANALYSIS")

    # Dataset composition
    ds_plus = {}
    for r in d_plus:
        ds = r.get("dataset", "unknown")
        ds_plus[ds] = ds_plus.get(ds, 0) + 1
    ds_minus = {}
    for r in d_minus:
        ds = r.get("dataset", "unknown")
        ds_minus[ds] = ds_minus.get(ds, 0) + 1

    all_ds = sorted(set(ds_plus) | set(ds_minus))
    print(f"  Dataset composition:")
    print_row_str("", "D+", "D-", "D+ ratio")
    for ds in all_ds:
        np_ = ds_plus.get(ds, 0)
        nm = ds_minus.get(ds, 0)
        ratio = np_ / (np_ + nm) if (np_ + nm) > 0 else 0
        print(f"    {ds:26s} {np_:>6d} {nm:>6d} {ratio:>10.1%}")

    # Accuracy parity
    c_plus = [r["correctness_score"] for r in d_plus if "correctness_score" in r]
    c_minus = [r["correctness_score"] for r in d_minus if "correctness_score" in r]

    if c_plus and c_minus:
        acc_plus = sum(c_plus) / len(c_plus)
        acc_minus = sum(c_minus) / len(c_minus)
        gap = abs(acc_plus - acc_minus)
        print(f"\n  Accuracy parity:")
        print(f"    D+: {acc_plus:.1%}  D-: {acc_minus:.1%}  gap: {gap:.1%}  balanced: {gap < 0.15}")

    # Determine which split to trim and what the default target size is
    if trim_split_name == "d_plus":
        to_trim, other = d_plus, d_minus
        trim_label, other_label = "D+", "D-"
    else:
        to_trim, other = d_minus, d_plus
        trim_label, other_label = "D-", "D+"

    if trim_to is None:
        trim_to = len(other)

    print(f"\n  Trimming {trim_label} ({len(to_trim)} -> {trim_to}):")
    print(f"  {'-' * 70}")

    _print_split_stats(f"{trim_label} (original)", to_trim)

    all_strategies = ["weighted", "accuracy", "length", "random"]
    results = {}
    for strat in all_strategies:
        trimmed = trim_split(to_trim, trim_to, strat,
                             acc_weight=acc_weight, len_weight=len_weight)
        label = strat
        if strat == "weighted":
            label = f"weighted ({acc_weight}acc+{len_weight}len)"
        marker = " <--" if strat == trim_strategy else ""
        _print_split_stats(f"  {label}{marker}", trimmed)
        results[strat] = trimmed

    print()
    _print_split_stats(f"{other_label} (reference)", other)

    return results[trim_strategy]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    scored_records = None
    d_plus = None
    d_minus = None

    if args.scored:
        scored_records = load_jsonl(args.scored)
        print(f"Loaded {len(scored_records)} scored records from {args.scored}")

    script_dir = Path(__file__).resolve().parent

    if args.contrastive_dir:
        contrastive_dir = Path(args.contrastive_dir)
    else:
        contrastive_dir = None

    # Load contrastive splits
    if contrastive_dir:
        dp = contrastive_dir / "d_plus.jsonl"
        dm = contrastive_dir / "d_minus.jsonl"
        if dp.exists() and dm.exists():
            d_plus = load_jsonl(str(dp))
            d_minus = load_jsonl(str(dm))
            print(f"Loaded D+ ({len(d_plus)}) and D- ({len(d_minus)}) from {contrastive_dir}")
        else:
            print(f"WARNING: d_plus.jsonl or d_minus.jsonl not found in {contrastive_dir}")

    if scored_records is None and (args.mode in ("summary", "correctness", "length", "all")):
        default_scored = sorted(script_dir.glob("results/scored_traces/*_scored.jsonl"))
        if default_scored:
            scored_records = load_jsonl(str(default_scored[-1]))
            print(f"Auto-loaded {len(scored_records)} scored records from {default_scored[-1].name}")

    mode = args.mode

    if mode in ("summary", "all") and scored_records:
        analyze_summary(scored_records)

    if mode in ("compare", "all") and d_plus and d_minus:
        analyze_compare(d_plus, d_minus)

    if mode in ("correctness", "all") and scored_records:
        analyze_correctness(scored_records)

    if mode in ("length", "all") and scored_records:
        analyze_length(scored_records)

    balance_kwargs = dict(
        trim_to=args.trim_to,
        trim_split_name=args.trim_split,
        trim_strategy=args.trim_strategy,
        acc_weight=args.acc_weight,
        len_weight=args.len_weight,
    )

    trimmed = None
    if mode in ("balance", "all") and d_plus and d_minus:
        trimmed = analyze_balance(d_plus, d_minus, **balance_kwargs)

    if args.save_trimmed:
        if trimmed is None and d_plus and d_minus:
            trimmed = analyze_balance(d_plus, d_minus, **balance_kwargs)

        if trimmed:
            import os
            base_dir = contrastive_dir if contrastive_dir else script_dir / "results" / "contrastive"
            trimmed_dir = base_dir / "trimmed"
            os.makedirs(trimmed_dir, exist_ok=True)

            if args.trim_split == "d_plus":
                trimmed_data, unchanged_data = trimmed, d_minus
                original_size = len(d_plus)
            else:
                trimmed_data, unchanged_data = trimmed, d_plus
                original_size = len(d_minus)

            with open(trimmed_dir / "d_plus.jsonl", "w") as f:
                data = trimmed_data if args.trim_split == "d_plus" else unchanged_data
                for r in data:
                    f.write(json.dumps(r) + "\n")

            with open(trimmed_dir / "d_minus.jsonl", "w") as f:
                data = trimmed_data if args.trim_split == "d_minus" else unchanged_data
                for r in data:
                    f.write(json.dumps(r) + "\n")

            strategy_desc = args.trim_strategy
            if args.trim_strategy == "weighted":
                strategy_desc = f"weighted ({args.acc_weight}acc+{args.len_weight}len)"

            summary = {
                "trimmed_split": args.trim_split,
                "strategy": strategy_desc,
                "original_size": original_size,
                "trimmed_size": len(trimmed_data),
                "other_size": len(unchanged_data),
                "acc_weight": args.acc_weight if args.trim_strategy == "weighted" else None,
                "len_weight": args.len_weight if args.trim_strategy == "weighted" else None,
            }
            with open(trimmed_dir / "trim_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            trim_label = "D+" if args.trim_split == "d_plus" else "D-"
            other_label = "D-" if args.trim_split == "d_plus" else "D+"
            print(f"\nSaved trimmed split to {trimmed_dir}/")
            print(f"  d_plus.jsonl: {trim_label if args.trim_split == 'd_plus' else other_label}"
                  f" ({original_size} -> {len(trimmed_data)}" 
                  f"{'  unchanged' if args.trim_split != 'd_plus' else ''})")
            print(f"  d_minus.jsonl: {trim_label if args.trim_split == 'd_minus' else other_label}"
                  f" ({original_size} -> {len(trimmed_data)}"
                  f"{'  unchanged' if args.trim_split != 'd_minus' else ''})")
            print(f"  trim_summary.json")


if __name__ == "__main__":
    main()

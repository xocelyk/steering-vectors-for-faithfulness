"""
Build D+/D- contrastive datasets from scored CoT traces.

Splits tasks into high-verbosity (D+) and low-verbosity (D-) sets based on
quartile thresholds of the factor_utilization verbosity score. This follows
the standard difference-in-means methodology for steering vector extraction.

Usage:
    uv run python experiments/phase2/build_contrastive_dataset.py \
        --input experiments/phase2/results/scored_traces/traces_*_scored.jsonl

Output:
    - d_plus.jsonl: Top quartile verbosity tasks (D+)
    - d_minus.jsonl: Bottom quartile verbosity tasks (D-)
    - split_summary.json: Statistics and validation results
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build D+/D- contrastive split from scored traces"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to scored JSONL from score_verbosity.py",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(SCRIPT_DIR / "results" / "contrastive"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--top_percentile", type=float, default=75.0,
        help="Percentile threshold for D+ (default: 75 = top quartile)",
    )
    parser.add_argument(
        "--bottom_percentile", type=float, default=25.0,
        help="Percentile threshold for D- (default: 25 = bottom quartile)",
    )
    parser.add_argument(
        "--require_factors", action="store_true", default=True,
        help="Only include records that have causal factors",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show D+/D- counts at various thresholds without writing files",
    )
    return parser.parse_args()


def load_scored_traces(jsonl_path: str) -> list[dict]:
    """Load scored trace records from JSONL."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_quartile_thresholds(
    scores: list[float],
    top_pct: float,
    bottom_pct: float,
) -> tuple[float, float]:
    """Compute percentile thresholds for D+ and D-."""
    q_top = np.percentile(scores, top_pct)
    q_bottom = np.percentile(scores, bottom_pct)
    return q_top, q_bottom


def split_by_verbosity(
    records: list[dict],
    q_top: float,
    q_bottom: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into D+, D-, and middle sets."""
    d_plus = []
    d_minus = []
    middle = []
    
    for r in records:
        score = r.get("verbosity_score")
        if score is None:
            continue
        
        if score >= q_top:
            d_plus.append(r)
        elif score <= q_bottom:
            d_minus.append(r)
        else:
            middle.append(r)
    
    return d_plus, d_minus, middle


def validate_split(
    d_plus: list[dict],
    d_minus: list[dict],
) -> dict:
    """Validate the contrastive split quality."""
    validation = {}
    
    scores_plus = [r["verbosity_score"] for r in d_plus]
    scores_minus = [r["verbosity_score"] for r in d_minus]
    
    if scores_plus and scores_minus:
        t_stat, p_value = stats.ttest_ind(scores_plus, scores_minus)
        cohens_d = (np.mean(scores_plus) - np.mean(scores_minus)) / np.sqrt(
            (np.std(scores_plus)**2 + np.std(scores_minus)**2) / 2
        )
        validation["verbosity_gap"] = {
            "d_plus_mean": float(np.mean(scores_plus)),
            "d_plus_std": float(np.std(scores_plus)),
            "d_minus_mean": float(np.mean(scores_minus)),
            "d_minus_std": float(np.std(scores_minus)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": p_value < 0.05,
        }
    
    datasets_plus = Counter(r.get("dataset", "unknown") for r in d_plus)
    datasets_minus = Counter(r.get("dataset", "unknown") for r in d_minus)
    
    all_datasets = set(datasets_plus.keys()) | set(datasets_minus.keys())
    balance_ratios = {}
    for ds in all_datasets:
        n_plus = datasets_plus.get(ds, 0)
        n_minus = datasets_minus.get(ds, 0)
        total = n_plus + n_minus
        if total > 0:
            balance_ratios[ds] = {
                "d_plus": n_plus,
                "d_minus": n_minus,
                "ratio": n_plus / total if total > 0 else 0,
            }
    
    validation["dataset_balance"] = {
        "by_dataset": balance_ratios,
        "overall_balanced": all(
            0.3 <= info["ratio"] <= 0.7 
            for info in balance_ratios.values() 
            if info["d_plus"] + info["d_minus"] >= 10
        ),
    }
    
    # Use correctness_score from the LLM judge (set by score_verbosity.py)
    scored_plus = [r["correctness_score"] for r in d_plus if "correctness_score" in r]
    scored_minus = [r["correctness_score"] for r in d_minus if "correctness_score" in r]
    
    acc_plus = sum(scored_plus) / len(scored_plus) if scored_plus else None
    acc_minus = sum(scored_minus) / len(scored_minus) if scored_minus else None
    
    validation["correctness"] = {
        "d_plus_accuracy": acc_plus,
        "d_plus_n_evaluated": len(scored_plus),
        "d_minus_accuracy": acc_minus,
        "d_minus_n_evaluated": len(scored_minus),
        "accuracy_gap": abs(acc_plus - acc_minus) if (acc_plus is not None and acc_minus is not None) else None,
        "balanced": abs(acc_plus - acc_minus) < 0.15 if (acc_plus is not None and acc_minus is not None) else None,
    }
    
    tokens_plus = [r.get("num_tokens", 0) for r in d_plus]
    tokens_minus = [r.get("num_tokens", 0) for r in d_minus]
    
    if tokens_plus and tokens_minus:
        validation["token_length"] = {
            "d_plus_mean": float(np.mean(tokens_plus)),
            "d_plus_std": float(np.std(tokens_plus)),
            "d_minus_mean": float(np.mean(tokens_minus)),
            "d_minus_std": float(np.std(tokens_minus)),
        }
        
        if scores_plus and scores_minus:
            all_scores = scores_plus + scores_minus
            all_tokens = tokens_plus + tokens_minus
            if len(all_scores) > 2:
                corr, corr_p = stats.pearsonr(all_scores, all_tokens)
                validation["token_length"]["correlation_with_verbosity"] = float(corr)
                validation["token_length"]["correlation_p_value"] = float(corr_p)
    
    return validation


def main():
    args = parse_args()
    
    records = load_scored_traces(args.input)
    print(f"Loaded {len(records)} records from {args.input}")
    
    degenerate = [r for r in records if r.get("degenerate_reason")]
    records = [r for r in records if not r.get("degenerate_reason")]
    if degenerate:
        print(f"  Filtered {len(degenerate)} degenerate records")

    if args.require_factors:
        records = [r for r in records if r.get("causal_factors")]
        print(f"  {len(records)} have causal factors")

    scored_records = [r for r in records if r.get("verbosity_score") is not None]
    print(f"  {len(scored_records)} have verbosity scores")
    
    if not scored_records:
        print("ERROR: No scored records found!")
        return
    
    scores = [r["verbosity_score"] for r in scored_records]

    if args.preview:
        print(f"\nVerbosity score distribution:")
        print(f"  Min: {min(scores):.3f}, Max: {max(scores):.3f}")
        print(f"  Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
        print(f"\n{'Threshold':>10s}  {'D- (<=)':>8s}  {'D+ (>=)':>8s}  {'Middle':>8s}")
        print(f"  {'-'*38}")
        for thresh in np.arange(0.1, 1.01, 0.1):
            n_minus = sum(1 for s in scores if s <= thresh)
            n_plus = sum(1 for s in scores if s >= (1.0 - thresh + min(scores)))
            # More useful: show symmetric-ish thresholds
            pass
        def _acc_str(recs):
            c = [r["correctness_score"] for r in recs if "correctness_score" in r]
            if not c:
                return "n/a"
            return f"{sum(c)/len(c):.0%} ({int(sum(c))}/{len(c)})"

        # Show fixed thresholds for D-
        print(f"\n  D- counts (verbosity_score <= threshold):")
        print(f"    {'thresh':>6s}  {'count':>5s}  {'accuracy':>12s}")
        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            matched = [r for r in scored_records if r["verbosity_score"] <= thresh]
            print(f"    <= {thresh:.1f}  {len(matched):>5d}  {_acc_str(matched):>12s}")
        # Show fixed thresholds for D+
        print(f"\n  D+ counts (verbosity_score >= threshold):")
        print(f"    {'thresh':>6s}  {'count':>5s}  {'accuracy':>12s}")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
            matched = [r for r in scored_records if r["verbosity_score"] >= thresh]
            print(f"    >= {thresh:.2f} {len(matched):>5d}  {_acc_str(matched):>12s}")
        # Show percentile-based splits
        print(f"\n  Percentile-based splits:")
        print(f"    {'split':>8s}  {'D-':>4s} (<= thresh)  {'D- acc':>10s}  {'D+':>4s} (>= thresh)  {'D+ acc':>10s}")
        for bottom_pct, top_pct in [(10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (40, 60)]:
            q_b = np.percentile(scores, bottom_pct)
            q_t = np.percentile(scores, top_pct)
            minus = [r for r in scored_records if r["verbosity_score"] <= q_b]
            plus = [r for r in scored_records if r["verbosity_score"] >= q_t]
            print(f"    P{bottom_pct:>2d}/P{top_pct:<2d}  {len(minus):>4d} (<={q_b:.3f})   {_acc_str(minus):>10s}  {len(plus):>4d} (>={q_t:.3f})   {_acc_str(plus):>10s}")
        return

    q_top, q_bottom = compute_quartile_thresholds(
        scores, args.top_percentile, args.bottom_percentile
    )
    
    print(f"\nVerbosity score distribution:")
    print(f"  Min: {min(scores):.3f}, Max: {max(scores):.3f}")
    print(f"  Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
    print(f"  Q{args.bottom_percentile:.0f} threshold (D-): <= {q_bottom:.3f}")
    print(f"  Q{args.top_percentile:.0f} threshold (D+): >= {q_top:.3f}")
    
    d_plus, d_minus, middle = split_by_verbosity(scored_records, q_top, q_bottom)
    
    print(f"\nSplit results:")
    print(f"  D+ (high verbosity): {len(d_plus)} tasks")
    print(f"  D- (low verbosity): {len(d_minus)} tasks")
    print(f"  Middle (excluded): {len(middle)} tasks")
    
    validation = validate_split(d_plus, d_minus)
    
    print(f"\nValidation:")
    if "verbosity_gap" in validation:
        vg = validation["verbosity_gap"]
        print(f"  Verbosity gap: D+={vg['d_plus_mean']:.3f} vs D-={vg['d_minus_mean']:.3f}")
        print(f"  Effect size (Cohen's d): {vg['cohens_d']:.2f}")
        print(f"  Significant (p<0.05): {vg['significant']}")
    
    if "dataset_balance" in validation:
        print(f"  Dataset balance: {validation['dataset_balance']['overall_balanced']}")
    
    if "correctness" in validation:
        corr = validation["correctness"]
        if corr["d_plus_accuracy"] is not None:
            print(f"  Accuracy - D+: {corr['d_plus_accuracy']:.2%}, D-: {corr['d_minus_accuracy']:.2%}")
            print(f"  Accuracy balanced: {corr['balanced']}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    d_plus_path = os.path.join(args.output_dir, "d_plus.jsonl")
    with open(d_plus_path, "w") as f:
        for r in d_plus:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote D+ to: {d_plus_path}")
    
    d_minus_path = os.path.join(args.output_dir, "d_minus.jsonl")
    with open(d_minus_path, "w") as f:
        for r in d_minus:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote D- to: {d_minus_path}")
    
    summary = {
        "input_file": args.input,
        "total_records": len(records),
        "degenerate_filtered": len(degenerate),
        "scored_records": len(scored_records),
        "top_percentile": args.top_percentile,
        "bottom_percentile": args.bottom_percentile,
        "q_top_threshold": q_top,
        "q_bottom_threshold": q_bottom,
        "d_plus_count": len(d_plus),
        "d_minus_count": len(d_minus),
        "middle_count": len(middle),
        "score_distribution": {
            "min": float(min(scores)),
            "max": float(max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
        },
        "validation": validation,
    }
    
    summary_path = os.path.join(args.output_dir, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()

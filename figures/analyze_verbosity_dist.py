"""
Analyze the distribution of verbosity_holistic_score values across all
conditions and questions in traces_latest.json.

Produces:
- Histogram of scores (0-10)
- Counts by condition type (base vs distilled)
- Trace counts under different splitting strategies
- Count of traces with no verbosity score
"""

import json
import sys
from collections import Counter
from pathlib import Path

DATA_PATH = Path("/Users/kyle/Documents/ws/steering-vectors-for-faithfulness/"
                 ".local/experiments/bbh-combined_03-18-2026/results/traces_latest.json")

def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    assert isinstance(data, list), f"Expected list, got {type(data)}"
    print(f"Total questions: {len(data)}")

    # ---- Extract all verbosity_holistic_score values ----
    all_scores = []        # (question_id, condition_name, score)
    degenerate_keys = []   # conditions marked degenerate
    missing_score = []     # conditions with no verbosity score

    for item in data:
        qid = item["id"]

        # Build set of all condition prefixes from degenerate keys
        degen_map = {}
        for k, v in item.items():
            if k.endswith("_degenerate"):
                prefix = k.replace("_degenerate", "")
                degen_map[prefix] = v

        # Find all verbosity_holistic_score keys
        score_keys = {k for k in item if k.endswith("_verbosity_holistic_score")}
        score_prefixes = {k.replace("_verbosity_holistic_score", "") for k in score_keys}

        for prefix, is_degen in degen_map.items():
            if is_degen:
                degenerate_keys.append((qid, prefix))
            elif prefix not in score_prefixes:
                # Not degenerate but no score -- truly missing
                missing_score.append((qid, prefix))

        for k in score_keys:
            prefix = k.replace("_verbosity_holistic_score", "")
            score = item[k]
            if score is None:
                missing_score.append((qid, prefix))
            else:
                all_scores.append((qid, prefix, score))

    print(f"\nTotal (condition, question) pairs with a verbosity score: {len(all_scores)}")
    print(f"Total degenerate (no generation, so no score expected): {len(degenerate_keys)}")
    print(f"Total non-degenerate but MISSING score: {len(missing_score)}")
    if missing_score:
        print("  Examples of missing:", missing_score[:5])

    # ---- Histogram of scores ----
    score_values = [s for _, _, s in all_scores]
    hist = Counter(score_values)
    print("\n--- Histogram of verbosity_holistic_score (0-10) ---")
    for bucket in range(0, 11):
        count = hist.get(bucket, 0)
        bar = "#" * count
        print(f"  {bucket:>2}: {count:>4}  {bar}")
    print(f"  Total: {len(score_values)}")

    # ---- Breakdown by model type ----
    base_scores = [s for _, p, s in all_scores if p.startswith("base")]
    distilled_scores = [s for _, p, s in all_scores if p.startswith("distilled")]
    # "baseline" is the base model with no steering
    baseline_scores = [s for _, p, s in all_scores if p.startswith("baseline")]
    other_scores = [s for _, p, s in all_scores
                    if not p.startswith("base") and not p.startswith("distilled")]

    print("\n--- Counts by model type ---")
    print(f"  base_* conditions:       {len(base_scores):>4}  (includes baseline)")
    print(f"    of which baseline_*:   {len(baseline_scores):>4}")
    print(f"  distilled_* conditions:  {len(distilled_scores):>4}")
    print(f"  other (unclassified):    {len(other_scores):>4}")
    if other_scores:
        other_prefixes = set(p for _, p, _ in all_scores
                             if not p.startswith("base") and not p.startswith("distilled"))
        print(f"    prefixes: {other_prefixes}")

    # Summary stats
    import statistics
    for label, scores in [("base_*", base_scores), ("distilled_*", distilled_scores), ("ALL", score_values)]:
        if not scores:
            continue
        print(f"\n  {label}: n={len(scores)}, mean={statistics.mean(scores):.2f}, "
              f"median={statistics.median(scores):.1f}, stdev={statistics.stdev(scores):.2f}, "
              f"min={min(scores)}, max={max(scores)}")

    # ---- Splitting strategies ----
    print("\n--- Splitting strategies ---")

    sorted_scores = sorted(score_values)
    n = len(sorted_scores)
    median_val = statistics.median(sorted_scores)

    # 1. Binary split at median
    low_med = [s for s in score_values if s < median_val]
    high_med = [s for s in score_values if s > median_val]
    at_med = [s for s in score_values if s == median_val]
    print(f"\n  1) Binary split at median ({median_val}):")
    print(f"     Below median (<{median_val}): {len(low_med)}")
    print(f"     At median (={median_val}):    {at_med and len(at_med) or 0}")
    print(f"     Above median (>{median_val}): {len(high_med)}")
    print(f"     Usable pairs (excluding at-median): {len(low_med)} low + {len(high_med)} high")

    # 2. Top/bottom quartile: scores <=3 vs >=7
    low_q = [s for s in score_values if s <= 3]
    high_q = [s for s in score_values if s >= 7]
    mid_q = [s for s in score_values if 3 < s < 7]
    print(f"\n  2) Top/bottom quartile (<=3 vs >=7):")
    print(f"     Low (<=3):  {len(low_q)}")
    print(f"     Mid (4-6):  {len(mid_q)}")
    print(f"     High (>=7): {len(high_q)}")
    print(f"     Usable pairs: min({len(low_q)}, {len(high_q)}) = {min(len(low_q), len(high_q))}")

    # 3. Top/bottom tercile
    tercile_low = sorted_scores[n // 3]
    tercile_high = sorted_scores[2 * n // 3]
    low_t = [s for s in score_values if s <= tercile_low]
    high_t = [s for s in score_values if s >= tercile_high]
    mid_t = [s for s in score_values if tercile_low < s < tercile_high]
    print(f"\n  3) Top/bottom tercile (<=P33={tercile_low} vs >=P67={tercile_high}):")
    print(f"     Low (<=P33={tercile_low}):  {len(low_t)}")
    print(f"     Mid:                    {len(mid_t)}")
    print(f"     High (>=P67={tercile_high}): {len(high_t)}")
    print(f"     Usable pairs: min({len(low_t)}, {len(high_t)}) = {min(len(low_t), len(high_t))}")

    # ---- Per-condition breakdown ----
    print("\n--- Per-condition score distribution ---")
    cond_scores = {}
    for _, prefix, score in all_scores:
        cond_scores.setdefault(prefix, []).append(score)

    # Sort by mean score
    for prefix in sorted(cond_scores, key=lambda p: statistics.mean(cond_scores[p])):
        scores = cond_scores[prefix]
        mean = statistics.mean(scores)
        print(f"  {prefix:<55} n={len(scores):>3}  mean={mean:.2f}  "
              f"scores: {dict(sorted(Counter(scores).items()))}")


if __name__ == "__main__":
    main()

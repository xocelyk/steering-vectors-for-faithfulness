"""
Plot results from scored steered traces.

Reads the per-record scored JSONL files and the baseline (unsteered) traces,
then produces a set of plots comparing the chosen metric (verbosity or
faithfulness), correctness, and token counts across layers and alpha values.

Usage:
    # Verbosity (default)
    uv run python experiments/phase3/plot_steered_results.py

    # Faithfulness
    uv run python experiments/phase3/plot_steered_results.py \
        --metric faithfulness \
        --scored_dir experiments/phase3/results/faithfulness/steered_stanford \
        --baseline experiments/phase2/results/scored_traces_faithfulness/traces_..._scored.jsonl \
        --output_dir experiments/phase3/results/faithfulness/plots_stanford
"""

from pathlib import Path

import argparse
import json
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE2_DIR = SCRIPT_DIR.parent / "phase2"

BASELINE_PATH = (
    PHASE2_DIR / "results" / "scored_traces"
    / "traces_deepseek-r1-distill-llama-8b_20260311_180636_scored.jsonl"
)

METRIC_CONFIG = {
    "verbosity": {
        "field": "verbosity_score",
        "display": "Verbosity",
    },
    "faithfulness": {
        "field": "faithfulness_score",
        "display": "Faithfulness",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot steered trace results")
    parser.add_argument(
        "--metric", type=str, default="verbosity", choices=list(METRIC_CONFIG),
        help="Which score field to plot (verbosity or faithfulness)",
    )
    parser.add_argument(
        "--scored_dir", type=str,
        default=str(SCRIPT_DIR / "results" / "scored_steered_traces"),
        help="Directory with scored steered JSONL files",
    )
    parser.add_argument(
        "--baseline", type=str, default=str(BASELINE_PATH),
        help="Path to baseline scored traces JSONL",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(SCRIPT_DIR / "results" / "plots"),
        help="Directory to save plots",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_all_scored(scored_dir: str) -> dict[tuple[int, float], list[dict]]:
    """Load all scored files, keyed by (layer, alpha)."""
    pattern = re.compile(r"layer_(\d+)_alpha_([\d.]+)(?:_scored)?\.jsonl$")
    data = {}
    for p in sorted(Path(scored_dir).glob("layer_*_alpha_*.jsonl")):
        m = pattern.match(p.name)
        if m:
            layer, alpha = int(m.group(1)), float(m.group(2))
            data[(layer, alpha)] = load_jsonl(str(p))
    return data


def compute_baseline_metrics(
    baseline_records: list[dict], steered_task_ids: set[str], metric_field: str,
) -> dict:
    matched = [r for r in baseline_records if r["task_id"] in steered_task_ids]
    v = [r[metric_field] for r in matched if metric_field in r]
    c = [r["correctness_score"] for r in matched if "correctness_score" in r]
    tokens = [r["num_tokens"] for r in matched if "num_tokens" in r]
    no_ans = sum(1 for r in matched if not r.get("final_answer", "").strip())
    return {
        "n": len(matched),
        "mean_metric": np.mean(v) if v else 0,
        "mean_correctness": np.mean(c) if c else 0,
        "mean_tokens": np.mean(tokens) if tokens else 0,
        "n_no_answer": no_ans,
    }


def compute_steered_metrics(records: list[dict], metric_field: str) -> dict:
    v = [r[metric_field] for r in records if metric_field in r]
    c = [r["correctness_score"] for r in records if "correctness_score" in r]
    tokens = [r["num_tokens"] for r in records if "num_tokens" in r]
    no_ans = sum(1 for r in records if not r.get("final_answer", "").strip())
    return {
        "n": len(records),
        "mean_metric": np.mean(v) if v else 0,
        "std_metric": np.std(v) if v else 0,
        "mean_correctness": np.mean(c) if c else 0,
        "mean_tokens": np.mean(tokens) if tokens else 0,
        "std_tokens": np.std(tokens) if tokens else 0,
        "n_no_answer": no_ans,
        "pct_no_answer": no_ans / len(records) if records else 0,
    }


def build_metrics_table(
    steered_data: dict[tuple[int, float], list[dict]], metric_field: str,
) -> tuple[list[int], list[float], dict]:
    """Build a structured metrics dict keyed by layer and alpha."""
    layers = sorted({k[0] for k in steered_data})
    alphas = sorted({k[1] for k in steered_data})
    metrics = {}
    for (layer, alpha), records in steered_data.items():
        metrics[(layer, alpha)] = compute_steered_metrics(records, metric_field)
    return layers, alphas, metrics


def plot_metric_by_alpha(
    ax, layers, alphas, metrics, metric_key, baseline_val,
    ylabel, title, layer_colors,
):
    """Line plot: metric vs alpha, one line per layer, with baseline reference."""
    for layer in layers:
        vals = [metrics.get((layer, a), {}).get(metric_key, np.nan) for a in alphas]
        ax.plot(alphas, vals, "o-", color=layer_colors[layer],
                label=f"Layer {layer}", linewidth=2, markersize=6)

    ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.5,
               alpha=0.7, label="Baseline (no steering)")
    ax.set_xlabel("Alpha (steering strength)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(alphas)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_heatmap(ax, layers, alphas, metrics, metric_key, title, fmt=".3f", cmap="RdYlGn"):
    """Heatmap of a metric across layers and alphas."""
    data = np.array([
        [metrics.get((layer, a), {}).get(metric_key, np.nan) for a in alphas]
        for layer in layers
    ])
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    for i in range(len(layers)):
        for j in range(len(alphas)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if val < (np.nanmin(data) + np.nanmax(data)) / 2 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cfg = METRIC_CONFIG[args.metric]
    metric_field = metric_cfg["field"]
    metric_display = metric_cfg["display"]

    steered_data = load_all_scored(args.scored_dir)
    if not steered_data:
        print("No scored steered traces found.")
        return

    steered_task_ids = set()
    for records in steered_data.values():
        steered_task_ids.update(r["task_id"] for r in records)

    baseline_records = load_jsonl(args.baseline)
    baseline = compute_baseline_metrics(baseline_records, steered_task_ids, metric_field)
    print(f"Baseline: {args.metric}={baseline['mean_metric']:.3f}, "
          f"correctness={baseline['mean_correctness']:.3f}, "
          f"tokens={baseline['mean_tokens']:.0f}")

    layers, alphas, metrics = build_metrics_table(steered_data, metric_field)
    print(f"Configs: {len(layers)} layers x {len(alphas)} alphas = {len(metrics)} total")

    layer_colors = {}
    cmap = plt.cm.tab10
    for i, layer in enumerate(layers):
        layer_colors[layer] = cmap(i / max(len(layers) - 1, 1))

    # --- Figure 1: Metric & Correctness vs Alpha ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_metric_by_alpha(
        ax1, layers, alphas, metrics, "mean_metric",
        baseline["mean_metric"],
        f"Mean {metric_display} Score", f"{metric_display} vs Steering Strength",
        layer_colors,
    )

    plot_metric_by_alpha(
        ax2, layers, alphas, metrics, "mean_correctness",
        baseline["mean_correctness"],
        "Mean Correctness", "Correctness vs Steering Strength",
        layer_colors,
    )

    fig1.suptitle(
        f"Effect of Steering on {metric_display} and Correctness",
        fontsize=14, y=1.02,
    )
    fig1.tight_layout()
    fig1_path = output_dir / f"{args.metric}_correctness_vs_alpha.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig1_path}")

    # --- Figure 2: Token Count & No-Answer Rate vs Alpha ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    plot_metric_by_alpha(
        ax3, layers, alphas, metrics, "mean_tokens",
        baseline["mean_tokens"],
        "Mean Token Count", "Token Count vs Steering Strength",
        layer_colors,
    )

    plot_metric_by_alpha(
        ax4, layers, alphas, metrics, "pct_no_answer",
        0.0,
        "% No Final Answer", "No-Answer Rate vs Steering Strength",
        layer_colors,
    )
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig2.suptitle("Effect of Steering on Generation Quality", fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / "tokens_noanswer_vs_alpha.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'tokens_noanswer_vs_alpha.png'}")

    # --- Figure 3: Heatmaps ---
    fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_heatmap(axes[0], layers, alphas, metrics, "mean_metric",
                 f"Mean {metric_display}", fmt=".3f", cmap="RdYlGn")
    plot_heatmap(axes[1], layers, alphas, metrics, "mean_correctness",
                 "Mean Correctness", fmt=".2f", cmap="RdYlGn")
    plot_heatmap(axes[2], layers, alphas, metrics, "pct_no_answer",
                 "% No Answer", fmt=".0%", cmap="RdYlGn_r")

    fig3.suptitle("Steering Results Heatmap (Layer x Alpha)", fontsize=14, y=1.02)
    fig3.tight_layout()
    fig3.savefig(output_dir / "heatmaps.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'heatmaps.png'}")

    # --- Figure 4: Per-task metric change (paired comparison) ---
    fig4, axes4 = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), sharey=True)
    if len(alphas) == 1:
        axes4 = [axes4]

    baseline_by_id = {r["task_id"]: r for r in baseline_records}

    for ai, alpha in enumerate(alphas):
        ax = axes4[ai]
        for layer in layers:
            records = steered_data.get((layer, alpha), [])
            deltas = []
            for r in records:
                tid = r["task_id"]
                if tid in baseline_by_id and metric_field in r:
                    bl = baseline_by_id[tid].get(metric_field)
                    if bl is not None:
                        deltas.append(r[metric_field] - bl)
            if deltas:
                ax.bar(
                    layers.index(layer), np.mean(deltas),
                    color=layer_colors[layer], label=f"Layer {layer}",
                    yerr=np.std(deltas) / np.sqrt(len(deltas)),
                    capsize=4,
                )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Alpha = {alpha}")
        ax.set_xlabel("Layer")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers])
        ax.grid(True, alpha=0.3, axis="y")
        if ai == 0:
            ax.set_ylabel(f"Mean Δ {metric_display} (steered - baseline)")

    fig4.suptitle(
        f"Per-Task {metric_display} Change from Baseline", fontsize=14, y=1.02,
    )
    fig4.tight_layout()
    fig4_path = output_dir / f"{args.metric}_delta.png"
    fig4.savefig(fig4_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig4_path}")

    # --- Print summary table ---
    header_metric = metric_display[:9]
    print(f"\n{'Layer':>5} {'Alpha':>6} {header_metric:>10} {'Correct':>8} {'Tokens':>7} {'No Ans':>7}")
    print("-" * 50)
    print(f"{'base':>5} {'--':>6} {baseline['mean_metric']:>10.3f} "
          f"{baseline['mean_correctness']:>8.3f} {baseline['mean_tokens']:>7.0f} "
          f"{baseline['n_no_answer']:>7}")
    for layer in layers:
        for alpha in alphas:
            m = metrics.get((layer, alpha))
            if m:
                print(f"{layer:>5} {alpha:>6.1f} {m['mean_metric']:>10.3f} "
                      f"{m['mean_correctness']:>8.3f} {m['mean_tokens']:>7.0f} "
                      f"{m['n_no_answer']:>7}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

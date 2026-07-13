#!/usr/bin/env python3
"""Baseline (unsteered) accuracy by model x dataset x cue.

Reads the scored baseline traces under
  experiments/transfer/runs_scored/<model>/baselines/<dataset>/<cue>/*.jsonl
and emits:
  - fig0_baseline_accuracy.png  (grouped bars: uncued accuracy, dataset x model)
  - baseline_accuracy.md        (full markdown table: model x dataset x cue)
  - baseline_accuracy.tex       (LaTeX table for the paper)

Accuracy is computed over judge-scored items only (items with a null
correctness_score are dropped from numerator and denominator; their count is
reported separately as `unscored`).
"""
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import mpl_config
import figstyle as S

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCORED = REPO / "experiments" / "transfer" / "runs_scored"
from paths import artifacts_dir
OUT = artifacts_dir()
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]  # ordered by size
MODEL_LABEL = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B",
               "gemma-3-12b-it": "Gemma-3 12B"}
DS = ["bbh", "gpqa", "mmlu"]
DS_LABEL = {"bbh": "BBH", "gpqa": "GPQA", "mmlu": "MMLU"}
CUES = ["uncued", "stanford", "xml", "grader", "insider"]
CUE_LABEL = {"uncued": "No cue", "stanford": "Stanford", "xml": "XML",
             "grader": "Grader", "insider": "Unethical"}

# --- gather ---
# acc[(model, ds, cue)] = [n_correct, n_scored, n_unscored]
acc = defaultdict(lambda: [0, 0, 0])
files = [f for f in glob.glob(str(SCORED / "*/baselines/*/*/*.jsonl"))
         if not f.endswith(" 2.jsonl")]  # skip the stray duplicate
for f in files:
    parts = Path(f).relative_to(SCORED).parts
    model, _, ds, cue = parts[0], parts[1], parts[2], parts[3]
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cs = d.get("correctness_score")
            if cs is None:
                acc[(model, ds, cue)][2] += 1
            else:
                acc[(model, ds, cue)][1] += 1
                acc[(model, ds, cue)][0] += int(cs == 1)


def rate(model, ds, cue):
    c, n, _ = acc[(model, ds, cue)]
    return (c / n) if n else float("nan")


def n_scored(model, ds, cue):
    return acc[(model, ds, cue)][1]


def cued_avg(model, ds):
    """Mean accuracy across the four cued conditions (n-weighted)."""
    cued = [c for c in CUES if c != "uncued"]
    cs = sum(acc[(model, ds, c)][0] for c in cued)
    ns = sum(acc[(model, ds, c)][1] for c in cued)
    return (cs / ns) if ns else float("nan")


# ============================ FIGURE ============================
# Headline: uncued ("straight up") accuracy, dataset x model grouped bars.
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(DS))
w = 0.26
for i, model in enumerate(MODELS):
    vals = [rate(model, ds, "uncued") for ds in DS]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=MODEL_LABEL[model],
                  color=S.MODEL_COLORS[model], edgecolor=S.EDGE, linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7, color=S.HEAT_TEXT)
ax.set_xticks(x)
ax.set_xticklabels([DS_LABEL[d] for d in DS])
ax.set_ylabel("Accuracy (no cue)")
ax.set_ylim(0, 1.0)
ax.set_title("Baseline accuracy by dataset and model")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
mpl_config.save(fig, str(OUT / "fig0_baseline_accuracy"), png=True, pdf=False)
print(f"wrote {OUT / 'fig0_baseline_accuracy.png'}")

# Companion: accuracy across all cue conditions (one panel per dataset).
fig, axes = plt.subplots(1, len(DS), figsize=(11.5, 3.8), sharey=True)
xc = np.arange(len(CUES))
for ax, ds in zip(axes, DS):
    for i, model in enumerate(MODELS):
        vals = [rate(model, ds, c) for c in CUES]
        ax.bar(xc + (i - 1) * w, vals, w, label=MODEL_LABEL[model],
               color=S.MODEL_COLORS[model], edgecolor=S.EDGE, linewidth=0.6)
    ax.axvline(0.5, ls=":", lw=1, color=S.EDGE, zorder=0)  # divide no-cue from cued
    ax.set_xticks(xc)
    ax.set_xticklabels([CUE_LABEL[c] for c in CUES], rotation=30, ha="right", fontsize=8)
    ax.set_title(DS_LABEL[ds])
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("Accuracy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Baseline accuracy by cue condition", y=1.13, fontsize=13, fontweight="bold")
fig.tight_layout()
mpl_config.save(fig, str(OUT / "fig0b_cued_accuracy"), png=True, pdf=False)
print(f"wrote {OUT / 'fig0b_cued_accuracy.png'}")

# ============================ MARKDOWN ============================
md = []
md.append("# Baseline (unsteered) accuracy\n")
md.append("Fraction of judge-graded items answered correctly, by model x dataset x "
          "cue condition. Accuracy is over scored items only; `unscored` counts "
          "items the judge left ungraded (truncated / failed) and excluded from the "
          "denominator. Source: `experiments/transfer/runs_scored/<model>/baselines/`.\n")
md.append("**No cue** is the model's raw capability; the cued columns show accuracy "
          "when a (sometimes misleading) cue toward one option is injected.\n")

# Per-dataset tables
for ds in DS:
    md.append(f"\n## {DS_LABEL[ds]}\n")
    header = "| Model | " + " | ".join(CUE_LABEL[c] for c in CUES) + " | Cued avg | n (uncued) |"
    sep = "|" + "---|" * (len(CUES) + 3)
    md.append(header)
    md.append(sep)
    for model in MODELS:
        cells = [f"{rate(model, ds, c):.3f}" for c in CUES]
        row = (f"| {MODEL_LABEL[model]} | " + " | ".join(cells) +
               f" | {cued_avg(model, ds):.3f} | {n_scored(model, ds, 'uncued')} |")
        md.append(row)

# Compact uncued summary
md.append("\n## Summary — no-cue accuracy (model x dataset)\n")
md.append("| Model | " + " | ".join(DS_LABEL[d] for d in DS) + " |")
md.append("|" + "---|" * (len(DS) + 1))
for model in MODELS:
    cells = [f"{rate(model, ds, 'uncued'):.3f}" for ds in DS]
    md.append(f"| {MODEL_LABEL[model]} | " + " | ".join(cells) + " |")

(OUT / "baseline_accuracy.md").write_text("\n".join(md) + "\n")
print(f"wrote {OUT / 'baseline_accuracy.md'}")

# ============================ LATEX ============================
tex = []
tex.append(r"% Baseline (unsteered) accuracy by model x dataset x cue.")
tex.append(r"% Accuracy over judge-scored items only.")
tex.append(r"\begin{table*}[t]\centering\small")
tex.append(r"\caption{Baseline (unsteered) accuracy by model, dataset, and cue "
           r"condition. \textbf{No cue} is raw capability; cued columns inject a "
           r"hint toward one option. Accuracy is computed over judge-scored items "
           r"only. \textbf{Cued} averages the four cued conditions.}")
tex.append(r"\label{tab:baseline-acc}")
tex.append(r"\begin{tabular}{ll rrrrr r}")
tex.append(r"\toprule")
tex.append(r"Dataset & Model & No cue & Stanford & XML & Grader & Unethical & Cued \\")
tex.append(r"\midrule")
for ds in DS:
    for i, model in enumerate(MODELS):
        dlab = DS_LABEL[ds] if i == 0 else ""
        vals = " & ".join(f"{rate(model, ds, c):.2f}" for c in CUES)
        tex.append(f"{dlab} & {MODEL_LABEL[model]} & {vals} & {cued_avg(model, ds):.2f} \\\\")
    tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
tex.append(r"\end{tabular}\end{table*}")
(OUT / "baseline_accuracy.tex").write_text("\n".join(tex) + "\n")
print(f"wrote {OUT / 'baseline_accuracy.tex'}")

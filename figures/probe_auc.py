#!/usr/bin/env python3
"""Probe test ROC-AUC vs layer, per model, one curve per cue (GPQA).
Methodology figure for layer selection (paper fig:probe-auc). Also a compact table.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mpl_config
import figstyle as S

# one color per cue, in CUES order below (stanford, xml, grader, insider)
COLS = [S.CUE_COLORS["stanford"], S.CUE_COLORS["xml"], S.CUE_COLORS["grader"], S.CUE_COLORS["insider"]]
ROOT = Path(__file__).resolve().parent.parent / "experiments" / "transfer" / "probes" / "meek"
from paths import artifacts_dir
OUT = artifacts_dir()
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
CUES = ["stanford", "xml", "grader", "insider"]
CLAB = {"stanford": "Stanford", "xml": "XML", "grader": "Grader", "insider": "Unethical"}

def load(model, cue):
    f = ROOT / f"{model}__gpqa__{cue}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    layers = sorted(d["layers"], key=lambda e: e["layer"])
    x = [e["layer"] for e in layers]
    test = [e["test_roc_auc"] for e in layers]
    return x, test, d["best_layer"], d["best_by_roc"]["test_roc_auc"]

fig, axes = mpl_config.figure(1, 3, width=11, sharey=True)
table_lines = ["# Probe test ROC-AUC by layer (GPQA cues)\n",
               "Linear probes (cue-acknowledgement) trained at every layer; **train ROC-AUC = 1.00 "
               "in every cell (overfit)**, test AUC below. Selected layer = argmax test AUC. "
               "Chance = 0.50. Note AUC barely exceeds chance and (see `layer_analysis.md`) does "
               "not predict steering quality.\n"]
for ax, model in zip(axes, MODELS):
    table_lines.append(f"\n**{MLAB[model]}** — selected layer (test AUC):")
    for cue, c in zip(CUES, COLS):
        r = load(model, cue)
        if not r:
            continue
        x, test, bl, bauc = r
        ax.plot(x, test, "-", color=c, lw=1.6, label=CLAB[cue])
        ax.plot([bl], [bauc], "o", color=c, markeredgecolor="#444", markersize=7, zorder=5)
        table_lines.append(f"- {CLAB[cue]}: **L{bl}** (AUC {bauc:.2f}); "
                           f"range across layers {min(test):.2f}–{max(test):.2f}")
    ax.axhline(0.5, color="#444", lw=1.0, ls="--", zorder=1)
    ax.set_title(MLAB[model], fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_ylim(0.45, 1.0)
axes[0].set_ylabel("Probe test ROC-AUC")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=7.5, title="Cue", title_fontsize=7.5)
# annotate the train-AUC overfit fact on the first panel
axes[0].text(0.03, 0.04, "train AUC $\\approx$ 1.00\n(overfit)", transform=axes[0].transAxes,
             fontsize=7.5, color="#555", va="bottom", zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
# figure title removed; caption is in the LaTeX \caption
mpl_config.save(fig, str(OUT / "fig8_probe_auc_by_layer"), png=True, pdf=True)
plt.close(fig)

# ---- fig8b: selected test AUC by dataset x cue, per model (the dataset pattern) ----
import json as _j
DS = ["bbh", "gpqa", "mmlu"]; DLAB = {"bbh": "BBH", "gpqa": "GPQA", "mmlu": "MMLU"}
def sel_auc(model, dataset, cue):
    f = ROOT / f"{model}__{dataset}__{cue}.json"
    if not f.exists(): return None
    return _j.load(open(f))["best_by_roc"]["test_roc_auc"]
DCOL = S.DATASET_COLORS
fig, axes = mpl_config.figure(1, 3, width=11, sharey=True)
w = 0.26
for ax, model in zip(axes, MODELS):
    xc = np.arange(len(CUES))
    for di, ds in enumerate(DS):
        vals = [sel_auc(model, ds, c) or np.nan for c in CUES]
        ax.bar(xc + (di - 1) * w, vals, w, color=DCOL[ds], edgecolor="#444",
               zorder=3, label=DLAB[ds] if model == MODELS[0] else None)
    ax.axhline(0.5, color="#444", lw=1.0, ls=":", zorder=4)
    ax.set_xticks(xc); ax.set_xticklabels([CLAB[c] for c in CUES], fontsize=8)
    ax.set_title(MLAB[model], fontsize=10); ax.set_ylim(0.45, 1.0)
axes[0].set_ylabel("Selected probe test ROC-AUC")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=8, title="Dataset", title_fontsize=8)
# figure title removed; caption is in the LaTeX \caption
mpl_config.save(fig, str(OUT / "fig8b_probe_auc_by_dataset"), png=True, pdf=True)
plt.close(fig)

# dataset summary into the md
import statistics as _st
table_lines.append("\n## Selected test AUC by dataset (median over cues, per model)\n")
table_lines.append("| Model | BBH | GPQA | MMLU |")
table_lines.append("|---|---|---|---|")
for model in MODELS:
    cells = {ds: [sel_auc(model, ds, c) for c in CUES if sel_auc(model, ds, c)] for ds in DS}
    table_lines.append(f"| {MLAB[model]} | " + " | ".join(
        f"{_st.median(cells[ds]):.2f}" if cells[ds] else "—" for ds in DS) + " |")
table_lines.append("\n**Probes are dataset-dependent:** strong on MMLU (0.71–0.98), moderate on BBH, "
                   "near chance on GPQA (0.57–0.70) and the unified pools (0.57–0.60). `fig8` shows "
                   "GPQA only (worst case); `fig8b` shows all datasets. Selected AUC = argmax over "
                   "layers on the test set (optimistic); train AUC = 1.00 everywhere (overfit).")
(OUT / "probe_auc_by_layer.md").write_text("\n".join(table_lines) + "\n")
print("wrote fig8_probe_auc_by_layer + fig8b_probe_auc_by_dataset + probe_auc_by_layer.md")

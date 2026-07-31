#!/usr/bin/env python3
"""Probe test AUROC vs layer, per model, one curve per cue (GPQA).
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
table_lines = ["# Probe test AUROC by layer (GPQA cues)\n",
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
    ax.set_title(MLAB[model], fontsize=13)
    ax.set_xlabel("Layer")
    ax.set_ylim(0.45, 1.0)
axes[0].set_ylabel("Probe test AUROC", fontsize=12.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=11, title="Cue", title_fontsize=11)
# annotate the train-AUC overfit fact on the first panel
axes[0].text(0.03, 0.04, "train AUC $\\approx$ 1.00\n(overfit)", transform=axes[0].transAxes,
             fontsize=10.5, color="#555", va="bottom", zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
# figure title removed; caption is in the LaTeX \caption
mpl_config.save(fig, str(OUT / "fig8_probe_auc_by_layer"), png=True, pdf=True)
plt.close(fig)

# ---- fig8b: selected test AUC by dataset x cue, per model (the dataset pattern) ----
import json as _j
DS = ["bbh", "gpqa", "mmlu"]; DLAB = {"bbh": "BBH", "gpqa": "GPQA", "mmlu": "MMLU"}
TRANSFER = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
def sel_auc(model, dataset, cue):
    f = ROOT / f"{model}__{dataset}__{cue}.json"
    if not f.exists(): return None
    return _j.load(open(f))["best_by_roc"]["test_roc_auc"]

def test_class_counts(model, dataset, cue):
    """(n_ack, n_noack) of the meek TEST split, from the scored cued baselines --
    the same labels the probe is evaluated against."""
    sp = TRANSFER / "splits" / model / dataset / cue / "meek_test_task_ids.txt"
    if not sp.exists(): return None
    ids = set(sp.read_text().split())
    fs = sorted(glob.glob(str(TRANSFER / "runs_scored" / model / "baselines" / dataset / cue / "*.jsonl")))
    if not fs: return None
    npos = nneg = 0
    for line in open(fs[-1]):
        line = line.strip()
        if not line: continue
        try: r = _j.loads(line)
        except _j.JSONDecodeError: continue
        if r.get("task_id") not in ids: continue
        s = r.get("faithfulness_score")
        if s is None: continue
        npos += int(s) == 1; nneg += int(s) != 1
    return (npos, nneg) if (npos and nneg) else None

def auc_ci90(a, model, dataset, cue):
    """90% CI for AUROC a: Hanley-McNeil SE mapped to the logit scale by the
    delta method and back-transformed, so the bounds stay inside (0,1) and the
    interval is asymmetric near the ceiling. Returns (down, up) half-widths.
    (Does not account for the argmax-over-layers selection, which the caption
    flags separately.)"""
    cc = test_class_counts(model, dataset, cue)
    if not cc or a is None: return (0.0, 0.0)
    n1, n2 = cc
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)
    se = max(var, 0.0) ** 0.5
    eps = 1e-6  # guard the logit against a == 0 or 1 exactly
    ac = min(max(a, eps), 1 - eps)
    se_logit = se / (ac * (1 - ac))
    logit = np.log(ac / (1 - ac))
    lo = 1 / (1 + np.exp(-(logit - 1.645 * se_logit)))
    hi = 1 / (1 + np.exp(-(logit + 1.645 * se_logit)))
    return (a - lo, hi - a)

DCOL = S.DATASET_COLORS
# dot-and-whisker, matching the style of the main-text effect figures
fig, axes = mpl_config.figure(1, 3, width=11, height=2.8, sharey=True)
for ax, model in zip(axes, MODELS):
    xc = np.arange(len(CUES))
    for di, ds in enumerate(DS):
        vals = [sel_auc(model, ds, c) or np.nan for c in CUES]
        errs = [auc_ci90(sel_auc(model, ds, c), model, ds, c) for c in CUES]
        lo = [e[0] for e in errs]
        hi = [e[1] for e in errs]
        ax.errorbar(xc + (di - 1) * 0.22, vals, yerr=[lo, hi],
                    fmt="o", linestyle="none", zorder=3,
                    label=DLAB[ds] if model == MODELS[0] else None,
                    color=DCOL[ds], markeredgecolor="#444", markersize=6,
                    ecolor="#444", elinewidth=1.1, capsize=2.5)
    ax.axhline(0.5, color="#444", lw=1.0, ls=":", zorder=2)
    ax.set_xlim(-0.5, len(CUES) - 0.5)
    ax.set_xticks(xc); ax.set_xticklabels([CLAB[c] for c in CUES], fontsize=11.5)
    ax.set_title(MLAB[model], fontsize=13); ax.set_ylim(0.45, 1.0)
axes[0].set_ylabel("Selected probe test AUROC", fontsize=12.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=11, title="Dataset", title_fontsize=11)
# figure title removed; caption is in the LaTeX \caption
mpl_config.save(fig, str(OUT / "fig8b_probe_auc_by_dataset"), png=True, pdf=True)
plt.close(fig)

# dataset summary into the md
import statistics as _st
table_lines.append("\n## Selected test AUC by dataset (mean over cues, per model)\n")
table_lines.append("| Model | BBH | GPQA | MMLU |")
table_lines.append("|---|---|---|---|")
for model in MODELS:
    cells = {ds: [sel_auc(model, ds, c) for c in CUES if sel_auc(model, ds, c)] for ds in DS}
    table_lines.append(f"| {MLAB[model]} | " + " | ".join(
        f"{_st.mean(cells[ds]):.2f}" if cells[ds] else "—" for ds in DS) + " |")
table_lines.append("\n**Probes are dataset-dependent:** strong on MMLU (0.71–0.98), moderate on BBH "
                   "(0.62–0.94), near chance on GPQA (0.57–0.70) and the unified pools (0.57–0.60). `fig8` shows "
                   "GPQA only (worst case); `fig8b` shows all datasets. Selected AUC = argmax over "
                   "layers on the test set (optimistic); train AUC = 1.00 everywhere (overfit).")
(OUT / "probe_auc_by_layer.md").write_text("\n".join(table_lines) + "\n")
print("wrote fig8_probe_auc_by_layer + fig8b_probe_auc_by_dataset + probe_auc_by_layer.md")

#!/usr/bin/env python3
"""Dedicated steering-coefficient (alpha) robustness figure.

Reads figures/agg.json, writes fig_alpha_robustness.{png,pdf} to the
paper's artifacts/ folder.

Motivation: the other figures collapse to a single alpha=5.0. This one shows the
matched-setting effect as a function of alpha so the robustness (the effect is
flat across the swept coefficients) is explicit rather than hidden.

Metrics + pooling are identical to fig4b in make_figures.py (matched GPQA cells,
train cue == eval cue, pooled over the 4 cues), so the alpha=5.0 column here
reconciles with fig4b exactly.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mpl_config
import figstyle as S

HERE = Path(__file__).resolve().parent
from paths import artifacts_dir
OUT = artifacts_dir()
OUT.mkdir(parents=True, exist_ok=True)
rows = json.load(open(HERE / "agg.json"))

# Only the alphas that form a COMPLETE sweep across every model x method x cell.
# (1.0 / 7.0 / 10.0 exist only as stray one-off runs on single cells.)
ALPHAS = ["2.5", "5.0", "7.5"]
APOS = {a: i for i, a in enumerate(ALPHAS)}

MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]  # by size: 4B, 9B, 12B
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
CUES = ["stanford", "xml", "grader", "insider"]
METHODS = ["contrastive", "synthetic", "opt-specific", "opt-generic"]
METHLAB = {"contrastive": "Contrastive", "synthetic": "Synthetic",
           "opt-specific": "Opt. specific", "opt-generic": "Opt. generic"}

Z = 1.645  # 90% normal quantile

def ci_prop(p, n):
    if p is None or not n:
        return 0.0
    return Z * (p * (1 - p) / n) ** 0.5

def ci_net(b, c, n):
    if not n:
        return 0.0
    var = ((b + c) / n - ((b - c) / n) ** 2) / n
    return Z * max(var, 0.0) ** 0.5

def matched_cells(method, model, alpha):
    """The 4 matched GPQA cells (train cue == eval cue) for one (method, model, alpha)."""
    cells = []
    for cue in CUES:
        for r in rows:
            if (r["alpha"] == alpha and r["method"] == method and r["model"] == model
                    and r["scenario"] == f"gpqa_{cue}" and r["eval_dataset"] == "gpqa"
                    and r["eval_cue"] == cue):
                cells.append(r)
    return cells

def pooled(method, model, alpha):
    cells = matched_cells(method, model, alpha)
    if not cells:
        return None
    nconv = sum(c["n_converted"] for c in cells)
    nunf = sum(c["n_unfaith_base"] for c in cells)
    nregr = sum(c["n_regressed"] for c in cells)
    n = sum(c["n"] for c in cells)
    conv = nconv / nunf if nunf else float("nan")
    net = (nconv - nregr) / n if n else float("nan")
    return dict(conv=conv, conv_ci=ci_prop(conv, nunf),
                net=net, net_ci=ci_net(nconv, nregr, n))

# ------------------------------------------------------------------ figure
# 1 row (delta) x 4 cols (methods); 3 model lines per panel.
fig, axes = mpl_config.figure(1, len(METHODS), width=12, height=3.0,
                              sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
x = [APOS[a] for a in ALPHAS]
colors = S.MODEL_COLORS

for j, method in enumerate(METHODS):
    ax_net = axes[j]
    for model in MODELS:
        pm = [pooled(method, model, a) for a in ALPHAS]
        net = [p["net"] if p else np.nan for p in pm]
        net_ci = [p["net_ci"] if p else 0 for p in pm]
        ax_net.plot(x, net, "-", color=colors[model], lw=0.9, alpha=0.45, zorder=2)
        ax_net.errorbar(x, net, yerr=net_ci, fmt="o", linestyle="none",
                        color=colors[model], markersize=6, markeredgecolor="#444",
                        capsize=2.5, elinewidth=1.1, ecolor="#444",
                        label=MLAB[model], zorder=3)
    ax_net.set_title(METHLAB[method], fontsize=13)
    ax_net.axhline(0, color="#444", lw=0.9, zorder=2)
    ax_net.set_xticks(x)
    ax_net.set_xticklabels(ALPHAS, fontsize=11.5)
    ax_net.set_xlim(-0.3, len(ALPHAS) - 0.7)
    ax_net.set_xlabel(r"Steering coefficient $\alpha$", fontsize=11.5)

axes[0].set_ylabel(r"$\Delta_{\mathrm{ack}}$", fontsize=12.5)
axes[0].set_ylim(-0.08, 0.13)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=11.5)
# figure title removed; caption is in the LaTeX \caption
mpl_config.save(fig, str(OUT / "fig_alpha_robustness"), png=True, pdf=True)
print(f"wrote {OUT / 'fig_alpha_robustness'}.png/.pdf")

# ------------------------------------------------------------------ console table
print("\nMatched-setting (GPQA, train cue == eval cue), pooled over 4 cues:")
for method in METHODS:
    print(f"\n### {method}")
    print(f"{'model':12s} " + "".join(f"{'a='+a:>10}" for a in ALPHAS))
    for metric, lab in [("conv", "conversion"), ("net", "netDfaith")]:
        for model in MODELS:
            vals = []
            for a in ALPHAS:
                p = pooled(method, model, a)
                vals.append(f"{p[metric]:+.3f}" if p else "    .   ")
            print(f"  {MLAB[model]:9s}[{lab:9s}] " + "".join(f"{v:>10}" for v in vals))

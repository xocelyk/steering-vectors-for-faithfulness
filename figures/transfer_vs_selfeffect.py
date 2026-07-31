#!/usr/bin/env python3
"""Transfer effect vs the self-steering effect of its train and test settings.

Each off-diagonal cell of the contrastive transfer matrices (cross-cue on GPQA,
cross-dataset with the Stanford cue; all three models, alpha=5) is plotted twice
at the same height (its transfer delta_ack): once at the self-steering delta_ack
of the vector's train setting (red) and once at that of the test setting it was
applied to (blue), joined by a connector. Test-side placements hug the y=x
diagonal (transfer ~= test setting's own effect) while train-side placements
scatter -- the effect is determined by the eval setting, not the train setting.
Caveat for the caption: a cell and its test-setting diagonal share eval items
and baseline, so some tightness around y=x is shared eval-set noise.
"""
import json
from pathlib import Path

import numpy as np
import mpl_config
import figstyle as S
from matplotlib.lines import Line2D

from paths import artifacts_dir
OUT = artifacts_dir()
OUT.mkdir(parents=True, exist_ok=True)
rows = json.load(open(Path(__file__).resolve().parent / "agg.json"))
A = "5.0"

MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
CUES = ["stanford", "xml", "grader", "insider"]
DS = ["bbh", "gpqa", "mmlu"]
DSCEN = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}

TRAIN_C, TEST_C = S.RED, S.TEAL


def get(model, scenario, ed, ec):
    r = [x for x in rows if x["alpha"] == A and x["method"] == "contrastive"
         and x["model"] == model and x["scenario"] == scenario
         and x["eval_dataset"] == ed and x["eval_cue"] == ec]
    return r[0] if r else None


def net(c):
    return c["steer_faith"] - c["base_faith"]


def collect(settings, cell_of):
    """(train_self, test_self, transfer) for every off-diagonal (train, test) pair."""
    pts = []
    for model in MODELS:
        self_d = {s: net(c) for s in settings if (c := cell_of(model, s, s))}
        for tr in settings:
            for te in settings:
                if tr == te:
                    continue
                c = cell_of(model, tr, te)
                if c and tr in self_d and te in self_d:
                    pts.append((self_d[tr], self_d[te], net(c)))
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
            np.array([p[2] for p in pts]))


def panel(ax, tr_self, te_self, transfer, title, xlabel):
    for a, b, t in zip(tr_self, te_self, transfer):
        ax.plot([a, b], [t, t], color="#BBB", lw=0.7, alpha=0.55, zorder=2)
    lo = min(tr_self.min(), te_self.min(), transfer.min()) - 0.02
    hi = max(tr_self.max(), te_self.max(), transfer.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1.0, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.scatter(tr_self, transfer, s=32, color=TRAIN_C,
               edgecolor="#444", linewidth=0.6, alpha=0.85, zorder=3)
    ax.scatter(te_self, transfer, s=32, color=TEST_C,
               edgecolor="#444", linewidth=0.6, alpha=0.85, zorder=3)
    ax.axhline(0, color="#444444", lw=0.8, zorder=1)
    ax.axvline(0, color="#444444", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(labelsize=11.5)
    r_tr = np.corrcoef(tr_self, transfer)[0, 1]
    r_te = np.corrcoef(te_self, transfer)[0, 1]
    print(f"{title}: n={len(transfer)}  train r={r_tr:+.3f}  test r={r_te:+.3f}")


cue = collect(CUES, lambda m, tr, te: get(m, f"gpqa_{tr}", "gpqa", te))
ds = collect(DS, lambda m, tr, te: get(m, DSCEN[tr], te, "stanford"))

fig, axes = mpl_config.figure(1, 2, width=9.2, height=4.6)
panel(axes[0], *cue, "Across cues (GPQA)",
      "Self-steering $\\Delta_{\\mathrm{ack}}$ of cue")
panel(axes[1], *ds, "Across datasets (Stanford cue)",
      "Self-steering $\\Delta_{\\mathrm{ack}}$ of dataset")
axes[0].set_ylabel("Transfer $\\Delta_{\\mathrm{ack}}$ (off-diagonal cell)", fontsize=13)

handles = [Line2D([], [], marker="o", ls="", color=TRAIN_C, markeredgecolor="#444",
                  label="Train setting"),
           Line2D([], [], marker="o", ls="", color=TEST_C, markeredgecolor="#444",
                  label="Test setting")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11.5,
           bbox_to_anchor=(0.5, -0.12))

mpl_config.save(fig, str(OUT / "transfer_vs_selfeffect"), png=True, pdf=True, dpi=300)
print(f"wrote {OUT}/transfer_vs_selfeffect.(png|pdf)")

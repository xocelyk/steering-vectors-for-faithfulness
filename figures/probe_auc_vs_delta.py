#!/usr/bin/env python3
"""Probe test AUROC vs steering effect (delta_ack) on the four GPQA cues
(matched setting, contrastive, alpha=5), all three models. Checks whether per-cue
decodability of acknowledgment predicts the per-cue steering effect. Writes a
figure + markdown to the paper's artifacts/."""
import json
import math
from pathlib import Path

import mpl_config
import figstyle as S

TR = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
from paths import artifacts_dir
OUT = artifacts_dir()
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
CUES = ["stanford", "xml", "grader", "insider"]
CLAB = {"stanford": "St.", "xml": "XML", "grader": "Gr.", "insider": "Un."}
CFULL = {"stanford": "Stanford", "xml": "XML", "grader": "Grader", "insider": "Unethical"}
Z = 1.645  # 90% normal-approximation CI, as in g12_alpha_table

agg = json.load(open(Path(__file__).resolve().parent / "agg.json"))

def cell(model, cue):
    c = [x for x in agg if x["alpha"] == "5.0" and x["method"] == "contrastive"
         and x["model"] == model and x["scenario"] == f"gpqa_{cue}"
         and x["eval_dataset"] == "gpqa" and x["eval_cue"] == cue]
    assert len(c) == 1, (model, cue, len(c))
    return c[0]

def probe_auc(model, cue):
    d = json.load(open(TR / "probes" / "meek" / f"{model}__gpqa__{cue}.json"))
    return d["best_by_roc"]["test_roc_auc"], d["best_layer"]

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")

pts = {m: [] for m in MODELS}
for m in MODELS:
    for cue in CUES:
        c = cell(m, cue)
        nc, nr, n = c["n_converted"], c["n_regressed"], c["n"]
        net = (nc - nr) / n
        hw = Z * math.sqrt(max((nc + nr - (nc - nr) ** 2 / n) / n ** 2, 0))
        auc, layer = probe_auc(m, cue)
        pts[m].append(dict(cue=cue, auc=auc, net=net, hw=hw, n=n, layer=layer))

rs = {m: pearson([p["auc"] for p in pts[m]], [p["net"] for p in pts[m]]) for m in MODELS}

CUE_MARKERS = {"stanford": "o", "xml": "s", "grader": "^", "insider": "D"}

fig, ax = mpl_config.figure(width=6.4, height=3.8)
for m in MODELS:
    col = S.MODEL_COLORS[m]
    for p in pts[m]:
        ax.errorbar(p["auc"], p["net"], yerr=p["hw"], fmt=CUE_MARKERS[p["cue"]],
                    color=col, markersize=6.5, markeredgecolor="#444", markeredgewidth=0.8,
                    elinewidth=1.1, ecolor="#444", capsize=2.5, zorder=3)
ax.axhline(0, color="#444", lw=0.8, zorder=1)
ax.set_xlabel("Probe test AUROC (selected layer)", fontsize=11.5)
ax.set_ylabel(r"$\Delta_{\mathrm{ack}}$", fontsize=11.5)
ax.tick_params(labelsize=11)
handles = [ax.plot([], [], "o", color=S.MODEL_COLORS[m], markeredgecolor="#444",
                   markeredgewidth=0.8, linestyle="none", label=MLAB[m])[0] for m in MODELS]
handles += [ax.plot([], [], CUE_MARKERS[c], color="#BBB", markeredgecolor="#444",
                    markeredgewidth=0.8, linestyle="none", label=CFULL[c])[0] for c in CUES]
fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=10)
mpl_config.save(fig, str(OUT / "fig8c_probe_auc_vs_delta_gpqa"), png=True, pdf=True)

md = ["# Probe AUROC vs delta_ack: GPQA cues, all models (matched, contrastive, alpha=5)\n",
      "Per-cue probe test AUROC (selected layer) against the matched steering effect, with "
      "90% normal-approximation CIs on delta_ack.\n",
      "| Model | Cue | Layer | Probe AUROC | n | delta_ack | 90% CI |",
      "|---|---|---|---|---|---|---|"]
for m in MODELS:
    for p in pts[m]:
        md.append(f"| {MLAB[m]} | {CFULL[p['cue']]} | L{p['layer']} | {p['auc']:.2f} | {p['n']} | "
                  f"{p['net']:+.2f} | +-{p['hw']:.2f} |")
md.append("\nPearson r across the four cue cells (n=4 per model, descriptive only): "
          + ", ".join(f"{MLAB[m]} **{rs[m]:+.2f}**" for m in MODELS) + ".")
(OUT / "probe_auc_vs_delta_gpqa.md").write_text("\n".join(md) + "\n")

for m in MODELS:
    print(f"{MLAB[m]}  (r = {rs[m]:+.3f})")
    for p in pts[m]:
        print(f"  {p['cue']:10s} L{p['layer']:<3d} auc={p['auc']:.2f} net={p['net']:+.2f} +-{p['hw']:.2f} n={p['n']}")

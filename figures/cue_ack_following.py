#!/usr/bin/env python3
"""Cue acknowledgement vs cue following (disclosure vs genuine faithfulness).

ack    = faithfulness_score == 1            (CoT mentions the cue)
follow = judge's model_answer_letter == cued letter (model actually picked the cued
         option; NOT merely 'answered incorrectly' — that proxy overcounts ~88% vs true ~40%)

The chosen letter comes from the correctness judge's structured model_answer_letter
field (scoring v3+); no regex parsing of the response text.
Matched by task_id, pooled over the 4 GPQA cues. Stacked-bar figure + 2x2 table PNG.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mpl_config

import figstyle as S
RED, TEAL = S.RED, S.TEAL
TEAL_LIGHT = "#8FBDBD"   # corrected + faithful (good, not disclosed cue use)
NEUTRAL = "#C9C6C0"      # correct, no mention (uninformative)
TR = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
from paths import artifacts_dir
OUT = artifacts_dir()
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
CUES = ["stanford", "xml", "grader", "insider"]

def load_jsonl(p):
    """Read (ack, follow) per task_id; the chosen letter is the judge's model_answer_letter."""
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fa = r.get("faithfulness_score")
            if fa is None:
                continue
            co = r.get("correctness_score")
            letter = r.get("model_answer_letter")
            ct = (r.get("cue_target_letter") or "").strip().upper()
            # USE is judge-gated: the cue points at a wrong option, so a correct answer is
            # never cue-use; require the judge's incorrect verdict plus the cued letter.
            follow = (letter is not None and letter == ct and co is not None and int(co) == 0)
            out[r["task_id"]] = (int(fa) == 1, follow)  # (ack, follow)
    return out or None

def baseline_idx(model, cue):
    fs = glob.glob(str(TR / "runs_scored" / model / "baselines" / "gpqa" / cue / "*.jsonl"))
    return load_jsonl(fs[0]) if fs else None

def steered_idx(model, cue, alpha="5.0"):
    # Pin the steering coefficient explicitly. The directory holds one file per alpha
    # (alpha2.5 / alpha5.0 / alpha7.5, and alpha1.0 for one cell); glob order is NOT
    # sorted, so taking fs[0] would load a nondeterministic, often-wrong alpha. The
    # main text reports alpha=5.
    fs = glob.glob(str(TR / "runs_steered_scored" / "contrastive" / "meek" / model
                       / f"gpqa_{cue}" / "gpqa" / cue / f"*alpha{alpha}.jsonl"))
    return load_jsonl(fs[0]) if fs else None

# joint counts: index 0..3 = (ack,follow) in order
# 0: follow & silent (not ack)   -> hidden cue use  (RED)
# 1: follow & ack                -> disclosed cue use (TEAL)
# 2: not-follow & ack            -> acknowledged, corrected (GREEN)
# 3: not-follow & silent         -> correct, no mention (SAND)
def joint(model, which, alpha="5.0"):
    counts = np.zeros(4)
    for cue in CUES:
        b = baseline_idx(model, cue); s = steered_idx(model, cue, alpha)
        if not b or not s:
            continue
        idx = b if which == "base" else s
        # restrict to task_ids present in both for a matched comparison
        for tid in (set(b) & set(s)):
            ack, fol = idx[tid]
            if fol and not ack: counts[0] += 1
            elif fol and ack:   counts[1] += 1
            elif (not fol) and ack: counts[2] += 1
            else: counts[3] += 1
    return counts

SEG = ["Uses cue, silent\n(hidden — unfaithful)", "Uses cue, acknowledges\n(disclosed)",
       "Doesn't use, acknowledges\n(corrected + faithful)", "Doesn't use, silent\n(correct, no mention)"]
SEGC = [RED, TEAL, TEAL_LIGHT, NEUTRAL]

# ---------- Figure: stacked bars, baseline vs steered per model ----------
fig, ax = mpl_config.figure(width=8.5)
xs = []; labels = []; pos = 0
data = {}
for model in MODELS:
    for which, tag in [("base", "base"), ("steer", "steer")]:
        c = joint(model, which); frac = c / c.sum()
        data[(model, which)] = frac
        bottom = 0
        for k in range(4):
            ax.bar(pos, frac[k], 0.8, bottom=bottom, color=SEGC[k], edgecolor="#444",
                   linewidth=0.6, zorder=3, label=SEG[k] if pos == 0 else None)
            bottom += frac[k]
        xs.append(pos); labels.append(tag); pos += 1
    pos += 0.6  # gap between models
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
# model labels under each pair
gap = 0
centers = [0.5, 2.5 + 0.6, 4.5 + 1.2]
for cx, model in zip(centers, MODELS):
    ax.text(cx, -0.13, MLAB[model], ha="center", va="top", fontsize=9, fontweight="bold",
            transform=ax.get_xaxis_transform())
ax.set_ylabel("Fraction of traces")
ax.set_ylim(0, 1)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
fig.suptitle("Cue acknowledgement vs. cue use: joint distribution\n"
             "contrastive · GPQA · pooled over 4 cues · $\\alpha{=}5$ · use = picked cued option",
             fontsize=11, fontweight="bold")
mpl_config.save(fig, str(OUT / "fig9_cue_ack_vs_following"), png=True, pdf=False)
plt.close(fig)

# ---------- Markdown table: joint ack x use fractions, baseline vs steered ----------
def write_joint_table_md():
    lines = ["# Cue acknowledgement × cue use — joint fractions\n",
             "Joint distribution of cue acknowledgement (CoT mentions the cue) and cue use "
             "(judge-extracted final answer equals the cued option), baseline vs. steered. Contrastive, "
             "GPQA, pooled over 4 cues, α=5. Columns sum to 1 within a row. **Uses & silent** is "
             "hidden cue use (unfaithful); **Ack rate** is faithfulness; **Use rate** is how often "
             "the model picked the cued option.\n",
             "| Model | Cond. | Uses & silent | Uses & ack | No-use & ack | No-use & silent | Ack rate | Use rate |",
             "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for which, tag in [("base", "baseline"), ("steer", "steered")]:
            f = data[(model, which)]
            ack = f[1] + f[2]; use = f[0] + f[1]
            name = MLAB[model] if which == "base" else ""
            lines.append(f"| {name} | {tag} | {f[0]:.2f} | {f[1]:.2f} | {f[2]:.2f} | "
                         f"{f[3]:.2f} | {ack:.2f} | {use:.2f} |")
    (OUT / "cue_ack_use_joint.md").write_text("\n".join(lines) + "\n")

write_joint_table_md()

# ---------- Appendix: joint fractions across the steering coefficient ----------
# Bigger version of the joint table covering alpha in {2.5, 5, 7.5}. Baseline is
# alpha-independent (computed on the alpha=5 matched set, as in the main text);
# each steered row uses its own alpha's matched set. Emits a markdown copy and a
# \input-able LaTeX table (label tab:joint) for the appendix.
ALPHAS_TBL = ["2.5", "5.0", "7.5"]
def _alpha_lab(a):  # "5.0" -> "5", "2.5" -> "2.5"
    s = a.rstrip("0").rstrip(".")
    return r"$\alpha{=}" + s + "$"

def write_alpha_joint_tables():
    def fr(model, which, alpha):
        c = joint(model, which, alpha); return c / c.sum()
    md = ["# Cue acknowledgement × cue use — joint fractions across steering coefficient\n",
          "Contrastive, GPQA, pooled over 4 cues. Baseline is α-independent; the main text "
          "reports α=5. The four joint cells sum to 1 within a row. **Uses & silent** is hidden "
          "cue use.\n",
          "| Model | Cond. | Uses & silent | Uses & ack | No-use & ack | No-use & silent | Ack rate | Use rate |",
          "|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[H]", r"    \centering\small",
           r"    \setlength{\tabcolsep}{2.5pt}",
           r"    \caption{Joint fractions of cue acknowledgment and cue use (judge-extracted answer)"
           r" across the steering coefficient $\alpha$ (contrastive, GPQA, pooled over 4 cues; the"
           r" four joint cells sum to~1). Baseline is $\alpha$-independent; the $\alpha{=}5$ row is"
           r" the value reported in the main text. ``Uses \& silent'' is hidden cue use.}",
           r"    \label{tab:joint}",
           r"    \begin{tabular}{ll cccc cc}", r"        \toprule",
           r"        Model & Cond. & Uses\,\&\,silent & Uses\,\&\,ack & No-use\,\&\,ack &"
           r" No-use\,\&\,silent & Ack rate & Use rate \\", r"        \midrule"]
    for mi, model in enumerate(MODELS):
        conds = [("base", "base", "5.0")] + [(_alpha_lab(a), "steer", a) for a in ALPHAS_TBL]
        for ci, (lab, which, a) in enumerate(conds):
            f = fr(model, which, a); ack = f[1] + f[2]; use = f[0] + f[1]
            mcell = (r"\multirow{4}{*}{" + MLAB[model] + "}") if ci == 0 else ""
            tex.append(f"        {mcell} & {lab} & {f[0]:.2f} & {f[1]:.2f} & {f[2]:.2f} &"
                       f" {f[3]:.2f} & {ack:.2f} & {use:.2f}" + r" \\")
            md.append(f"| {MLAB[model] if ci==0 else ''} | {'base' if which=='base' else 'α='+a} |"
                      f" {f[0]:.2f} | {f[1]:.2f} | {f[2]:.2f} | {f[3]:.2f} | {ack:.2f} | {use:.2f} |")
        tex.append(r"        \midrule" if mi < len(MODELS) - 1 else r"        \bottomrule")
    tex += [r"    \end{tabular}", r"\end{table}"]
    (OUT / "cue_ack_use_joint_alpha.tex").write_text("\n".join(tex) + "\n")
    (OUT / "cue_ack_use_joint_alpha.md").write_text("\n".join(md) + "\n")
    print("wrote cue_ack_use_joint_alpha.{tex,md}")

write_alpha_joint_tables()

# ---------- print summary ----------
print(f"{'model':14s} {'cond':9s}  hidden  disclosed  ack&corr  clean  | ack_rate use_rate")
for model in MODELS:
    for which, tag in [("base", "baseline"), ("steer", "steered")]:
        f = data[(model, which)]
        print(f"{MLAB[model]:14s} {tag:9s}  {f[0]:.2f}    {f[1]:.2f}      {f[2]:.2f}     {f[3]:.2f}  | "
              f"{f[1]+f[2]:.2f}     {f[0]+f[1]:.2f}")
print("\nwrote fig9_cue_ack_vs_following + cue_ack_use_joint.md")

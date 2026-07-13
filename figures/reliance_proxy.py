#!/usr/bin/env python3
"""Reliance-proxy validation table (tab:proxy): accuracy vs. cue use.

Per model x {baseline, steered}: Correct%, Follow% (= final answer is the cued
option), Other-wrong%, % errors cued (= Follow/(Follow+Other)), and n. Pooled over
datasets, scenarios, methods, and alpha in {1,5} (the steered side reads every
contrastive/optimization/synthetic file at alpha 1.0 or 5.0; baseline is the cued
baselines, uncued excluded).

The chosen letter is the correctness judge's structured model_answer_letter field
(scoring v3+); no regex parsing of the response text.
"""
import json, glob, re
from pathlib import Path
TR = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
from paths import artifacts_dir
OUT = artifacts_dir()
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}

def baseline_files(model):
    return [f for f in glob.glob(str(TR / "runs_scored" / model / "baselines" / "*" / "*" / "*.jsonl"))
            if "/uncued/" not in f]
def steered_files(model, alphas=("1.0", "5.0")):
    rx = re.compile(r"alpha(" + "|".join(a.replace(".", r"\.") for a in alphas) + r")\.jsonl$")
    return [f for f in glob.glob(str(TR / "runs_steered_scored" / "**" / "*.jsonl"), recursive=True)
            if f"/{model}/" in f and rx.search(f)]

def tally(files):
    n = correct = follow = other = 0
    for fp in files:
        for line in open(fp):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            n += 1
            let = r.get("model_answer_letter")
            ct = (r.get("cue_target_letter") or "").strip().upper()
            cs = r.get("correctness_score")
            # Judge-gated: a correct answer is never cue-use (the cue points at a wrong
            # option). Correct -> correct; among incorrect, cued letter -> follow, else other.
            if cs is not None and int(cs) == 1:
                correct += 1
            elif cs is not None and int(cs) == 0:
                if let is not None and ct and let == ct: follow += 1
                elif let is not None: other += 1
    return dict(n=n, correct=correct, follow=follow, other=other)

rows = {}
print(f"{'model':14s} {'cond':6s} {'n':>6s} {'Corr%':>6s} {'Foll%':>6s} {'Other%':>7s} {'%errCued':>9s}")
for m in MODELS:
    for cond, files in [("base", baseline_files(m)),
                        ("steer", steered_files(m))]:
        t = tally(files); rows[(m, cond)] = t
        n, c, f, o = t["n"], t["correct"], t["follow"], t["other"]
        ec = 100 * f / (f + o) if (f + o) else 0
        print(f"{m:14s} {cond:6s} {n:6d} {100*c/n:6.0f} {100*f/n:6.0f} {100*o/n:7.0f} {ec:9.0f}")

# ---- markdown + LaTeX (\input-able as tab:proxy) ----
def cells(m, cond):
    t = rows[(m, cond)]; n, c, f, o = t["n"], t["correct"], t["follow"], t["other"]
    ec = round(100 * f / (f + o)) if (f + o) else 0
    return round(100*c/n), round(100*f/n), round(100*o/n), ec, n

md = ["# Reliance-proxy validation: accuracy vs. cue use\n",
      "Pooled over datasets, scenarios, methods, and alpha in {1,5}. Follow% and "
      "Other-wrong% are shares of all responses; % errors cued = Follow/(Follow+Other). "
      "The chosen letter is the correctness judge's model_answer_letter field.\n",
      "| Model | Cond. | Correct% | Follow% | Other-wrong% | % errors cued | n |",
      "|---|---|--:|--:|--:|--:|--:|"]
tex = [r"\begin{table}[H]", r"    \centering\small",
       r"    \caption{Reliance-proxy validation (pooled over datasets, scenarios, methods, $\alpha\in\{1,5\}$).",
       r"    Follow\% and Other-wrong\% are shares of all responses; \% errors cued $=$ Follow$/$(Follow$+$Other).}",
       r"    \label{tab:proxy}", r"    \begin{tabular}{ll rrrr r}", r"        \toprule",
       r"        Model & Cond. & Correct\% & Follow\% & Other-wrong\% & \% errors cued & $n$ \\",
       r"        \midrule"]
def grp(n): return f"{n:,}".replace(",", "{,}")
for mi, m in enumerate(MODELS):
    for ci, (cond, lab) in enumerate([("base", "base"), ("steer", "steer")]):
        cc, ff, oo, ec, n = cells(m, cond)
        mcell = (r"\multirow{2}{*}{" + MLAB[m] + "}") if ci == 0 else ""
        tex.append(f"        {mcell} & {lab} & {cc} & {ff} & {oo} & {ec} & {grp(n)} " + r"\\")
        md.append(f"| {MLAB[m] if ci==0 else ''} | {lab} | {cc} | {ff} | {oo} | {ec} | {n} |")
    tex.append(r"        \midrule" if mi < len(MODELS) - 1 else r"        \bottomrule")
tex += [r"    \end{tabular}", r"\end{table}"]
(OUT / "reliance_proxy_validation.md").write_text("\n".join(md) + "\n")
(OUT / "reliance_proxy.tex").write_text("\n".join(tex) + "\n")
print("\nwrote reliance_proxy_validation.md + reliance_proxy.tex")

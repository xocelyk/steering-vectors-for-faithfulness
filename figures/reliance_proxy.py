#!/usr/bin/env python3
"""Reliance-proxy validation table (tab:proxy): accuracy vs. cue use.

Per model x {baseline, steered}: Correct%, Follow% (= final answer is the cued
option), Other-wrong%, % errors cued (= Follow/(Follow+Other)), and n. Pooled over
datasets, scenarios, methods, and the swept alpha in {2.5,5,7.5} (the steered side
reads every contrastive/optimization/synthetic file at those alphas; baseline is the
cued baselines, uncued excluded).

The chosen letter is the correctness judge's structured model_answer_letter field
(scoring v3+); no regex parsing of the response text.
"""
import json, glob, re
from scoring import prog_correct
from pathlib import Path
TR = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
from paths import artifacts_dir
OUT = artifacts_dir()
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}

def baseline_files(model):
    return [f for f in glob.glob(str(TR / "runs_scored" / model / "baselines" / "*" / "*" / "*.jsonl"))
            if "/uncued/" not in f]
def steered_files(model, alphas=("2.5", "5.0", "7.5")):
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
            cs = prog_correct(r)
            # Programmatic grading from the extracted letter (scoring.py): a correct
            # answer is never cue-use (the cue points at a wrong option). Correct ->
            # correct; among incorrect, cued letter -> follow, else other.
            if cs == 1:
                correct += 1
            elif cs == 0:
                if ct and let == ct: follow += 1
                else: other += 1
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
# The table reports only the errors-cued statistic the paper cites: it is a
# within-row ratio, so the base/steer population differences (steered rows are
# ~70% GPQA and test-split only; baseline rows are dataset-balanced and include
# train items) do not enter it. The share columns (Correct/Follow/Other) are
# intentionally omitted: cross-row comparisons of them are mixture artifacts.
def cells(m, cond):
    t = rows[(m, cond)]; n, f, o = t["n"], t["follow"], t["other"]
    ec = round(100 * f / (f + o)) if (f + o) else 0
    return ec, n

md = ["# Reliance-proxy validation: errors concentrate on the cued option\n",
      "Pooled over datasets, scenarios, methods, and the swept alpha in {2.5,5,7.5}. "
      "% errors cued = Follow/(Follow+Other), a within-row ratio. The chosen letter is the "
      "judge's model_answer_letter field.\n",
      "| Model | Cond. | % errors cued | n |",
      "|---|---|--:|--:|"]
tex = [r"\begin{table}[H]", r"    \centering\small",
       r"    \caption{Reliance-proxy validation (pooled over datasets, scenarios, methods, and the swept $\alpha\in\{2.5,5,7.5\}$).",
       r"    Errors cued is the share of incorrect answers that select the specifically cued option;",
       r"    with four answer options an incidental error would land on the cued option roughly a third",
       r"    of the time. The statistic is computed within each row, so it is insensitive to the",
       r"    differing dataset mixtures of the baseline and steered pools.}",
       r"    \label{tab:proxy}", r"    \begin{tabular}{@{}ll r r@{}}", r"        \toprule",
       r"        Model & Cond. & Errors cued (\%) & $n$ \\",
       r"        \midrule"]
def grp(n): return f"{n:,}".replace(",", "{,}")
for mi, m in enumerate(MODELS):
    for ci, (cond, lab) in enumerate([("base", "base"), ("steer", "steer")]):
        ec, n = cells(m, cond)
        mcell = (r"\multirow{2}{*}{" + MLAB[m] + "}") if ci == 0 else ""
        tex.append(f"        {mcell} & {lab} & {ec} & {grp(n)} " + r"\\")
        md.append(f"| {MLAB[m] if ci==0 else ''} | {lab} | {ec} | {n} |")
    tex.append(r"        \midrule" if mi < len(MODELS) - 1 else r"        \bottomrule")
tex += [r"    \end{tabular}", r"\end{table}"]
(OUT / "reliance_proxy_validation.md").write_text("\n".join(md) + "\n")
(OUT / "reliance_proxy.tex").write_text("\n".join(tex) + "\n")
print("\nwrote reliance_proxy_validation.md + reliance_proxy.tex")

#!/usr/bin/env python3
"""Generate artifacts/g12_alpha.tex (tab:g12-alpha): Gemma-3 12B delta cue ack,
Stanford cue, by dataset x steering coefficient alpha, with 90% CI. Reads agg.json.
Pairs with tab:g12-corr (mechanism). Caption uses the bare delta convention (no
"net" modifier), matching the rest of the paper."""
import json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
from paths import artifacts_dir
OUT = artifacts_dir()
agg = json.load(open(HERE / "agg.json"))

M = "gemma-3-12b-it"
DS = ["mmlu", "bbh", "gpqa"]          # ordered by effect size, descending
DL = {"bbh": "BBH", "gpqa": "GPQA", "mmlu": "MMLU"}
SCEN = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}
ALPHAS = ["2.5", "5.0", "7.5"]
AL = {"2.5": "2.5", "5.0": "5", "7.5": "7.5"}
Z = 1.645

def find(ds, a):
    c = [x for x in agg if x["alpha"] == a and x["method"] == "contrastive" and x["model"] == M
         and x["scenario"] == SCEN[ds] and x["eval_dataset"] == ds and x["eval_cue"] == "stanford"]
    return c[0] if c else None

def net_ci(c):
    nc, nr, n = c["n_converted"], c["n_regressed"], c["n"]
    net = (nc - nr) / n
    hw = Z * math.sqrt(max((nc + nr - (nc - nr) ** 2 / n) / n ** 2, 0))
    return net, hw

def cell(c):
    if not c: return "--"
    net, hw = net_ci(c)
    return f"${net:+.2f}{{\\pm}}{hw:.2f}$"

L = []
L.append(r"\begin{table}[H]")
L.append(r"    \centering\small")
L.append(r"    \caption{$\Delta_{\mathrm{ack}}$ for Gemma-3 12B by dataset (Stanford cue,")
L.append(r"    contrastive) at each steering coefficient $\alpha$ ($\pm$90\% normal-approximation CI).")
L.append(r"    The per-dataset value is close to constant across $\alpha$. We report")
L.append(r"    this variation as observed and do not explain it; the steering layer also differs by")
L.append(r"    dataset (\cref{tab:g12-corr}, \cref{sec:limits}).}")
L.append(r"    \label{tab:g12-alpha}")
L.append(r"    \begin{tabular}{@{}l r r ccc@{}}")
L.append(r"        \toprule")
L.append(r"        Dataset & Layer & $n$ & $\alpha{=}2.5$ & $\alpha{=}5$ & $\alpha{=}7.5$ \\")
L.append(r"        \midrule")
for ds in DS:
    c5 = find(ds, "5.0")
    meta = f"{c5['layer']} & {c5['n']}" if c5 else "&"
    cells = " & ".join(cell(find(ds, a)) for a in ALPHAS)
    L.append(f"        {DL[ds]} & {meta} & {cells} \\\\")
L.append(r"        \bottomrule")
L.append(r"    \end{tabular}")
L.append(r"\end{table}")

(OUT / "g12_alpha.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))
print("\nwrote", OUT / "g12_alpha.tex")

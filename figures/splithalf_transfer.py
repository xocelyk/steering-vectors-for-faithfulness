#!/usr/bin/env python3
"""Split-half control for the transfer-vs-selfeffect correlation.

A transfer cell (train A -> test E) and the test-setting diagonal (E -> E) are
measured on the same eval items against the same baseline run, so their
sampling noise is shared, which inflates the raw correlation between transfer
delta_ack and the test setting's self-steering delta_ack (the blue placements
in transfer_vs_selfeffect hugging y=x). Control: randomly split E's task_ids
in half, compute the diagonal on one half and the transfer effects on the
other (independent items and baselines), and compare against the same-half
correlation at matched sample size. The remaining gap to 1 is attenuation
from halved n, quantified by the split-half reliabilities of both quantities;
the disattenuated correlation (Spearman correction) estimates the true-score
relationship.

Reads the raw runs_scored/runs_steered_scored trees (contrastive, alpha=5).
Writes splithalf_transfer.tex (appendix table) to the artifacts dir.
"""
import glob
import json
import random
from pathlib import Path

import numpy as np

from paths import artifacts_dir

ROOT = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
STEERED = ROOT / "runs_steered_scored" / "contrastive" / "meek"
BASE = ROOT / "runs_scored"
OUT = artifacts_dir()
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
CUES = ["stanford", "xml", "grader", "insider"]
DS = ["bbh", "gpqa", "mmlu"]
DSCEN = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}
ALPHA = "5.0"
N_SPLITS = 500


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    return None  # LFS pointer / corrupt
    return out


def fscore(r):
    v = r.get("faithfulness_score")
    return int(v) if v is not None else None


def baseline_idx(model, ds, cue):
    recs = []
    for fp in glob.glob(str(BASE / model / "baselines" / ds / cue / "*.jsonl")):
        r = load_jsonl(fp)
        if r:
            recs.extend(r)
    return {r["task_id"]: fscore(r) for r in recs
            if r.get("task_id") is not None and fscore(r) is not None}


def steered_deltas(model, scenario, ds, cue, base):
    """{task_id: steer_faith - base_faith} for the alpha-5 run of one cell."""
    for fp in glob.glob(str(STEERED / model / scenario / ds / cue / "*.jsonl")):
        fn = Path(fp).name
        if "alpha" not in fn:
            continue
        alpha = fn.split("alpha")[-1].replace(".jsonl", "").rstrip(".")
        if alpha != ALPHA:
            continue
        recs = load_jsonl(fp)
        if not recs:
            continue
        d = {}
        for r in recs:
            tid = r.get("task_id")
            s = fscore(r)
            if tid in base and s is not None:
                d[tid] = s - base[tid]
        if d:
            return d
    return None


def build(settings, locate):
    """cells[(model, tr, te)] and diag[(model, s)] -> per-task delta dicts."""
    cells, diag = {}, {}
    for model in MODELS:
        for tr in settings:
            for te in settings:
                scen, ds, cue = locate(tr, te)
                base = baseline_idx(model, ds, cue)
                if not base:
                    continue
                d = steered_deltas(model, scen, ds, cue, base)
                if d is None:
                    continue
                if tr == te:
                    diag[(model, te)] = d
                else:
                    cells[(model, tr, te)] = d
    return cells, diag


def analyze(cells, diag):
    keys = [k for k in cells if (k[0], k[2]) in diag and (k[0], k[1]) in diag]

    # full-data correlation (what the transfer_vs_selfeffect figure shows)
    x = [np.mean(list(diag[(m, te)].values())) for (m, tr, te) in keys]
    y = [np.mean(list(cells[k].values())) for k in keys]
    r_full = np.corrcoef(x, y)[0, 1]

    r_same, r_diff, rel_x, rel_y, r_corr = [], [], [], [], []
    for k_split in range(N_SPLITS):
        rng = random.Random(10_000 + k_split)
        halves = {}
        # sorted: set iteration order is hash-randomized across processes, and the
        # shared rng makes split outcomes depend on it; sorting keeps runs identical
        for (model, te) in sorted({(m, te) for (m, tr, te) in keys}):
            tids = sorted(diag[(model, te)].keys())
            rng.shuffle(tids)
            halves[(model, te)] = set(tids[: len(tids) // 2])
        x1, x2, y1, y2 = [], [], [], []
        for (model, tr, te) in keys:
            h1 = halves[(model, te)]
            dxa = [v for t, v in diag[(model, te)].items() if t in h1]
            dxb = [v for t, v in diag[(model, te)].items() if t not in h1]
            dya = [v for t, v in cells[(model, tr, te)].items() if t in h1]
            dyb = [v for t, v in cells[(model, tr, te)].items() if t not in h1]
            if not (dxa and dxb and dya and dyb):
                continue
            x1.append(np.mean(dxa)); x2.append(np.mean(dxb))
            y1.append(np.mean(dya)); y2.append(np.mean(dyb))
        x1, x2, y1, y2 = map(np.array, (x1, x2, y1, y2))
        rs_ = np.corrcoef(x1, y1)[0, 1]   # shared items+baseline, half n
        rd_ = np.corrcoef(x1, y2)[0, 1]   # disjoint items, half n
        rx_ = np.corrcoef(x1, x2)[0, 1]   # split-half reliability of diagonal
        ry_ = np.corrcoef(y1, y2)[0, 1]   # split-half reliability of transfer
        r_same.append(rs_); r_diff.append(rd_); rel_x.append(rx_); rel_y.append(ry_)
        if rx_ > 0 and ry_ > 0:
            r_corr.append(rd_ / np.sqrt(rx_ * ry_))
    stat = lambda a: (np.mean(a), np.std(a))
    return dict(n=len(keys), r_full=r_full, r_same=stat(r_same),
                r_diff=stat(r_diff), rel_x=stat(rel_x), rel_y=stat(rel_y),
                r_corr=stat(r_corr))


cue_res = analyze(*build(CUES, lambda tr, te: (f"gpqa_{tr}", "gpqa", te)))
ds_res = analyze(*build(DS, lambda tr, te: (DSCEN[tr], te, "stanford")))

pm = lambda s: f"${s[0]:.2f} \\pm {s[1]:.2f}$"
rows = [("Cues (GPQA)", cue_res),
        ("Datasets (Stanford)", ds_res)]
for name, r in rows:
    print(f"{name}: n_cells={r['n']}  full r={r['r_full']:+.3f}  "
          f"same-half={r['r_same'][0]:+.3f}+/-{r['r_same'][1]:.3f}  "
          f"diff-half={r['r_diff'][0]:+.3f}+/-{r['r_diff'][1]:.3f}  "
          f"rel_diag={r['rel_x'][0]:+.3f}  rel_trans={r['rel_y'][0]:+.3f}  "
          f"disattenuated={r['r_corr'][0]:+.3f}+/-{r['r_corr'][1]:.3f}")

tex = [
    "% Split-half control for the transfer-vs-selfeffect correlation.",
    "% Generated by figures/splithalf_transfer.py; mean +/- sd over "
    f"{N_SPLITS} random splits.",
    "\\begin{table}[H]\\centering\\footnotesize",
    "\\setlength{\\tabcolsep}{3pt}",
    "\\caption{Split-half control for the correlation between the"
    " transfer effect and the test setting's self-steering effect"
    " (\\cref{fig:transfer-selfeffect}). Same- and different-half"
    " correlations use half the eval items for both quantities, so they are"
    " directly comparable; the different-half correlation removes the shared"
    " items and baseline. Reliability columns are the split-half"
    " reliabilities of the diagonal and transfer $\\Delta_{\\mathrm{ack}}$;"
    " the last column applies the Spearman attenuation correction to the"
    " different-half correlation. Mean $\\pm$ sd over "
    f"{N_SPLITS} random splits.}}",
    "\\label{tab:splithalf-transfer}",
    "\\begin{tabular}{@{}l r r r r r r r@{}}",
    "\\toprule",
    "Matrix & $n$ & Full $r$ & Same $r$ & Diff $r$ &"
    " Rel.\\ diag & Rel.\\ transf. & Disatt.\\ $r$ \\\\",
    "\\midrule",
]
for name, r in rows:
    tex.append(f"{name} & {r['n']} & ${r['r_full']:.2f}$ & {pm(r['r_same'])} &"
               f" {pm(r['r_diff'])} & {pm(r['rel_x'])} & {pm(r['rel_y'])} &"
               f" {pm(r['r_corr'])} \\\\")
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(OUT / "splithalf_transfer.tex").write_text("\n".join(tex) + "\n")
print(f"wrote {OUT / 'splithalf_transfer.tex'}")

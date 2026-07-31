#!/usr/bin/env python3
"""Aggregate scored steered traces vs. baselines into paper tables.

Output: figures/agg.json  (all per-cell stats)
        printed summary tables.
"""
import json, os, glob, collections
from pathlib import Path

from scoring import prog_correct

ROOT = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
STEERED = ROOT / "runs_steered_scored"
BASE = ROOT / "runs_scored"

MODELS = ["gemma-3-4b-it", "gemma-3-12b-it", "qwen3.5-9b"]
CUES = ["stanford", "xml", "grader", "insider"]
DATASETS = ["bbh", "gpqa", "mmlu"]

def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                return None  # LFS pointer / corrupt
    return out

def index_by_task(records):
    d = {}
    for r in records:
        tid = r.get("task_id")
        if tid is not None:
            d[tid] = r
    return d

# ---- Load baselines: baseline[model][dataset][cue] = {task_id: record}
baseline = collections.defaultdict(lambda: collections.defaultdict(dict))
for model in MODELS:
    for ds in DATASETS:
        for cue in CUES:
            files = glob.glob(str(BASE / model / "baselines" / ds / cue / "*.jsonl"))
            recs = []
            for fp in files:
                r = load_jsonl(fp)
                if r:
                    recs.extend(r)
            if recs:
                baseline[model][ds][cue] = index_by_task(recs)

def score(r, key):
    v = r.get(key)
    return int(v) if v is not None else None

def cell_stats(steered_recs, base_idx):
    """Compare steered vs baseline over matched task_ids."""
    n=0; bf=0; sf=0
    unfaith_base=0; converted=0
    faith_base=0; regressed=0
    bacc=sacc=0; nacc=0
    bdeg=sdeg=0; ndeg=0
    for r in steered_recs:
        tid = r.get("task_id")
        b = base_idx.get(tid)
        if b is None:
            continue
        sfs = score(r, "faithfulness_score"); bfs = score(b, "faithfulness_score")
        if sfs is None or bfs is None:
            continue
        n += 1
        bf += bfs; sf += sfs
        if bfs == 0:
            unfaith_base += 1
            if sfs == 1: converted += 1
        else:
            faith_base += 1
            if sfs == 0: regressed += 1
        # accuracy: programmatic letter-match grading (see scoring.py)
        ba = prog_correct(b); sa = prog_correct(r)
        if ba is not None and sa is not None:
            nacc += 1; bacc += ba; sacc += sa
        # degeneracy
        bd = score(b, "degenerate_score"); sd = score(r, "degenerate_score")
        if bd is not None and sd is not None:
            ndeg += 1; bdeg += bd; sdeg += sd
    if n == 0:
        return None
    return dict(
        n=n,
        base_faith=bf/n, steer_faith=sf/n,
        n_converted=converted, n_regressed=regressed,
        n_unfaith_base=unfaith_base,
        conversion=(converted/unfaith_base) if unfaith_base else None,
        n_faith_base=faith_base,
        regression=(regressed/faith_base) if faith_base else None,
        n_acc=nacc, n_deg=ndeg,
        acc_delta=((sacc-bacc)/nacc) if nacc else None,
        base_acc=(bacc/nacc) if nacc else None, steer_acc=(sacc/nacc) if nacc else None,
        degen_delta=((sdeg-bdeg)/ndeg) if ndeg else None,
        base_degen=(bdeg/ndeg) if ndeg else None, steer_degen=(sdeg/ndeg) if ndeg else None,
    )

# ---- Walk steered tree
results = []  # list of dict rows
method_dirs = {
    "contrastive": STEERED / "contrastive" / "meek",
    "synthetic": STEERED / "synthetic" / "meek",
    "opt-specific": STEERED / "optimization" / "specific__eval-meek",
    "opt-generic": STEERED / "optimization" / "generic__eval-meek",
}

for method, mdir in method_dirs.items():
    if not mdir.exists():
        continue
    for model in MODELS:
        moddir = mdir / model
        if not moddir.exists():
            continue
        for scenario in sorted(os.listdir(moddir)):
            sdir = moddir / scenario
            if not sdir.is_dir():
                continue
            for fp in glob.glob(str(sdir / "*" / "*" / "*.jsonl")):
                parts = Path(fp).parts
                eval_cue = parts[-2]
                eval_ds = parts[-3]
                alpha = None
                fn = Path(fp).name
                if "alpha" in fn:
                    alpha = fn.split("alpha")[-1].replace(".jsonl","").rstrip(".")
                recs = load_jsonl(fp)
                if not recs:
                    continue
                base_idx = baseline.get(model,{}).get(eval_ds,{}).get(eval_cue,{})
                if not base_idx:
                    continue
                st = cell_stats(recs, base_idx)
                if st is None:
                    continue
                row = dict(method=method, model=model, scenario=scenario,
                           eval_dataset=eval_ds, eval_cue=eval_cue, alpha=alpha,
                           layer=recs[0].get("layer"), **st)
                results.append(row)

out = Path(__file__).resolve().parent / "agg.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {len(results)} cells to {out}")

# Quick alpha comparison: mean conversion by method+alpha
print("\n=== mean conversion / acc_delta / degen_delta by method, alpha ===")
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in results:
    k=(r["method"], r["alpha"])
    if r["conversion"] is not None: agg[k]["conv"].append(r["conversion"])
    if r["acc_delta"] is not None: agg[k]["acc"].append(r["acc_delta"])
    if r["degen_delta"] is not None: agg[k]["deg"].append(r["degen_delta"])
for k in sorted(agg):
    d=agg[k]
    mean=lambda x:sum(x)/len(x) if x else float('nan')
    print(f"{k[0]:14s} a={k[1]:4s}  conv={mean(d['conv']):.3f} (n={len(d['conv'])})  "
          f"acc_d={mean(d['acc']):+.3f}  degen_d={mean(d['deg']):+.3f}")

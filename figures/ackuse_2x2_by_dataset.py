#!/usr/bin/env python3
"""Gemma-3 12B: ack x use 2x2 joint, by dataset (Stanford cue), baseline vs steered.
Datasets were varied only on the Stanford cue, so we use the matched Stanford-cue
contrastive vectors per dataset. Saves a markdown artifact."""
import json, glob
from scoring import prog_correct
from pathlib import Path
import numpy as np

TR = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
from paths import artifacts_dir
OUT = artifacts_dir()
MODEL = "gemma-3-12b-it"

def load(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except: continue
        if r.get("faithfulness_score") is None: continue
        let = r.get("model_answer_letter")
        ct = (r.get("cue_target_letter") or "").strip().upper()
        # programmatic tri-state grading: True / False / None (no letter -> excluded
        # from acc); see scoring.py
        cs = prog_correct(r)
        correct = (cs == 1) if cs is not None else None
        # USE = the extracted letter is the cued letter. No correctness gate: the cue
        # points at a wrong option, so the cued letter is never correct under
        # programmatic grading.
        out[r["task_id"]] = dict(ack=r["faithfulness_score"] == 1,
                                 used=(let is not None and ct != "" and let == ct),
                                 correct=correct)
    return out

DS = ["bbh", "gpqa", "mmlu"]
SCEN = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}

def joint(idx):
    # returns dict with fractions of the 4 cells + marginals
    n = len(idx)
    us = sum(1 for v in idx.values() if v["used"] and not v["ack"])      # used & silent (hidden)
    ua = sum(1 for v in idx.values() if v["used"] and v["ack"])          # used & ack (disclosed)
    na = sum(1 for v in idx.values() if not v["used"] and v["ack"])      # not-used & ack
    ns = sum(1 for v in idx.values() if not v["used"] and not v["ack"])  # not-used & silent
    used = [v for v in idx.values() if v["used"]]
    notused = [v for v in idx.values() if not v["used"]]
    # accuracy: exclude traces with no extracted answer letter from the denominator
    scored = [v["correct"] for v in idx.values() if v["correct"] is not None]
    return dict(n=n, hidden=us/n, disclosed=ua/n, ackcorr=na/n, clean=ns/n,
                ack=(ua+na)/n, use=(us+ua)/n,
                ack_g_use=(ua/len(used) if used else float("nan")),
                ack_g_notuse=(na/len(notused) if notused else float("nan")),
                acc=(np.mean(scored) if scored else float("nan")))

rows = []
print(f"{'dataset':6s} {'cond':5s} {'n':>4s} {'hidden':>7s} {'discl':>6s} {'ack&¬use':>9s} {'clean':>6s} | {'ack':>5s} {'use':>5s} {'ack|use':>8s} {'ack|¬use':>9s} {'acc':>5s}")
for ds in DS:
    bp = glob.glob(str(TR / "runs_scored" / MODEL / "baselines" / ds / "stanford" / "*.jsonl"))
    # Pin alpha=5 (dir holds one file per steering coef; glob order is unsorted, so a
    # bare *.jsonl + [0] would load a nondeterministic, often-wrong alpha).
    sp = glob.glob(str(TR / "runs_steered_scored" / "contrastive" / "meek" / MODEL / SCEN[ds] / ds / "stanford" / "*alpha5.0.jsonl"))
    if not bp or not sp:
        print(f"{ds}: missing ({bool(bp)},{bool(sp)})"); continue
    b = load(bp[0]); s = load(sp[0])
    ids = set(b) & set(s)
    bidx = {i: b[i] for i in ids}; sidx = {i: s[i] for i in ids}
    # Accuracy on the matched both-scored subset, so base and steer share a
    # denominator and agree with agg.json / layer_analysis (which require both
    # base and steer correctness to be non-null). joint()'s per-condition acc
    # otherwise uses a slightly larger base denominator and disagrees by ~0.01.
    acc_ids = [i for i in ids if bidx[i]["correct"] is not None and sidx[i]["correct"] is not None]
    macc = {"base":  np.mean([bidx[i]["correct"] for i in acc_ids]) if acc_ids else float("nan"),
            "steer": np.mean([sidx[i]["correct"] for i in acc_ids]) if acc_ids else float("nan")}
    for cond, idx in [("base", bidx), ("steer", sidx)]:
        j = joint(idx); j["acc"] = macc[cond]; rows.append((ds, cond, j))
        print(f"{ds:6s} {cond:5s} {j['n']:4d} {j['hidden']:7.2f} {j['disclosed']:6.2f} {j['ackcorr']:9.2f} "
              f"{j['clean']:6.2f} | {j['ack']:5.2f} {j['use']:5.2f} {j['ack_g_use']:8.2f} {j['ack_g_notuse']:9.2f} {j['acc']:5.2f}")

# ---- correlates: probe AUC, layer, conversion/regression/net (from agg.json) ----
def probe_auc(ds):
    f = TR / "probes" / "meek" / f"{MODEL}__{ds}__stanford.json"
    if not f.exists(): return None, None
    d = json.load(open(f)); return d["best_by_roc"]["test_roc_auc"], d["best_layer"]
_agg = json.load(open(Path(__file__).resolve().parent / "agg.json"))
def agg_cell(ds):
    scen = SCEN[ds]
    c = [x for x in _agg if x["alpha"] == "5.0" and x["method"] == "contrastive" and x["model"] == MODEL
         and x["scenario"] == scen and x["eval_dataset"] == ds and x["eval_cue"] == "stanford"]
    return c[0] if c else None
# per-dataset base/steer ack for the ack gain
ackrate = {}
for ds, cond, j in rows:
    ackrate.setdefault(ds, {})[cond] = j["ack"]

# ---- markdown ----
L = ["# Gemma-3 12B: cue acknowledgment x cue use, by dataset (Stanford cue)\n",
     "Joint distribution of acknowledgment and cue use, baseline vs. steered, for Gemma-3 12B on each "
     "dataset, using the matched Stanford-cue contrastive vector (datasets were varied only on the "
     "Stanford cue). Matched by task_id. The four joint cells sum to 1. **Hidden** = used & not "
     "acknowledged (the unfaithful cell); **Ack** = acknowledgment rate; **Use** = picked the cued option.\n",
     "## Joint distribution (base vs. steered)\n",
     "| Dataset | Cond. | n | Hidden (use,¬ack) | Disclosed (use,ack) | ¬use,ack | ¬use,¬ack | Ack | Use | Ack\\|use | Ack\\|¬use | Acc |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for ds, cond, j in rows:
    L.append(f"| {ds.upper()} | {cond} | {j['n']} | {j['hidden']:.2f} | {j['disclosed']:.2f} | "
             f"{j['ackcorr']:.2f} | {j['clean']:.2f} | {j['ack']:.2f} | {j['use']:.2f} | "
             f"{j['ack_g_use']:.2f} | {j['ack_g_notuse']:.2f} | {j['acc']:.2f} |")

L += ["\n## Correlates of the per-dataset effect\n",
      "One row per dataset (Stanford cue). **Probe AUC** = test ROC-AUC of the cue-acknowledgment "
      "probe at the selected layer; **Conv/Regr** are the converted and regressed fractions of all "
      "traces (same denominator as Net, so Conv − Regr = Net up to rounding); **Ack gain** = steered "
      "minus baseline acknowledgment rate (from the joint table above).\n",
      "| Dataset | Probe AUC | Layer | Base Ack | Base Acc | Conv | Regr | Net | Ack gain |",
      "|---|---|---|---|---|---|---|---|---|"]
for ds in DS:
    auc, layer = probe_auc(ds); c = agg_cell(ds)
    if c is None: continue
    gain = ackrate[ds]["steer"] - ackrate[ds]["base"]
    L.append(f"| {ds.upper()} | {auc:.2f} | L{layer} | {c['base_faith']:.2f} | {c['base_acc']:.2f} | "
             f"{c['n_converted']/c['n']:.2f} | {c['n_regressed']/c['n']:.2f} | {c['steer_faith']-c['base_faith']:+.2f} | {gain:+.2f} |")

L += ["\n## Takeaways\n",
      "- **Tracks probe AUC (decodability).** MMLU has the most decodable acknowledgment direction "
      "(0.75 > 0.68 BBH > 0.61 GPQA) and the largest effect; the effect orders with AUC across these "
      "three datasets. Consistent with the probes being strongest on MMLU.",
      "- **High conversion *with* low regression.** MMLU both converts a lot (0.29 of traces) and "
      "breaks little (0.03 regressed); GPQA converts 0.22 but regresses 0.16 (churn), so its net "
      "washes out; BBH does neither. Low regression is the signature of a well-identified direction.",
      "- **Not explained by baseline accuracy** (BBH 0.64 vs MMLU 0.62, very different effect) or by "
      "acknowledgment headroom (base ack 0.60/0.55/0.62 is similar).",
      "- **Caveats.** n = 3 datasets, one cue; the layer is confounded with the dataset (MMLU L10, "
      "BBH L3, GPQA L15), so this could be a layer effect; and globally probe AUC does NOT predict net "
      "(Pearson r ~ -0.09 across all 24 scenarios), so the AUC ordering here may be slice-specific or "
      "noise. Treat as a hypothesis for the fixed-vector layer sweep."]
(OUT / "ackuse_2x2_gemma12b_by_dataset.md").write_text("\n".join(L) + "\n")
print("\nwrote ackuse_2x2_gemma12b_by_dataset.md")

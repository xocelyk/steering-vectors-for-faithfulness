#!/usr/bin/env python3
"""Layer analysis: relate each scenario's steering layer to outcomes.
Writes a markdown table per model to the paper's artifacts/ (layer_analysis.md).
Method = contrastive, alpha=5 (layer is shared across methods for a scenario)."""
import json
from pathlib import Path
import torch

VEC = Path(__file__).resolve().parent.parent / "experiments" / "transfer" / "vectors" / "contrastive" / "meek"

def probe_auc(model, scenario):
    """Test ROC-AUC of the probe at the layer that was selected for this scenario."""
    p = VEC / model / f"{scenario}.pt"
    if not p.exists():
        return None
    try:
        o = torch.load(str(p), map_location="cpu", weights_only=False)
        return o.get("layer_test_roc_auc")
    except Exception:
        return None

HERE = Path(__file__).resolve().parent
from paths import artifacts_dir
OUT = artifacts_dir()
rows = json.load(open(HERE / "agg.json"))
A = "5.0"; METHOD = "contrastive"

MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
CUES = ["stanford", "xml", "grader", "insider"]
DS = ["bbh", "gpqa", "mmlu"]

def get(model, scenario, ed, ec):
    return [r for r in rows if r["alpha"] == A and r["method"] == METHOD
            and r["model"] == model and r["scenario"] == scenario
            and r["eval_dataset"] == ed and r["eval_cue"] == ec]

# scenario -> list of (eval_dataset, eval_cue) cells that define its "primary" evaluation
def scenario_cells(scenario):
    if scenario.startswith("gpqa_") and scenario != "gpqa_all":
        cue = scenario.split("_")[1]
        return [("gpqa", cue)]
    if scenario == "gpqa_all":
        return [("gpqa", c) for c in CUES]
    if scenario == "stanford_bbh":
        return [("bbh", "stanford")]
    if scenario == "stanford_mmlu":
        return [("mmlu", "stanford")]
    if scenario == "stanford_all":
        return [(d, "stanford") for d in DS]
    return []

# label per scenario
SLAB = {"gpqa_stanford": "GPQA · Stanford", "gpqa_xml": "GPQA · XML",
        "gpqa_grader": "GPQA · Grader", "gpqa_insider": "GPQA · Unethical",
        "gpqa_all": "GPQA · all-cues (unified)",
        "stanford_bbh": "BBH · Stanford", "stanford_mmlu": "MMLU · Stanford",
        "stanford_all": "all-data · Stanford (unified)"}
SCENARIOS = list(SLAB)

def pool(model, scenario):
    cells = []
    for ed, ec in scenario_cells(scenario):
        cells += get(model, scenario, ed, ec)
    if not cells:
        return None
    nconv = sum(c["n_converted"] for c in cells)
    nunf = sum(c["n_unfaith_base"] for c in cells)
    nregr = sum(c["n_regressed"] for c in cells)
    nfaith = sum(c["n_faith_base"] for c in cells)
    n = sum(c["n"] for c in cells)
    nacc = sum(c["n_acc"] for c in cells)
    ndeg = sum(c["n_deg"] for c in cells)
    bacc = sum((c["base_acc"] or 0) * c["n_acc"] for c in cells) / nacc if nacc else float("nan")
    sacc = sum((c["steer_acc"] or 0) * c["n_acc"] for c in cells) / nacc if nacc else float("nan")
    bdeg = sum((c["base_degen"] or 0) * c["n_deg"] for c in cells) / ndeg if ndeg else float("nan")
    sdeg = sum((c["steer_degen"] or 0) * c["n_deg"] for c in cells) / ndeg if ndeg else float("nan")
    # Converted/regressed as fractions of all n traces (same denominator as Δ),
    # so the columns decompose the change: Conv - Regr = Δ up to rounding.
    return dict(layer=cells[0]["layer"], n=n,
                conv=nconv / n,
                regr=nregr / n,
                net=(nconv - nregr) / n,
                base_acc=bacc, steer_acc=sacc, dacc=sacc - bacc,
                ddeg=sdeg - bdeg)

def model_table(model):
    data = []
    for sc in SCENARIOS:
        p = pool(model, sc)
        if p:
            data.append((sc, p))
    data.sort(key=lambda kv: kv[1]["layer"])  # sort by steering layer

    lines = [f"## {MLAB[model]}\n",
             "| Layer | Probe AUC | Scenario (eval) | n | Conv | Regr | Δ | Acc base→steer | ΔAcc | ΔDeg |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for sc, p in data:
        auc = probe_auc(model, sc)
        auc_s = f"{auc:.2f}" if auc is not None else "—"
        lines.append(
            f"| {p['layer']} | {auc_s} | {SLAB[sc]} | {p['n']} | {p['conv']:.2f} | "
            f"{p['regr']:.2f} | {p['net']:+.2f} | {p['base_acc']:.2f}→{p['steer_acc']:.2f} | "
            f"{p['dacc']:+.2f} | {p['ddeg']:+.2f} |"
        )
    return lines, data

doc = ["# Steering outcome by selected layer\n",
       "Per scenario, the steering layer (argmax probe test ROC-AUC) vs. outcomes, for the "
       "contrastive vectors at α=5. Rows are sorted by steering layer. **Conv** and **Regr** are the "
       "converted and regressed fractions of all n traces, so Conv − Regr = **Δ**, the overall "
       "change in cue-acknowledgment rate (steered − baseline), up to rounding (positive Δ = "
       "steering improves faithfulness overall). **Acc** = task accuracy, **Deg** = degeneracy.\n",
       "**Caveat.** Probe train ROC-AUC = 1.00 in every cell (overfit), and selected-layer AUC does "
       "not track Δ. Each vector was only steered at its own probe-selected layer, and that layer "
       "is confounded with the scenario (cue/dataset) — so this table cannot attribute outcomes to "
       "layer vs. vector/cue source. Low-accuracy rows are the hard GPQA-cue scenarios (baseline-low), "
       "not a steering-induced collapse. Separating layer from vector/cue needs a fixed-vector layer "
       "sweep.\n"]
for m in MODELS:
    lines, d = model_table(m)
    doc += lines + [""]
    print(f"{m}: {len(d)} scenarios, layers {sorted(p['layer'] for _,p in d)}")
(OUT / "layer_analysis.md").write_text("\n".join(doc) + "\n")
print(f"\nWrote layer_analysis.md to {OUT}")

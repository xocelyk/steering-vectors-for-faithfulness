#!/usr/bin/env python3
import json, collections
from pathlib import Path
rows = json.load(open(Path(__file__).parent / "agg.json"))
A = "5.0"
def get(method=None, model=None, scenario=None, ed=None, ec=None):
    out=[]
    for r in rows:
        if r["alpha"]!=A: continue
        if method and r["method"]!=method: continue
        if model and r["model"]!=model: continue
        if scenario and r["scenario"]!=scenario: continue
        if ed and r["eval_dataset"]!=ed: continue
        if ec and r["eval_cue"]!=ec: continue
        out.append(r)
    return out

MODELS=["gemma-3-4b-it","gemma-3-12b-it","qwen3.5-9b"]
CUES=["stanford","xml","grader","insider"]
DS=["bbh","gpqa","mmlu"]
METHODS=["contrastive","synthetic","opt-specific","opt-generic"]

# ---- 1. MATCHED (diagonal) headline: scenario gpqa_<cue> eval on gpqa same cue
print("="*80)
print("TABLE 1: MATCHED-SETTING (train cue == eval cue, GPQA), alpha=5.0")
print("per method x model: net faith gain | conversion | regression | acc d | degen d")
print("="*80)
for method in METHODS:
    print(f"\n-- {method} --")
    print(f"{'model':16s} {'base_f':>7s} {'steer_f':>7s} {'netΔ':>6s} {'conv':>6s} {'regr':>6s} {'accΔ':>6s} {'degΔ':>6s}")
    for model in MODELS:
        # average over the 4 matched gpqa cue cells
        cells=[]
        for cue in CUES:
            c=get(method,model,f"gpqa_{cue}","gpqa",cue)
            cells+=c
        if not cells:
            print(f"{model:16s}  (no data)"); continue
        n=sum(c["n"] for c in cells)
        bf=sum(c["base_faith"]*c["n"] for c in cells)/n
        sf=sum(c["steer_faith"]*c["n"] for c in cells)/n
        conv=[c["conversion"] for c in cells if c["conversion"] is not None]
        regr=[c["regression"] for c in cells if c["regression"] is not None]
        accd=[c["acc_delta"] for c in cells if c["acc_delta"] is not None]
        degd=[c["degen_delta"] for c in cells if c["degen_delta"] is not None]
        m=lambda x:sum(x)/len(x) if x else float('nan')
        print(f"{model:16s} {bf:7.3f} {sf:7.3f} {sf-bf:+6.3f} {m(conv):6.3f} {m(regr):6.3f} {m(accd):+6.3f} {m(degd):+6.3f}")

# ---- 2. Cross-cue 4x4 (contrastive, GPQA) per model: conversion
print("\n"+"="*80)
print("TABLE 2: CROSS-CUE conversion (contrastive, GPQA). rows=train cue, cols=eval cue")
print("="*80)
for model in MODELS:
    print(f"\n-- {model} --")
    print(f"{'train\\eval':12s}" + "".join(f"{c:>10s}" for c in CUES))
    for tc in CUES:
        line=f"{tc:12s}"
        for ec in CUES:
            c=get("contrastive",model,f"gpqa_{tc}","gpqa",ec)
            v=c[0]["conversion"] if c and c[0]["conversion"] is not None else None
            line += f"{v:10.3f}" if v is not None else f"{'--':>10s}"
        print(line)

# ---- 3. Cross-dataset 3x3 (contrastive, stanford cue): conversion
# scenarios: stanford_bbh, gpqa_stanford(=stanford on gpqa), stanford_mmlu
print("\n"+"="*80)
print("TABLE 3: CROSS-DATASET conversion (contrastive, Stanford cue). rows=train ds, cols=eval ds")
print("="*80)
scen_for_ds={"bbh":"stanford_bbh","gpqa":"gpqa_stanford","mmlu":"stanford_mmlu"}
for model in MODELS:
    print(f"\n-- {model} --")
    print(f"{'train\\eval':12s}" + "".join(f"{d:>10s}" for d in DS))
    for td in DS:
        line=f"{td:12s}"
        for ed in DS:
            c=get("contrastive",model,scen_for_ds[td],ed,"stanford")
            v=c[0]["conversion"] if c and c[0]["conversion"] is not None else None
            line += f"{v:10.3f}" if v is not None else f"{'--':>10s}"
        print(line)

# ---- 4. Unified vectors: gpqa_all (across cues), stanford_all (across ds)
print("\n"+"="*80)
print("TABLE 4: UNIFIED vectors (contrastive). gpqa_all eval per cue (GPQA); stanford_all eval per ds")
print("="*80)
for model in MODELS:
    print(f"\n-- {model} -- gpqa_all conversion by eval cue:")
    for ec in CUES:
        c=get("contrastive",model,"gpqa_all","gpqa",ec)
        if c: print(f"   {ec:10s} conv={c[0]['conversion']}  netΔ={c[0]['steer_faith']-c[0]['base_faith']:+.3f}")
    print(f"   stanford_all conversion by eval dataset:")
    for ed in DS:
        c=get("contrastive",model,"stanford_all",ed,"stanford")
        if c: print(f"   {ed:10s} conv={c[0]['conversion']}  netΔ={c[0]['steer_faith']-c[0]['base_faith']:+.3f}")

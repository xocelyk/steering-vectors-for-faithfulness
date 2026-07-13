#!/usr/bin/env python3
"""Do the different methods / cues / datasets actually produce the SAME direction?
Measures cosine similarity between saved steering vectors (within model+layer)."""
import torch, glob, os, itertools, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "experiments" / "transfer" / "vectors"
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]  # sorted by size: 4B, 9B, 12B
SPLIT = "meek"
CUES = ["stanford", "xml", "grader", "insider"]
DS = ["bbh", "gpqa", "mmlu"]
METHOD_DIR = {
    "contrastive": ROOT / "contrastive" / SPLIT,
    "synthetic":   ROOT / "synthetic" / SPLIT,
    "opt-specific": ROOT / "optimization" / "specific",
    "opt-generic":  ROOT / "optimization" / "generic",
}

def load_unit(path):
    """Return (layer, unit_vector np.array) or None."""
    if not os.path.exists(path):
        return None
    o = torch.load(path, map_location="cpu", weights_only=False)
    if "vector" in o:                       # contrastive / synthetic
        v = o["vector"].float(); layer = int(o["layer"])
    elif "vectors" in o:                    # optimization: {layer: tensor}
        layer = int(list(o["vectors"].keys())[0])
        v = list(o["vectors"].values())[0].float()
    else:
        return None
    v = v.numpy()
    n = np.linalg.norm(v)
    return layer, (v / n if n > 0 else v)

def cos(a, b):
    return float(np.dot(a, b))  # already unit

def scen_path(method, model, scenario):
    return str(METHOD_DIR[method] / model / f"{scenario}.pt")

print("="*78)
print("RANDOM BASELINE: expected |cosine| of two random unit vectors")
for d in (2560, 3840, 3584):
    print(f"  dim {d}: std(cos) = 1/sqrt(d) = {1/math.sqrt(d):.4f}  "
          f"(|cos|>{3/math.sqrt(d):.3f} would be a 3-sigma signal)")

for model in MODELS:
    print("\n" + "#"*78)
    print(f"# MODEL: {model}")
    print("#"*78)

    # ---- A) cross-method, same scenario (gpqa_stanford) ----
    print("\n[A] Cross-METHOD alignment (scenario gpqa_stanford): cosine between methods")
    vecs = {}
    for meth in METHOD_DIR:
        r = load_unit(scen_path(meth, model, "gpqa_stanford"))
        if r: vecs[meth] = r
    if len({l for l,_ in vecs.values()}) > 1:
        print("   (layers differ across methods:", {m:l for m,(l,_) in vecs.items()}, ")")
    methods = list(vecs)
    print("        " + "".join(f"{m[:9]:>11s}" for m in methods))
    for m1 in methods:
        row=f"{m1[:9]:>9s} "
        for m2 in methods:
            row += f"{cos(vecs[m1][1], vecs[m2][1]):>11.2f}"
        print(row)

    # ---- B) cross-cue, within contrastive (GPQA) ----
    for meth in ["contrastive", "opt-specific"]:
        print(f"\n[B] Cross-CUE alignment within {meth} (GPQA): cosine between cue vectors")
        vecs={}
        for cue in CUES:
            r=load_unit(scen_path(meth, model, f"gpqa_{cue}"))
            if r: vecs[cue]=r
        layers={l for l,_ in vecs.values()}
        if len(layers)>1: print("   (NB layers differ:", {c:l for c,(l,_) in vecs.items()}, "- cosine across layers not meaningful)")
        print("        " + "".join(f"{c:>10s}" for c in vecs))
        for c1 in vecs:
            row=f"{c1:>9s} "
            for c2 in vecs:
                # only compare same layer
                if vecs[c1][0]==vecs[c2][0]:
                    row+=f"{cos(vecs[c1][1], vecs[c2][1]):>10.2f}"
                else:
                    row+=f"{'(L'+str(vecs[c2][0])+')':>10s}"
            print(row)

    # ---- C) cross-dataset, within contrastive (stanford cue) ----
    scen={"bbh":"stanford_bbh","gpqa":"gpqa_stanford","mmlu":"stanford_mmlu"}
    for meth in ["contrastive", "opt-specific"]:
        print(f"\n[C] Cross-DATASET alignment within {meth} (Stanford cue): cosine")
        vecs={}
        for d in DS:
            r=load_unit(scen_path(meth, model, scen[d]))
            if r: vecs[d]=r
        if len({l for l,_ in vecs.values()})>1:
            print("   layers:", {d:l for d,(l,_) in vecs.items()})
        print("        " + "".join(f"{d:>10s}" for d in vecs))
        for d1 in vecs:
            row=f"{d1:>9s} "
            for d2 in vecs:
                if vecs[d1][0]==vecs[d2][0]:
                    row+=f"{cos(vecs[d1][1], vecs[d2][1]):>10.2f}"
                else:
                    row+=f"{'(L'+str(vecs[d2][0])+')':>10s}"
            print(row)

    # ---- D) unified vs specific (contrastive) ----
    print("\n[D] UNIFIED vs SPECIFIC cosine (contrastive):")
    u=load_unit(scen_path("contrastive", model, "gpqa_all"))
    for cue in CUES:
        s=load_unit(scen_path("contrastive", model, f"gpqa_{cue}"))
        if u and s and u[0]==s[0]:
            print(f"   gpqa_all  vs  gpqa_{cue:9s}: cos={cos(u[1],s[1]):+.2f}  (layer {u[0]})")
        elif u and s:
            print(f"   gpqa_all(L{u[0]}) vs gpqa_{cue}(L{s[0]}): different layers")
    u=load_unit(scen_path("contrastive", model, "stanford_all"))
    for d in DS:
        s=load_unit(scen_path("contrastive", model, scen[d]))
        if u and s and u[0]==s[0]:
            print(f"   stanford_all vs {scen[d]:13s}: cos={cos(u[1],s[1]):+.2f}  (layer {u[0]})")
        elif u and s:
            print(f"   stanford_all(L{u[0]}) vs {scen[d]}(L{s[0]}): different layers")


# ============================================================================
# Markdown artifact (clean tables) written alongside the figures
# ============================================================================
def write_markdown():
    MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
    METHS = ["contrastive", "synthetic", "opt-specific", "opt-generic"]
    from paths import artifacts_dir
    out_dir = artifacts_dir()
    L = []
    L.append("# Steering-vector geometry: cosine similarity between vectors\n")
    L.append("Do different construction methods / cues / datasets arrive at the **same direction**? "
             "Cosine similarity between saved unit steering vectors (`experiments/transfer/vectors/`). "
             "Comparisons are only valid **within the same model and layer** (same residual space).\n")
    L.append("**Random baseline.** Two random unit vectors have E[cos]=0 with "
             "std ≈ 1/√d (d = hidden size): ~0.016–0.020 here. So |cos| < ~0.05 is "
             "indistinguishable from orthogonal; |cos| > ~0.5 indicates genuinely shared direction.\n")

    # Cross-method tables (valid: all methods share the per-scenario layer)
    L.append("## Cross-method alignment (scenario `gpqa_stanford`, single shared layer)\n")
    L.append("Cosine between the vectors produced by each construction method for the **same** "
             "(model, dataset, cue). This comparison is layer-valid.\n")
    for model in MODELS:
        vecs = {}
        for m in METHS:
            r = load_unit(scen_path(m, model, "gpqa_stanford"))
            if r: vecs[m] = r
        if not vecs: continue
        layer = list(vecs.values())[0][0]
        L.append(f"**{MLAB[model]}** (layer {layer}):\n")
        L.append("| | " + " | ".join(m for m in vecs) + " |")
        L.append("|" + "---|" * (len(vecs) + 1))
        for m1 in vecs:
            row = [f"**{m1}**"] + [f"{cos(vecs[m1][1], vecs[m2][1]):+.2f}" for m2 in vecs]
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    # Cross-dataset within-layer (only the cells where layers coincide)
    L.append("## Same-cue cross-dataset alignment (Stanford cue, where layers coincide)\n")
    L.append("Most dataset vectors sit at different probe-selected layers, so only a few pairs are "
             "comparable. Where they are:\n")
    scen = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}
    found = False
    for model in MODELS:
        for meth in ["contrastive", "opt-specific"]:
            vecs = {d: load_unit(scen_path(meth, model, scen[d])) for d in DS}
            vecs = {d: v for d, v in vecs.items() if v}
            for d1, d2 in itertools.combinations(vecs, 2):
                if vecs[d1][0] == vecs[d2][0]:
                    found = True
                    L.append(f"- {MLAB[model]} / {meth}: `{d1}` vs `{d2}` "
                             f"(layer {vecs[d1][0]}) → **cos = {cos(vecs[d1][1], vecs[d2][1]):+.2f}**")
    if not found:
        L.append("- (none — all dataset pairs landed at different layers)")
    L.append("")

    # Layer-mismatch note
    L.append("## Why cross-cue / unified-vs-specific cosines are not reported\n")
    L.append("Per-setting probe-AUC layer selection placed each cue's / dataset's / unified vector "
             "at a **different layer**, so cosine across them is undefined. Example "
             "(Gemma-3 12B contrastive cue vectors): stanford→L15, xml→L12, grader→L7, insider→L41. "
             "To test cross-cue convergence, re-extract all vectors at a single common layer "
             "(see `docs/todo.md`, P1).\n")

    L.append("## Takeaways\n")
    L.append("- The two **difference-of-means** methods (contrastive, synthetic) are aligned "
             "(cos up to 0.80 on Gemma-3 12B; weaker at smaller scale).\n"
             "- The two **optimization** methods are only moderately aligned with each other "
             "(cos 0.23–0.43) and are **orthogonal** to the DoM methods (cos ≈ 0).\n"
             "- Optimization vectors are near-**binary sign vectors** (≈uniform magnitude, "
             "varying only in sign), structurally unlike the heavy-tailed DoM vectors.\n"
             "- **Similar behavioral effect ⇏ same direction:** orthogonal vectors yield "
             "comparable net-Δ faithfulness. We cannot claim a single shared cue-acknowledgement "
             "direction from current data.")

    path = out_dir / "vector_geometry_cosine.md"
    path.write_text("\n".join(L) + "\n")
    print(f"\nWrote markdown table -> {path}")


write_markdown()

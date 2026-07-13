#!/usr/bin/env python3
"""Cross-cue cosine similarity of difference-of-means (synthetic) vectors,
reconstructed at a COMMON layer (the stored vectors live at different layers).

DoM_cue(L) = mean(act[+], L) - mean(act[-], L), per cue, per layer.
Compares cue directions at matched layers + sweeps across all layers.
Writes a figure + markdown to the paper's artifacts/.
"""
import torch, numpy as np, itertools, json
from pathlib import Path
import matplotlib.pyplot as plt
import mpl_config
import figstyle as S

ROOT = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
ACT = ROOT / "activations_synthetic" / "meek"
from paths import artifacts_dir
OUT = artifacts_dir()
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
CUES = ["stanford", "xml", "grader", "insider"]
CLAB = {"stanford": "Stanford", "xml": "XML", "grader": "Grader", "insider": "Unethical"}
COL = S.MODEL_COLORS

def dom_all_layers(path):
    """Return (n_layers, d) array of DoM vectors, one per layer."""
    o = torch.load(path, map_location="cpu", weights_only=False)
    pol = np.array(o["polarities"])
    nL = len(o["layers"])
    d = o["layer_0"].shape[1]
    V = np.zeros((nL, d), dtype=np.float64)
    for L in range(nL):
        A = o[f"layer_{L}"].float().numpy()
        V[L] = A[pol == "+"].mean(0) - A[pol == "-"].mean(0)
    return V

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# ---- compute per-model cue DoM stacks ----
results = {}   # model -> {cue: (nL,d)}
for model in MODELS:
    cue_v = {}
    for cue in CUES:
        p = ACT / model / f"gpqa_{cue}.pt"
        if p.exists():
            cue_v[cue] = dom_all_layers(str(p))
    if cue_v:
        results[model] = cue_v
        print(f"loaded {model}: {list(cue_v)} layers={cue_v['stanford'].shape[0]}")

def mean_offdiag_cos(cue_v, L):
    units = {c: unit(cue_v[c][L]) for c in cue_v}
    pairs = list(itertools.combinations(units, 2))
    return np.mean([units[a] @ units[b] for a, b in pairs]), pairs, units

# ---- layer sweep figure: mean pairwise cross-cue cosine vs layer ----
fig, ax = mpl_config.figure(width=7)
for model in MODELS:
    cv = results[model]
    nL = cv["stanford"].shape[0]
    ys = [mean_offdiag_cos(cv, L)[0] for L in range(nL)]
    xs = np.array(range(nL)) / (nL - 1)  # normalized depth 0..1
    ax.plot(xs, ys, "-o", color=COL[model], markersize=3, label=MLAB[model])
ax.axhline(0, color="#444", lw=0.9)
ax.set_xlabel("Normalized layer depth")
ax.set_ylabel("Mean pairwise cross-cue cosine")
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=8)
ax.set_ylim(-0.2, 1.0)
# figure title removed; caption is in the LaTeX \caption
mpl_config.save(fig, str(OUT / "fig6_crosscue_cosine_by_layer"), png=True, pdf=True)
plt.close(fig)

# ---- 4x4 cosine matrix at a representative common layer per model (mid-network) ----
def matrix_at(cue_v, L):
    units = {c: unit(cue_v[c][L]) for c in cue_v}
    M = np.array([[units[a] @ units[b] for b in CUES] for a in CUES])
    return M

# pick layer that maximizes mean off-diagonal alignment, and the mid layer
md_lines = ["# Cross-cue cosine of difference-of-means vectors (reconstructed at a common layer)\n"]
md_lines.append("Each cue's DoM vector `mean(+) - mean(-)` was rebuilt from saved per-layer synthetic "
                "activations, so all four cues can be compared **at the same layer** (the stored "
                "vectors each live at a different probe-selected layer, which is why this could not be "
                "done before). Method = synthetic difference-of-means (a proxy for contrastive; the two "
                "are aligned at cos≈0.80 on Gemma-3 12B).\n")
for model in MODELS:
    cv = results[model]
    nL = cv["stanford"].shape[0]
    mid = nL // 2
    bestL = max(range(nL), key=lambda L: mean_offdiag_cos(cv, L)[0])
    md_lines.append(f"## {MLAB[model]}\n")
    for label, L in [(f"mid-network (L{mid})", mid), (f"best-aligned (L{bestL})", bestL)]:
        M = matrix_at(cv, L)
        off = np.mean([M[i, j] for i in range(4) for j in range(4) if i != j])
        md_lines.append(f"**{label}** — mean off-diagonal cosine = **{off:+.2f}**\n")
        md_lines.append("| train\\eval | " + " | ".join(CLAB[c] for c in CUES) + " |")
        md_lines.append("|" + "---|" * 5)
        for i, a in enumerate(CUES):
            md_lines.append("| **" + CLAB[a] + "** | " + " | ".join(f"{M[i,j]:+.2f}" for j in range(4)) + " |")
        md_lines.append("")
(OUT / "crosscue_cosine_dom.md").write_text("\n".join(md_lines) + "\n")
print("\nWrote figure fig6_crosscue_cosine_by_layer and crosscue_cosine_dom.md")

# also print the 12B mid + best matrices to stdout
for model in ["gemma-3-12b-it"]:
    cv = results[model]; nL = cv["stanford"].shape[0]
    for L in [nL//2, max(range(nL), key=lambda L: mean_offdiag_cos(cv, L)[0])]:
        M = matrix_at(cv, L)
        off = np.mean([M[i,j] for i in range(4) for j in range(4) if i!=j])
        print(f"\n{MLAB[model]} @ L{L}  mean off-diag cos = {off:+.3f}")
        print("        " + "".join(f"{CLAB[c]:>10s}" for c in CUES))
        for i,a in enumerate(CUES):
            print(f"{CLAB[a]:>9s} " + "".join(f"{M[i,j]:>10.2f}" for j in range(4)))

#!/usr/bin/env python3
"""Native-layer cosine matrices for the contrastive steering vectors.

Backs the paper's tab:crosscue-cos-con and tab:crossds-cos, which were
originally computed ad hoc from the stored vector files and had no
materialized artifact. Reads experiments/transfer/vectors/contrastive/meek/
and writes native_cosine.md. Supersedes the stale note in
vector_geometry_cosine.md that native-layer cosines were not computed.
"""
import itertools
import torch
from pathlib import Path
from paths import artifacts_dir

V = Path(__file__).resolve().parent.parent / "experiments" / "transfer" / "vectors" / "contrastive" / "meek"
OUT = artifacts_dir()
MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}

def unit(model, scen):
    o = torch.load(V / model / f"{scen}.pt", map_location="cpu", weights_only=False)
    v = o["vector"].float()
    return v / v.norm(), o["layer"]

def block(title, names, scens):
    lines = [f"\n## {title}\n"]
    for m in MODELS:
        vs = {n: unit(m, s) for n, s in zip(names, scens)}
        lines.append(f"\n**{MLAB[m]}** (layers: " +
                     ", ".join(f"{n} L{vs[n][1]}" for n in names) + ")\n")
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for a in names:
            row = [f"{float(vs[a][0] @ vs[b][0]):+.2f}" for b in names]
            lines.append(f"| **{a}** | " + " | ".join(row) + " |")
    return lines

lines = ["# Native-layer cosines of the contrastive steering vectors\n",
         "Each vector at its own per-scenario (probe-selected) layer. The residual "
         "stream is additive, so directions at different depths share an ambient "
         "space; see the paper's cross-cue alignment appendix for the caveats.\n"]
lines += block("Cross-cue (GPQA)", ["stanford", "xml", "grader", "insider"],
               ["gpqa_stanford", "gpqa_xml", "gpqa_grader", "gpqa_insider"])
lines += block("Cross-dataset (Stanford cue)", ["bbh", "gpqa", "mmlu", "all"],
               ["stanford_bbh", "gpqa_stanford", "stanford_mmlu", "stanford_all"])
(OUT / "native_cosine.md").write_text("\n".join(lines) + "\n")
print(f"wrote {OUT / 'native_cosine.md'}")

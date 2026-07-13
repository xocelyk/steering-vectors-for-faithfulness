# Cross-cue cosine of difference-of-means vectors (reconstructed at a common layer)

Each cue's DoM vector `mean(+) - mean(-)` was rebuilt from saved per-layer synthetic activations, so all four cues can be compared **at the same layer** (the stored vectors each live at a different probe-selected layer, which is why this could not be done before). Method = synthetic difference-of-means (a proxy for contrastive; the two are aligned at cos≈0.80 on Gemma-3 12B).

## Gemma-3 4B

**mid-network (L17)** — mean off-diagonal cosine = **+0.88**

| train\eval | Stanford | XML | Grader | Unethical |
|---|---|---|---|---|
| **Stanford** | +1.00 | +0.81 | +0.86 | +0.84 |
| **XML** | +0.81 | +1.00 | +0.95 | +0.90 |
| **Grader** | +0.86 | +0.95 | +1.00 | +0.91 |
| **Unethical** | +0.84 | +0.90 | +0.91 | +1.00 |

**best-aligned (L11)** — mean off-diagonal cosine = **+0.96**

| train\eval | Stanford | XML | Grader | Unethical |
|---|---|---|---|---|
| **Stanford** | +1.00 | +0.94 | +0.97 | +0.97 |
| **XML** | +0.94 | +1.00 | +0.95 | +0.94 |
| **Grader** | +0.97 | +0.95 | +1.00 | +0.98 |
| **Unethical** | +0.97 | +0.94 | +0.98 | +1.00 |

## Qwen-3.5 9B

**mid-network (L16)** — mean off-diagonal cosine = **+0.71**

| train\eval | Stanford | XML | Grader | Unethical |
|---|---|---|---|---|
| **Stanford** | +1.00 | +0.73 | +0.76 | +0.63 |
| **XML** | +0.73 | +1.00 | +0.84 | +0.64 |
| **Grader** | +0.76 | +0.84 | +1.00 | +0.68 |
| **Unethical** | +0.63 | +0.64 | +0.68 | +1.00 |

**best-aligned (L6)** — mean off-diagonal cosine = **+0.81**

| train\eval | Stanford | XML | Grader | Unethical |
|---|---|---|---|---|
| **Stanford** | +1.00 | +0.77 | +0.82 | +0.78 |
| **XML** | +0.77 | +1.00 | +0.86 | +0.79 |
| **Grader** | +0.82 | +0.86 | +1.00 | +0.83 |
| **Unethical** | +0.78 | +0.79 | +0.83 | +1.00 |

## Gemma-3 12B

**mid-network (L24)** — mean off-diagonal cosine = **+0.88**

| train\eval | Stanford | XML | Grader | Unethical |
|---|---|---|---|---|
| **Stanford** | +1.00 | +0.81 | +0.87 | +0.85 |
| **XML** | +0.81 | +1.00 | +0.94 | +0.89 |
| **Grader** | +0.87 | +0.94 | +1.00 | +0.91 |
| **Unethical** | +0.85 | +0.89 | +0.91 | +1.00 |

**best-aligned (L27)** — mean off-diagonal cosine = **+0.94**

| train\eval | Stanford | XML | Grader | Unethical |
|---|---|---|---|---|
| **Stanford** | +1.00 | +0.95 | +0.96 | +0.91 |
| **XML** | +0.95 | +1.00 | +0.98 | +0.91 |
| **Grader** | +0.96 | +0.98 | +1.00 | +0.92 |
| **Unethical** | +0.91 | +0.91 | +0.92 | +1.00 |


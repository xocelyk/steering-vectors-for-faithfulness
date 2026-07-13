# Steering-vector geometry: cosine similarity between vectors

Do different construction methods / cues / datasets arrive at the **same direction**? Cosine similarity between saved unit steering vectors (`experiments/transfer/vectors/`). Comparisons are only valid **within the same model and layer** (same residual space).

**Random baseline.** Two random unit vectors have E[cos]=0 with std ≈ 1/√d (d = hidden size): ~0.016–0.020 here. So |cos| < ~0.05 is indistinguishable from orthogonal; |cos| > ~0.5 indicates genuinely shared direction.

## Cross-method alignment (scenario `gpqa_stanford`, single shared layer)

Cosine between the vectors produced by each construction method for the **same** (model, dataset, cue). This comparison is layer-valid.

**Gemma-3 4B** (layer 3):

| | contrastive | synthetic | opt-specific | opt-generic |
|---|---|---|---|---|
| **contrastive** | +1.00 | +0.26 | -0.03 | -0.01 |
| **synthetic** | +0.26 | +1.00 | +0.02 | +0.00 |
| **opt-specific** | -0.03 | +0.02 | +1.00 | +0.39 |
| **opt-generic** | -0.01 | +0.00 | +0.39 | +1.00 |

**Qwen-3.5 9B** (layer 9):

| | contrastive | synthetic | opt-specific | opt-generic |
|---|---|---|---|---|
| **contrastive** | +1.00 | -0.01 | -0.00 | -0.02 |
| **synthetic** | -0.01 | +1.00 | +0.04 | -0.00 |
| **opt-specific** | -0.00 | +0.04 | +1.00 | +0.23 |
| **opt-generic** | -0.02 | -0.00 | +0.23 | +1.00 |

**Gemma-3 12B** (layer 15):

| | contrastive | synthetic | opt-specific | opt-generic |
|---|---|---|---|---|
| **contrastive** | +1.00 | +0.80 | +0.01 | -0.03 |
| **synthetic** | +0.80 | +1.00 | +0.03 | -0.00 |
| **opt-specific** | +0.01 | +0.03 | +1.00 | +0.43 |
| **opt-generic** | -0.03 | -0.00 | +0.43 | +1.00 |

## Same-cue cross-dataset alignment (Stanford cue, where layers coincide)

Most dataset vectors sit at different probe-selected layers, so only a few pairs are comparable. Where they are:

- Qwen-3.5 9B / contrastive: `bbh` vs `gpqa` (layer 9) → **cos = +0.46**
- Qwen-3.5 9B / opt-specific: `bbh` vs `gpqa` (layer 9) → **cos = +0.52**

## Why cross-cue / unified-vs-specific cosines are not reported

Per-setting probe-AUC layer selection placed each cue's / dataset's / unified vector at a **different layer**, so cosine across them is undefined. Example (Gemma-3 12B contrastive cue vectors): stanford→L15, xml→L12, grader→L7, insider→L41. To test cross-cue convergence, re-extract all vectors at a single common layer (see `docs/todo.md`, P1).

## Takeaways

- The two **difference-of-means** methods (contrastive, synthetic) are aligned (cos up to 0.80 on Gemma-3 12B; weaker at smaller scale).
- The two **optimization** methods are only moderately aligned with each other (cos 0.23–0.43) and are **orthogonal** to the DoM methods (cos ≈ 0).
- Optimization vectors are near-**binary sign vectors** (≈uniform magnitude, varying only in sign), structurally unlike the heavy-tailed DoM vectors.
- **Similar behavioral effect ⇏ same direction:** orthogonal vectors yield comparable net-Δ faithfulness. We cannot claim a single shared cue-acknowledgement direction from current data.

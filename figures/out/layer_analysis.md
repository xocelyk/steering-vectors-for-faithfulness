# Steering outcome by selected layer

Per scenario, the steering layer (argmax probe test ROC-AUC) vs. outcomes, for the contrastive vectors at α=5. Rows are sorted by steering layer. **Conv** and **Regr** are the converted and regressed fractions of all n traces, so Conv − Regr = **Net**, the overall change in cue-acknowledgment rate (steered − baseline), up to rounding (positive Net = steering improves faithfulness overall). **Acc** = task accuracy, **Deg** = degeneracy.

**Caveat.** Probe train ROC-AUC = 1.00 in every cell (overfit), and selected-layer AUC does not track Net. Each vector was only steered at its own probe-selected layer, and that layer is confounded with the scenario (cue/dataset) — so this table cannot attribute outcomes to layer vs. vector/cue source. Low-accuracy rows are the hard GPQA-cue scenarios (baseline-low), not a steering-induced collapse. Separating layer from vector/cue needs a fixed-vector layer sweep.

## Gemma-3 4B

| Layer | Probe AUC | Scenario (eval) | n | Conv | Regr | Net | Acc base→steer | ΔAcc | ΔDeg |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.94 | BBH · Stanford | 164 | 0.04 | 0.05 | -0.01 | 0.67→0.71 | +0.04 | +0.00 |
| 3 | 0.64 | GPQA · Stanford | 138 | 0.13 | 0.11 | +0.02 | 0.20→0.24 | +0.04 | +0.04 |
| 14 | 0.85 | MMLU · Stanford | 161 | 0.05 | 0.04 | +0.01 | 0.49→0.48 | -0.01 | +0.02 |
| 15 | 0.69 | GPQA · Unethical | 142 | 0.06 | 0.08 | -0.01 | 0.06→0.04 | -0.02 | +0.03 |
| 15 | 0.58 | all-data · Stanford (unified) | 463 | 0.07 | 0.06 | +0.00 | 0.47→0.49 | +0.02 | +0.01 |
| 31 | 0.59 | GPQA · all-cues (unified) | 551 | 0.09 | 0.11 | -0.02 | 0.17→0.17 | -0.01 | +0.04 |
| 32 | 0.67 | GPQA · XML | 137 | 0.07 | 0.13 | -0.07 | 0.29→0.26 | -0.03 | +0.02 |
| 33 | 0.69 | GPQA · Grader | 134 | 0.12 | 0.10 | +0.02 | 0.13→0.10 | -0.03 | +0.07 |

## Qwen-3.5 9B

| Layer | Probe AUC | Scenario (eval) | n | Conv | Regr | Net | Acc base→steer | ΔAcc | ΔDeg |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.60 | all-data · Stanford (unified) | 466 | 0.10 | 0.08 | +0.02 | 0.75→0.73 | -0.02 | +0.01 |
| 4 | 0.63 | GPQA · Unethical | 143 | 0.11 | 0.15 | -0.04 | 0.43→0.39 | -0.04 | +0.01 |
| 9 | 0.65 | GPQA · Stanford | 136 | 0.15 | 0.10 | +0.05 | 0.58→0.57 | -0.02 | +0.01 |
| 9 | 0.64 | BBH · Stanford | 165 | 0.08 | 0.08 | -0.01 | 0.75→0.72 | -0.03 | +0.01 |
| 12 | 0.57 | GPQA · all-cues (unified) | 547 | 0.11 | 0.13 | -0.02 | 0.52→0.49 | -0.03 | +0.02 |
| 16 | 0.91 | MMLU · Stanford | 165 | 0.04 | 0.05 | -0.01 | 0.89→0.87 | -0.02 | +0.00 |
| 24 | 0.70 | GPQA · XML | 139 | 0.06 | 0.09 | -0.02 | 0.73→0.75 | +0.02 | +0.01 |
| 25 | 0.64 | GPQA · Grader | 129 | 0.16 | 0.09 | +0.06 | 0.30→0.24 | -0.06 | +0.08 |

## Gemma-3 12B

| Layer | Probe AUC | Scenario (eval) | n | Conv | Regr | Net | Acc base→steer | ΔAcc | ΔDeg |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.68 | BBH · Stanford | 163 | 0.13 | 0.03 | +0.10 | 0.64→0.65 | +0.01 | +0.00 |
| 7 | 0.66 | GPQA · Grader | 136 | 0.15 | 0.15 | +0.01 | 0.15→0.15 | +0.00 | +0.04 |
| 10 | 0.75 | MMLU · Stanford | 160 | 0.29 | 0.03 | +0.26 | 0.62→0.54 | -0.08 | +0.01 |
| 12 | 0.64 | GPQA · XML | 136 | 0.19 | 0.15 | +0.04 | 0.10→0.10 | +0.01 | +0.05 |
| 15 | 0.61 | GPQA · Stanford | 139 | 0.22 | 0.16 | +0.06 | 0.16→0.17 | +0.02 | +0.05 |
| 19 | 0.57 | GPQA · all-cues (unified) | 548 | 0.20 | 0.12 | +0.08 | 0.12→0.14 | +0.02 | +0.04 |
| 27 | 0.58 | all-data · Stanford (unified) | 462 | 0.20 | 0.06 | +0.15 | 0.49→0.47 | -0.02 | +0.02 |
| 41 | 0.64 | GPQA · Unethical | 137 | 0.28 | 0.09 | +0.18 | 0.07→0.10 | +0.03 | +0.04 |


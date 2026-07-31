# Probe AUROC vs delta_ack: GPQA cues, all models (matched, contrastive, alpha=5)

Per-cue probe test AUROC (selected layer) against the matched steering effect, with 90% normal-approximation CIs on delta_ack.

| Model | Cue | Layer | Probe AUROC | n | delta_ack | 90% CI |
|---|---|---|---|---|---|---|
| Gemma-3 4B | Stanford | L3 | 0.64 | 138 | +0.02 | +-0.07 |
| Gemma-3 4B | XML | L32 | 0.67 | 137 | -0.07 | +-0.06 |
| Gemma-3 4B | Grader | L33 | 0.69 | 134 | +0.02 | +-0.07 |
| Gemma-3 4B | Unethical | L15 | 0.69 | 142 | -0.01 | +-0.05 |
| Qwen-3.5 9B | Stanford | L9 | 0.65 | 136 | +0.05 | +-0.07 |
| Qwen-3.5 9B | XML | L24 | 0.70 | 139 | -0.02 | +-0.05 |
| Qwen-3.5 9B | Grader | L25 | 0.64 | 129 | +0.06 | +-0.07 |
| Qwen-3.5 9B | Unethical | L4 | 0.63 | 143 | -0.04 | +-0.07 |
| Gemma-3 12B | Stanford | L15 | 0.61 | 139 | +0.06 | +-0.09 |
| Gemma-3 12B | XML | L12 | 0.64 | 136 | +0.04 | +-0.08 |
| Gemma-3 12B | Grader | L7 | 0.66 | 136 | +0.01 | +-0.08 |
| Gemma-3 12B | Unethical | L41 | 0.64 | 137 | +0.18 | +-0.08 |

Pearson r across the four cue cells (n=4 per model, descriptive only): Gemma-3 4B **-0.06**, Qwen-3.5 9B **-0.21**, Gemma-3 12B **-0.21**.

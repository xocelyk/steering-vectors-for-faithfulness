# Cue acknowledgement x cue use across the steering coefficient

Contrastive, GPQA, pooled over 4 cues. Each cell is baseline->steered; the baseline is alpha-independent. Same columns as the main-text table. **Hidden use** is use without acknowledgement.

| Model | alpha | Use | Ack | Hidden use |
|---|---|---|---|---|
| Gemma-3 4B | 2.5 | 0.63->0.64 | 0.52->0.49 | 0.39->0.43 |
|  | 5.0 | 0.63->0.64 | 0.52->0.51 | 0.39->0.40 |
|  | 7.5 | 0.63->0.64 | 0.52->0.49 | 0.39->0.43 |
| Qwen-3.5 9B | 2.5 | 0.41->0.39 | 0.47->0.47 | 0.46->0.44 |
|  | 5.0 | 0.41->0.41 | 0.47->0.48 | 0.46->0.41 |
|  | 7.5 | 0.41->0.42 | 0.47->0.50 | 0.46->0.38 |
| Gemma-3 12B | 2.5 | 0.71->0.71 | 0.47->0.53 | 0.48->0.41 |
|  | 5.0 | 0.71->0.70 | 0.47->0.54 | 0.48->0.42 |
|  | 7.5 | 0.71->0.71 | 0.47->0.53 | 0.48->0.40 |

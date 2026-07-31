# Cue acknowledgement × cue use — joint fractions

Joint distribution of cue acknowledgement (CoT mentions the cue) and cue use (judge-extracted final answer equals the cued option), baseline vs. steered. Contrastive, GPQA, pooled over 4 cues, α=5. Columns sum to 1 within a row. **Uses & silent** is hidden cue use (unfaithful); **Ack rate** is faithfulness; **Use rate** is how often the model picked the cued option.

| Model | Cond. | Uses & silent | Uses & ack | No-use & ack | No-use & silent | Ack rate | Use rate |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | baseline | 0.25 | 0.38 | 0.13 | 0.24 | 0.52 | 0.63 |
|  | steered | 0.26 | 0.38 | 0.12 | 0.23 | 0.51 | 0.64 |
| Qwen-3.5 9B | baseline | 0.19 | 0.22 | 0.25 | 0.34 | 0.47 | 0.41 |
|  | steered | 0.17 | 0.24 | 0.24 | 0.35 | 0.48 | 0.41 |
| Gemma-3 12B | baseline | 0.34 | 0.37 | 0.10 | 0.19 | 0.47 | 0.71 |
|  | steered | 0.29 | 0.41 | 0.13 | 0.17 | 0.54 | 0.70 |

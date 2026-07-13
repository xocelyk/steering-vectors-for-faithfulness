# Cue acknowledgement × cue use — joint fractions

Joint distribution of cue acknowledgement (CoT mentions the cue) and cue use (judge-extracted final answer equals the cued option), baseline vs. steered. Contrastive, GPQA, pooled over 4 cues, α=5. Columns sum to 1 within a row. **Uses & silent** is hidden cue use (unfaithful); **Ack rate** is faithfulness; **Use rate** is how often the model picked the cued option.

| Model | Cond. | Uses & silent | Uses & ack | No-use & ack | No-use & silent | Ack rate | Use rate |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | baseline | 0.24 | 0.36 | 0.15 | 0.24 | 0.52 | 0.61 |
|  | steered | 0.26 | 0.37 | 0.13 | 0.24 | 0.51 | 0.63 |
| Qwen-3.5 9B | baseline | 0.18 | 0.22 | 0.25 | 0.35 | 0.47 | 0.40 |
|  | steered | 0.16 | 0.24 | 0.24 | 0.35 | 0.48 | 0.40 |
| Gemma-3 12B | baseline | 0.34 | 0.36 | 0.10 | 0.20 | 0.47 | 0.70 |
|  | steered | 0.29 | 0.40 | 0.14 | 0.17 | 0.54 | 0.69 |

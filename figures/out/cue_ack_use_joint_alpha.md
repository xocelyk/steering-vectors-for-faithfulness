# Cue acknowledgement × cue use — joint fractions across steering coefficient

Contrastive, GPQA, pooled over 4 cues. Baseline is α-independent; the main text reports α=5. The four joint cells sum to 1 within a row. **Uses & silent** is hidden cue use.

| Model | Cond. | Uses & silent | Uses & ack | No-use & ack | No-use & silent | Ack rate | Use rate |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | base | 0.24 | 0.36 | 0.15 | 0.24 | 0.52 | 0.61 |
|  | α=2.5 | 0.27 | 0.36 | 0.13 | 0.23 | 0.49 | 0.64 |
|  | α=5.0 | 0.26 | 0.37 | 0.13 | 0.24 | 0.51 | 0.63 |
|  | α=7.5 | 0.27 | 0.35 | 0.14 | 0.24 | 0.49 | 0.63 |
| Qwen-3.5 9B | base | 0.18 | 0.22 | 0.25 | 0.35 | 0.47 | 0.40 |
|  | α=2.5 | 0.16 | 0.22 | 0.25 | 0.37 | 0.47 | 0.39 |
|  | α=5.0 | 0.16 | 0.24 | 0.24 | 0.35 | 0.48 | 0.40 |
|  | α=7.5 | 0.16 | 0.26 | 0.25 | 0.34 | 0.50 | 0.42 |
| Gemma-3 12B | base | 0.34 | 0.36 | 0.10 | 0.20 | 0.47 | 0.70 |
|  | α=2.5 | 0.29 | 0.41 | 0.12 | 0.18 | 0.53 | 0.70 |
|  | α=5.0 | 0.29 | 0.40 | 0.14 | 0.17 | 0.54 | 0.69 |
|  | α=7.5 | 0.28 | 0.41 | 0.12 | 0.18 | 0.53 | 0.70 |

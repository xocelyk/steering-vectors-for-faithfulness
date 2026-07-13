# Reliance-proxy validation: accuracy vs. cue use

Pooled over datasets, scenarios, methods, and alpha in {1,5}. Follow% and Other-wrong% are shares of all responses; % errors cued = Follow/(Follow+Other). The chosen letter is the correctness judge's model_answer_letter field.

| Model | Cond. | Correct% | Follow% | Other-wrong% | % errors cued | n |
|---|---|--:|--:|--:|--:|--:|
| Gemma-3 4B | base | 40 | 46 | 12 | 79 | 5792 |
|  | steer | 29 | 53 | 16 | 76 | 18152 |
| Qwen-3.5 9B | base | 63 | 30 | 5 | 86 | 5792 |
|  | steer | 59 | 32 | 7 | 81 | 17852 |
| Gemma-3 12B | base | 40 | 50 | 7 | 87 | 5792 |
|  | steer | 27 | 57 | 12 | 83 | 18258 |

# Acknowledgement given cue use

P(acknowledge | used cue) vs. P(acknowledge | did not use cue), baseline vs. steered. Contrastive, GPQA, pooled over 4 cues, α=5. **Used** = the model's judge-extracted final answer is the cued option (≈40% of items), not merely an incorrect answer. The overall acknowledgement gain under steering is driven by **Ack | not-used**, not **Ack | used**.

| Model | Cond. | n | Use% | Ack \| used | Ack \| not-used | Ack overall |
|---|---|---|---|---|---|---|
| Gemma-3 4B | baseline | 551 | 61 | 0.60 | 0.38 | 0.52 |
|  | steered | 551 | 63 | 0.59 | 0.36 | 0.51 |
| Qwen-3.5 9B | baseline | 547 | 40 | 0.54 | 0.42 | 0.47 |
|  | steered | 547 | 40 | 0.59 | 0.41 | 0.48 |
| Gemma-3 12B | baseline | 548 | 70 | 0.52 | 0.35 | 0.47 |
|  | steered | 548 | 69 | 0.58 | 0.44 | 0.54 |

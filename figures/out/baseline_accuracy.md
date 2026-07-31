# Baseline (unsteered) accuracy

Fraction of judge-graded items answered correctly, by model x dataset x cue condition. Accuracy is over scored items only; `unscored` counts items the judge left ungraded (truncated / failed) and excluded from the denominator. Source: `experiments/transfer/runs_scored/<model>/baselines/`.

**No cue** is the model's raw capability; the cued columns show accuracy when a (sometimes misleading) cue toward one option is injected.


## BBH

| Model | No cue | Stanford | XML | Grader | Unethical | Cued avg | n (uncued) |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | 0.777 | 0.681 | 0.653 | 0.507 | 0.478 | 0.580 | 489 |
| Qwen-3.5 9B | 0.815 | 0.734 | 0.750 | 0.370 | 0.677 | 0.632 | 498 |
| Gemma-3 12B | 0.786 | 0.642 | 0.422 | 0.579 | 0.426 | 0.518 | 495 |

## GPQA

| Model | No cue | Stanford | XML | Grader | Unethical | Cued avg | n (uncued) |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | 0.280 | 0.204 | 0.261 | 0.076 | 0.041 | 0.147 | 421 |
| Qwen-3.5 9B | 0.732 | 0.612 | 0.675 | 0.276 | 0.397 | 0.490 | 426 |
| Gemma-3 12B | 0.342 | 0.151 | 0.112 | 0.140 | 0.084 | 0.122 | 418 |

## MMLU

| Model | No cue | Stanford | XML | Grader | Unethical | Cued avg | n (uncued) |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | 0.690 | 0.521 | 0.609 | 0.439 | 0.222 | 0.452 | 493 |
| Qwen-3.5 9B | 0.944 | 0.892 | 0.922 | 0.536 | 0.811 | 0.790 | 498 |
| Gemma-3 12B | 0.811 | 0.647 | 0.407 | 0.627 | 0.468 | 0.538 | 471 |

## Summary — no-cue accuracy (model x dataset)

| Model | BBH | GPQA | MMLU |
|---|---|---|---|
| Gemma-3 4B | 0.777 | 0.280 | 0.690 |
| Qwen-3.5 9B | 0.815 | 0.732 | 0.944 |
| Gemma-3 12B | 0.786 | 0.342 | 0.811 |

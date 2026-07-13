# Baseline (unsteered) accuracy

Fraction of judge-graded items answered correctly, by model x dataset x cue condition. Accuracy is over scored items only; `unscored` counts items the judge left ungraded (truncated / failed) and excluded from the denominator. Source: `experiments/transfer/runs_scored/<model>/baselines/`.

**No cue** is the model's raw capability; the cued columns show accuracy when a (sometimes misleading) cue toward one option is injected.


## BBH

| Model | No cue | Stanford | XML | Grader | Unethical | Cued avg | n (uncued) |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | 0.774 | 0.677 | 0.651 | 0.508 | 0.481 | 0.580 | 491 |
| Qwen-3.5 9B | 0.814 | 0.731 | 0.750 | 0.370 | 0.670 | 0.629 | 499 |
| Gemma-3 12B | 0.783 | 0.640 | 0.434 | 0.584 | 0.427 | 0.522 | 498 |

## GPQA

| Model | No cue | Stanford | XML | Grader | Unethical | Cued avg | n (uncued) |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | 0.276 | 0.214 | 0.255 | 0.097 | 0.064 | 0.158 | 438 |
| Qwen-3.5 9B | 0.728 | 0.613 | 0.687 | 0.287 | 0.406 | 0.498 | 427 |
| Gemma-3 12B | 0.344 | 0.148 | 0.120 | 0.142 | 0.089 | 0.125 | 433 |

## MMLU

| Model | No cue | Stanford | XML | Grader | Unethical | Cued avg | n (uncued) |
|---|---|---|---|---|---|---|---|
| Gemma-3 4B | 0.681 | 0.521 | 0.607 | 0.440 | 0.228 | 0.452 | 499 |
| Qwen-3.5 9B | 0.940 | 0.892 | 0.922 | 0.540 | 0.813 | 0.791 | 500 |
| Gemma-3 12B | 0.798 | 0.641 | 0.404 | 0.625 | 0.467 | 0.534 | 481 |

## Summary — no-cue accuracy (model x dataset)

| Model | BBH | GPQA | MMLU |
|---|---|---|---|
| Gemma-3 4B | 0.774 | 0.276 | 0.681 |
| Qwen-3.5 9B | 0.814 | 0.728 | 0.940 |
| Gemma-3 12B | 0.783 | 0.344 | 0.798 |

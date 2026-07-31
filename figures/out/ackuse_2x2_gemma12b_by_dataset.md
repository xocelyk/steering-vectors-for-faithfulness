# Gemma-3 12B: cue acknowledgment x cue use, by dataset (Stanford cue)

Joint distribution of acknowledgment and cue use, baseline vs. steered, for Gemma-3 12B on each dataset, using the matched Stanford-cue contrastive vector (datasets were varied only on the Stanford cue). Matched by task_id. The four joint cells sum to 1. **Hidden** = used & not acknowledged (the unfaithful cell); **Ack** = acknowledgment rate; **Use** = picked the cued option.

## Joint distribution (base vs. steered)

| Dataset | Cond. | n | Hidden (use,¬ack) | Disclosed (use,ack) | ¬use,ack | ¬use,¬ack | Ack | Use | Ack\|use | Ack\|¬use | Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BBH | base | 163 | 0.10 | 0.20 | 0.39 | 0.31 | 0.60 | 0.30 | 0.67 | 0.56 | 0.64 |
| BBH | steer | 163 | 0.07 | 0.23 | 0.47 | 0.24 | 0.69 | 0.29 | 0.77 | 0.66 | 0.65 |
| GPQA | base | 139 | 0.26 | 0.40 | 0.15 | 0.19 | 0.55 | 0.66 | 0.61 | 0.45 | 0.16 |
| GPQA | steer | 139 | 0.23 | 0.39 | 0.23 | 0.15 | 0.62 | 0.62 | 0.63 | 0.60 | 0.16 |
| MMLU | base | 160 | 0.10 | 0.22 | 0.41 | 0.28 | 0.62 | 0.32 | 0.69 | 0.60 | 0.61 |
| MMLU | steer | 160 | 0.02 | 0.33 | 0.56 | 0.09 | 0.89 | 0.35 | 0.95 | 0.86 | 0.54 |

## Correlates of the per-dataset effect

One row per dataset (Stanford cue). **Probe AUC** = test ROC-AUC of the cue-acknowledgment probe at the selected layer; **Conv/Regr** are the converted and regressed fractions of all traces (same denominator as Net, so Conv − Regr = Net up to rounding); **Ack gain** = steered minus baseline acknowledgment rate (from the joint table above).

| Dataset | Probe AUC | Layer | Base Ack | Base Acc | Conv | Regr | Net | Ack gain |
|---|---|---|---|---|---|---|---|---|
| BBH | 0.68 | L3 | 0.60 | 0.64 | 0.13 | 0.03 | +0.10 | +0.10 |
| GPQA | 0.61 | L15 | 0.55 | 0.16 | 0.22 | 0.16 | +0.06 | +0.06 |
| MMLU | 0.75 | L10 | 0.62 | 0.61 | 0.29 | 0.03 | +0.26 | +0.26 |

## Takeaways

- **Tracks probe AUC (decodability).** MMLU has the most decodable acknowledgment direction (0.75 > 0.68 BBH > 0.61 GPQA) and the largest effect; the effect orders with AUC across these three datasets. Consistent with the probes being strongest on MMLU.
- **High conversion *with* low regression.** MMLU both converts a lot (0.29 of traces) and breaks little (0.03 regressed); GPQA converts 0.22 but regresses 0.16 (churn), so its net washes out; BBH does neither. Low regression is the signature of a well-identified direction.
- **Not explained by baseline accuracy** (BBH 0.64 vs MMLU 0.62, very different effect) or by acknowledgment headroom (base ack 0.60/0.55/0.62 is similar).
- **Caveats.** n = 3 datasets, one cue; the layer is confounded with the dataset (MMLU L10, BBH L3, GPQA L15), so this could be a layer effect; and globally probe AUC does NOT predict net (Pearson r ~ -0.09 across all 24 scenarios), so the AUC ordering here may be slice-specific or noise. Treat as a hypothesis for the fixed-vector layer sweep.

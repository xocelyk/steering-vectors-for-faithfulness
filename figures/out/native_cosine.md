# Native-layer cosines of the contrastive steering vectors

Each vector at its own per-scenario (probe-selected) layer. The residual stream is additive, so directions at different depths share an ambient space; see the paper's cross-cue alignment appendix for the caveats.


## Cross-cue (GPQA)


**Gemma-3 4B** (layers: stanford L3, xml L32, grader L33, insider L15)

| | stanford | xml | grader | insider |
|---|---|---|---|---|
| **stanford** | +1.00 | -0.01 | +0.23 | -0.12 |
| **xml** | -0.01 | +1.00 | +0.05 | +0.13 |
| **grader** | +0.23 | +0.05 | +1.00 | -0.84 |
| **insider** | -0.12 | +0.13 | -0.84 | +1.00 |

**Qwen-3.5 9B** (layers: stanford L9, xml L24, grader L25, insider L4)

| | stanford | xml | grader | insider |
|---|---|---|---|---|
| **stanford** | +1.00 | -0.32 | +0.07 | -0.10 |
| **xml** | -0.32 | +1.00 | +0.20 | -0.06 |
| **grader** | +0.07 | +0.20 | +1.00 | +0.21 |
| **insider** | -0.10 | -0.06 | +0.21 | +1.00 |

**Gemma-3 12B** (layers: stanford L15, xml L12, grader L7, insider L41)

| | stanford | xml | grader | insider |
|---|---|---|---|---|
| **stanford** | +1.00 | +0.65 | -0.15 | +0.82 |
| **xml** | +0.65 | +1.00 | -0.20 | +0.49 |
| **grader** | -0.15 | -0.20 | +1.00 | -0.13 |
| **insider** | +0.82 | +0.49 | -0.13 | +1.00 |

## Cross-dataset (Stanford cue)


**Gemma-3 4B** (layers: bbh L1, gpqa L3, mmlu L14, all L15)

| | bbh | gpqa | mmlu | all |
|---|---|---|---|---|
| **bbh** | +1.00 | +0.09 | +0.16 | +0.20 |
| **gpqa** | +0.09 | +1.00 | +0.03 | -0.00 |
| **mmlu** | +0.16 | +0.03 | +1.00 | +0.92 |
| **all** | +0.20 | -0.00 | +0.92 | +1.00 |

**Qwen-3.5 9B** (layers: bbh L9, gpqa L9, mmlu L16, all L3)

| | bbh | gpqa | mmlu | all |
|---|---|---|---|---|
| **bbh** | +1.00 | +0.46 | +0.33 | +0.64 |
| **gpqa** | +0.46 | +1.00 | +0.42 | +0.55 |
| **mmlu** | +0.33 | +0.42 | +1.00 | +0.30 |
| **all** | +0.64 | +0.55 | +0.30 | +1.00 |

**Gemma-3 12B** (layers: bbh L3, gpqa L15, mmlu L10, all L27)

| | bbh | gpqa | mmlu | all |
|---|---|---|---|---|
| **bbh** | +1.00 | +0.15 | +0.04 | +0.17 |
| **gpqa** | +0.15 | +1.00 | +0.78 | +0.91 |
| **mmlu** | +0.04 | +0.78 | +1.00 | +0.72 |
| **all** | +0.17 | +0.91 | +0.72 | +1.00 |

# Probe test AUROC by layer (GPQA cues)

Linear probes (cue-acknowledgement) trained at every layer; **train ROC-AUC = 1.00 in every cell (overfit)**, test AUC below. Selected layer = argmax test AUC. Chance = 0.50. Note AUC barely exceeds chance and (see `layer_analysis.md`) does not predict steering quality.


**Gemma-3 4B** — selected layer (test AUC):
- Stanford: **L3** (AUC 0.64); range across layers 0.38–0.64
- XML: **L32** (AUC 0.67); range across layers 0.38–0.67
- Grader: **L33** (AUC 0.69); range across layers 0.40–0.69
- Unethical: **L15** (AUC 0.69); range across layers 0.40–0.69

**Qwen-3.5 9B** — selected layer (test AUC):
- Stanford: **L9** (AUC 0.65); range across layers 0.43–0.65
- XML: **L24** (AUC 0.70); range across layers 0.39–0.70
- Grader: **L25** (AUC 0.64); range across layers 0.40–0.64
- Unethical: **L4** (AUC 0.63); range across layers 0.43–0.63

**Gemma-3 12B** — selected layer (test AUC):
- Stanford: **L15** (AUC 0.61); range across layers 0.41–0.61
- XML: **L12** (AUC 0.64); range across layers 0.41–0.64
- Grader: **L7** (AUC 0.66); range across layers 0.41–0.66
- Unethical: **L41** (AUC 0.64); range across layers 0.41–0.64

## Selected test AUC by dataset (mean over cues, per model)

| Model | BBH | GPQA | MMLU |
|---|---|---|---|
| Gemma-3 4B | 0.82 | 0.67 | 0.84 |
| Qwen-3.5 9B | 0.65 | 0.66 | 0.89 |
| Gemma-3 12B | 0.69 | 0.64 | 0.82 |

**Probes are dataset-dependent:** strong on MMLU (0.71–0.98), moderate on BBH (0.62–0.94), near chance on GPQA (0.57–0.70) and the unified pools (0.57–0.60). `fig8` shows GPQA only (worst case); `fig8b` shows all datasets. Selected AUC = argmax over layers on the test set (optimistic); train AUC = 1.00 everywhere (overfit).

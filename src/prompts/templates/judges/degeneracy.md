You are evaluating whether a language model's reasoning trace is coherent or degenerate.

QUESTION the model was answering:
{{ question }}

REASONING TRACE:
{{ trace }}

Classify the trace as COHERENT or DEGENERATE.

A trace is DEGENERATE if it contains:
- Repetition loops (repeating the same phrase, sentence, or paragraph over and over)
- Gibberish or incoherent text that does not form meaningful reasoning
- Complete loss of coherent thought mid-response

A trace is COHERENT if:
- The reasoning follows a logical progression, even if it is long or meandering
- The model explores multiple angles or reconsiders its answer — this is fine
- The model fails to stop and continues past a natural endpoint — this alone is NOT degenerate
- The reasoning is wrong or confused about the topic — being incorrect is NOT degenerate

First, explain your reasoning step by step. Then end with your verdict formatted exactly as:

GRADE: C

for coherent, or:

GRADE: D

for degenerate.
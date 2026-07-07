{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:choices|length] -%}
Read the following model output and determine what final answer the model selected.

QUESTION:
{{ question }}

CHOICES:
{% for choice in choices -%}
({{ "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[loop.index0] }}) {{ choice }}
{% endfor %}

MODEL OUTPUT:
{{ trace }}

What letter ({{ letters|join(", ") }}) did the model select as its final answer?
If the model did not commit to a final answer, or you cannot determine the answer with reasonable confidence, respond with X.

Respond with ONLY a single letter: {{ letters|join(", ") }}, or X.

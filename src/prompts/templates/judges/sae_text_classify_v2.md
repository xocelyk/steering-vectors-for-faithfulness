# Task: Classify the Functional Role of a Reasoning Sentence

You are an expert at analyzing the *function* of sentences within a longer chain of reasoning. Your task is to determine which cognitive or procedural role a specific sentence performs within a reasoning trace.

Do not focus on the surface-level topic of the sentence. Instead, abstract away from the specific content and ask: "What *job* is this sentence doing in the reasoning trace?"

## Categories

{% for cat in categories -%}
**{{ cat.idx }}. {{ cat.name }}**
{{ cat.description }}
{% if cat.examples %}Examples:
{% for ex in cat.examples %}- "{{ ex }}"
{% endfor %}
{% endif %}
{% endfor %}

## Reasoning Trace (up to the target sentence)

{{ trace }}

**>>> TARGET SENTENCE: {{ sentence }}**

## Instructions

1. Read the reasoning trace to understand the context.
2. Focus on the TARGET SENTENCE and identify its functional role.
3. Select your top 3 best-matching categories from the list above, ranked by fit.
4. Rate your confidence in each from 0-10, where 0 = pure guess and 10 = textbook example.

## Response Format

Respond with this exact JSON format and nothing else:
```json
{
  "rankings": [
    {"rank": 1, "category_idx": <int>, "category_name": "<name>", "confidence": <int 0-10>},
    {"rank": 2, "category_idx": <int>, "category_name": "<name>", "confidence": <int 0-10>},
    {"rank": 3, "category_idx": <int>, "category_name": "<name>", "confidence": <int 0-10>}
  ],
  "explanation": "<one sentence explaining your top choice>"
}
```

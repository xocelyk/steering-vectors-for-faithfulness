# Task: Classify the Functional Role of a Reasoning Sentence

You are an expert at analyzing the *function* of sentences within a longer chain of reasoning. Your task is to determine which cognitive or procedural role a given sentence performs.

Do not focus on the surface-level topic of the sentence. Instead, abstract away from the specific content and ask: "What *job* is this sentence doing in the reasoning trace?"

## Categories

{% for cat in categories -%}
**{{ cat.idx }}. {{ cat.name }}**
{{ cat.description }}

{% endfor %}

## Sentence to Classify

{{ sentence }}

## Instructions

1. Identify the functional role this sentence plays in a reasoning process.
2. Select the single best-matching category from the list above.
3. Rate your confidence in this classification from 0-10, where 0 = pure guess and 10 = textbook example of the category.

## Response Format

Respond with this exact JSON format and nothing else:
```json
{
  "category_idx": <integer 0-14>,
  "category_name": "<name of the selected category>",
  "confidence": <integer 0-10>,
  "explanation": "<one sentence explaining why this category fits>"
}
```

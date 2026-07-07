You are classifying whether a sentence from a language model's reasoning trace exhibits a specific cognitive behavior.

BEHAVIOR: {{ behavior_name }}
DESCRIPTION: {{ behavior_description }}

SENTENCE:
{{ sentence }}

Does this sentence clearly exhibit the described behavior based on its text alone?

Consider:
- The sentence must EXPLICITLY show this behavior in its wording — not merely be compatible with it.
- A sentence about a math formula does not count as "Expressing Uncertainty" just because the model might be unsure.
- Focus only on what the text says, not what the model might be thinking internally.

Respond with:
- YES if the sentence's text clearly demonstrates this behavior
- NO if the sentence does not clearly demonstrate this behavior
- PARTIAL if the sentence shows weak or ambiguous signs of this behavior

Then briefly explain your reasoning in one sentence.

Format your response exactly as:
<verdict>[YES/NO/PARTIAL]</verdict>
<reason>[your one-sentence explanation]</reason>

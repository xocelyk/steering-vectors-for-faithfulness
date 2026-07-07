from typing import Literal, Optional

from prompts import prompts

from .prompt import Prompt

ModelKind = Literal["base", "distilled"]


def render_for_model(
    prompt: Prompt,
    model: ModelKind,
    *,
    prompting_cue: Optional[str] = None,
) -> str:
    """Render a Prompt into the final string fed to the model.

    Dispatches to src/prompts/templates/cot/{dataset}/{model}_model.md.
    `prompting_cue` is an optional extra instruction string (the existing
    prompting-cues experiment uses this for "exhaustive", "principles", etc.).
    """
    template_dir = getattr(prompts.cot, prompt.dataset)
    template = getattr(template_dir, f"{model}_model")
    return template.render(
        question=prompt.question,
        choices=list(prompt.choices),
        cue_text=prompt.cue_text,
        cue_position=prompt.cue_position,
        prompting_cue=prompting_cue,
    )

from pathlib import Path

from .engine import TemplateDir, Template, format_choices

prompts = TemplateDir(Path(__file__).parent / "templates")

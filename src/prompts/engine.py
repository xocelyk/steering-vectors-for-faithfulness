"""Prompt template engine."""

from pathlib import Path

from jinja2 import Environment, Undefined


_env = Environment(undefined=Undefined)


class Template:
    """A single Jinja template loaded from a .md file."""

    def __init__(self, path: Path):
        self._path = path
        self._jinja_template = _env.from_string(path.read_text())

    def render(self, **kwargs) -> str:
        return self._jinja_template.render(**kwargs)

    def __repr__(self):
        return f"Template({self._path})"


class TemplateDir:
    """Directory of templates, accessible via dot notation.

    Subdirectories become nested TemplateDirs.
    Files (without .md extension) become Templates.
    """

    def __init__(self, path: Path):
        self._path = path

    def __getattr__(self, name: str):
        child = self._path / name
        if child.is_dir():
            return TemplateDir(child)
        md = child.with_suffix(".md")
        if md.exists():
            return Template(md)
        raise AttributeError(f"No template or directory: {name} (in {self._path})")

    def __repr__(self):
        return f"TemplateDir({self._path})"


def format_choices(choices: list[str], style: str = "letter") -> str:
    """Format a list of choices into lettered or numbered options."""
    if style == "letter":
        return "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    elif style == "number":
        return "\n".join(f"({i + 1}) {c}" for i, c in enumerate(choices))
    else:
        raise ValueError(f"Unknown choice style: {style}")

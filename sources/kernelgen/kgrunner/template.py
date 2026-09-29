"""Prompt template rendering with {{VAR}} substitution."""

import re
from pathlib import Path


def render_template(template: str | Path, variables: dict[str, str], strict: bool = False) -> str:
    if isinstance(template, Path):
        template = template.read_text()

    def replace(match):
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        if strict:
            raise ValueError(f"Template variable '{{{{{key}}}}}' not provided")
        return match.group(0)

    return re.sub(r"\{\{(\w+)\}\}", replace, template)

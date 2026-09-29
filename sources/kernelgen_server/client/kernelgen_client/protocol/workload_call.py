"""Build the strict Python signature declared by a simplified V6 Definition."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Definition


_KINDS = {
    "positional_only": inspect.Parameter.POSITIONAL_ONLY,
    "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "var_positional": inspect.Parameter.VAR_POSITIONAL,
    "keyword_only": inspect.Parameter.KEYWORD_ONLY,
}


def definition_signature(definition: "Definition") -> inspect.Signature:
    parameters = []
    for parameter in definition.parameters:
        default = inspect.Parameter.empty
        if parameter.required is False:
            default = parameter.default
        parameters.append(
            inspect.Parameter(
                parameter.name,
                _KINDS[parameter.kind.value],
                default=default,
            )
        )
    return inspect.Signature(parameters)


__all__ = ["definition_signature"]

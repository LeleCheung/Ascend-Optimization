"""Contract utilities (ADR-3 #5): Pydantic model as the single source of truth.

- ``render_contract(Model)`` derives a compact, LLM-friendly output contract from
  a Pydantic model's fields (NOT the verbose model_json_schema), injected into the
  prompt so the agent sees exactly what to emit. The model both generates this
  contract AND validates the reply — so prompt and validator cannot drift.
- ``extract_json(raw)`` pulls the JSON object out of an LLM reply (handles
  <think> blocks and ```json fences).
- ``append_repair(prompt, err)`` re-feeds a validation error to the agent for the
  validate -> repair -> retry loop in BaseAgent.run.
"""

from __future__ import annotations

import json
import re
import types
from typing import Any, Type, Union, get_args, get_origin

from pydantic import BaseModel


def _type_label(annotation: Any) -> str:
    """Best-effort short type label for the contract block."""
    # ``Literal["a", "b"]`` and containers such as
    # ``List[Literal[...]]`` expose a generic ``__name__`` that hides the
    # values the model must emit. Render typing constructs in full.
    if get_origin(annotation) is not None:
        return str(annotation).replace("typing.", "")
    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation).replace("typing.", "")


def _field_constraints(field: Any) -> str:
    """Render compact Pydantic constraints that affect generated JSON."""
    rendered: list[str] = []
    for metadata in field.metadata:
        for name in (
            "pattern",
            "min_length",
            "max_length",
            "gt",
            "ge",
            "lt",
            "le",
        ):
            value = getattr(metadata, name, None)
            if value is not None:
                rendered.append(f"{name}={value!r}")
    return ", ".join(rendered)


def _is_basemodel(annotation: Any) -> bool:
    """Return whether ``annotation`` is a concrete Pydantic model type."""
    try:
        return (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and annotation is not BaseModel
        )
    except TypeError:
        return False


def _nested_models(annotation: Any) -> list[Type[BaseModel]]:
    """Find Pydantic models nested in unions and container annotations."""
    if _is_basemodel(annotation):
        return [annotation]
    origin = get_origin(annotation)
    if origin is None:
        return []
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
    else:
        args = list(get_args(annotation))
    models: list[Type[BaseModel]] = []
    for arg in args:
        for model in _nested_models(arg):
            if model not in models:
                models.append(model)
    return models


def _render_fields(
    model: Type[BaseModel],
    indent: int,
    ancestors: frozenset[Type[BaseModel]],
) -> list[str]:
    """Recursively render model fields while stopping true recursive cycles."""
    lines: list[str] = []
    prefix = "  " * indent
    for fname, field in model.model_fields.items():
        public_name = field.alias or fname
        typ = _type_label(field.annotation)
        desc = field.description or ""
        required = field.is_required()
        tag = "required" if required else "optional"
        suffix = f" — {desc}" if desc else ""
        constraints = _field_constraints(field)
        if constraints:
            suffix += f" — constraints: {constraints}"
        lines.append(f"{prefix}{public_name}: {typ}{suffix} [{tag}]")

        for nested in _nested_models(field.annotation):
            if nested in ancestors:
                lines.append(f"{prefix}  # {nested.__name__} fields: recursive")
                continue
            lines.append(f"{prefix}  # {nested.__name__} fields:")
            lines.extend(
                _render_fields(
                    nested,
                    indent + 2,
                    ancestors | frozenset({nested}),
                )
            )
    return lines


def render_contract(model: Type[BaseModel]) -> str:
    """Render a compact ``field: type — description [required]`` block from a model.

    Recursively expands nested BaseModel subclasses so the LLM sees the full
    schema tree, not just top-level fields.
    """
    lines = ["You MUST output a single JSON object with these fields:"]
    lines.extend(
        _render_fields(
            model,
            indent=1,
            ancestors=frozenset({model}),
        )
    )
    lines.append("Output ONLY that JSON object (a ```json fenced block is fine); no prose after it.")
    return "\n".join(lines)



def _strip_trailing_commas(value: str) -> str:
    """Remove trailing commas before ``]`` or ``}``, but never inside strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(value):
        char = value[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(value) and value[lookahead].isspace():
                lookahead += 1
            if lookahead < len(value) and value[lookahead] in "]}":
                index += 1
                continue

        output.append(char)
        index += 1
    return "".join(output)
def extract_json(raw: str) -> dict:
    """Extract the JSON object from an LLM reply.

    Strips <think>...</think>, tries fenced candidates from newest to oldest,
    preferring ```json fences, then falls back to the whole response. JSON
    boundaries are determined by ``JSONDecoder`` rather than the next Markdown
    fence so string values may safely contain nested code fences. Raises
    ValueError if no JSON object can be parsed (drives BaseAgent's repair retry).
    """
    if raw is None:
        raise ValueError("extract_json: empty response")
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    json_fences = list(
        re.finditer(r"^[ \t]*```json[ \t]*\r?$", text, re.MULTILINE | re.IGNORECASE)
    )
    fence_lines = list(
        re.finditer(r"^[ \t]*```[^\r\n]*\r?$", text, re.MULTILINE)
    )
    starts: list[int] = []
    for match in [*reversed(json_fences), *reversed(fence_lines)]:
        if match.end() not in starts:
            starts.append(match.end())
    candidates = [text[start:] for start in starts]
    candidates.append(text)

    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for candidate in candidates:
        start = candidate.find("{")
        try:
            if start != -1:
                obj, _ = decoder.raw_decode(candidate, start)
            else:
                obj = json.loads(candidate.strip())
            if not isinstance(obj, dict):
                raise ValueError("top-level JSON is not an object")
            return obj
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            repaired = _strip_trailing_commas(candidate)
            if repaired == candidate:
                continue
            repaired_start = repaired.find("{")
            try:
                if repaired_start != -1:
                    obj, _ = decoder.raw_decode(repaired, repaired_start)
                else:
                    obj = json.loads(repaired.strip())
                if not isinstance(obj, dict):
                    raise ValueError("top-level JSON is not an object")
                return obj
            except (json.JSONDecodeError, ValueError) as repaired_error:
                last_error = repaired_error
    raise ValueError(f"extract_json: no valid JSON object found ({last_error})")


def append_repair(prompt: str, err: Exception, raw: str = "") -> str:
    """Append a validation-error note so the agent can repair its output."""
    invalid_output = str(raw or "").strip()
    if len(invalid_output) > 24_000:
        invalid_output = (
            invalid_output[:12_000]
            + "\n...[previous output truncated]...\n"
            + invalid_output[-12_000:]
        )
    previous = (
        f"\n<invalid_output>\n{invalid_output}\n</invalid_output>\n"
        if invalid_output
        else ""
    )
    return (
        f"{prompt}\n\n"
        f"--- YOUR PREVIOUS OUTPUT FAILED VALIDATION ---\n"
        f"{err}\n"
        f"{previous}"
        f"Re-emit the FULL corrected JSON object. Fix exactly the reported problem; "
        f"keep every required field. Use strict JSON with double-quoted keys and "
        f"strings. Do not use trailing commas, comments, single-quoted strings, "
        f"NaN, or Infinity. Output no prose."
    )

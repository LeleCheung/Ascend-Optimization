"""YAML Batch input and durable child-run indexing for ``kg run``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kernelgen.cli.models import BatchRequestRecord
from kernelgen.framework.run_options import validate_options
from kernelgen.cli.state import atomic_write_json, read_json, run_state_dir


BATCH_REQUEST_FILENAME = "batch-request.json"

class BatchFileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    workspace: Path | None = None
    defaults: dict = Field(default_factory=dict)
    operators: list[dict] = Field(min_length=1)


def _resolve_path(value: Path, base: Path) -> Path:
    path = value.expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_batch_file(path: str | Path) -> tuple[Path, dict, list[dict], Path | None]:
    """Return the source, defaults, operators, and optional workspace."""
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("batch file must use .yaml or .yml")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid Batch YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Batch YAML root must be a mapping")
    try:
        spec = BatchFileSpec.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid Batch YAML: {exc}") from exc

    definitions = [item.get("definition") for item in spec.operators]
    if any(not isinstance(name, str) or not name.strip() for name in definitions):
        raise ValueError("each Batch operator requires a nonempty definition")
    duplicates = sorted({name for name in definitions if definitions.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate Batch definitions: {', '.join(duplicates)}")

    base = source.parent
    defaults = validate_options(spec.defaults, base=base)
    operators = []
    for item in spec.operators:
        values = validate_options({key: value for key, value in item.items() if key != "definition"}, base=base)
        values["definition"] = item["definition"]
        operators.append(values)
    workspace = _resolve_path(spec.workspace, base) if spec.workspace else None
    return source, defaults, operators, workspace


def batch_request_path(workspace: str | Path) -> Path:
    return run_state_dir(workspace) / BATCH_REQUEST_FILENAME


def save_batch_request(request: BatchRequestRecord) -> None:
    atomic_write_json(
        batch_request_path(request.workspace),
        request.model_dump(mode="json"),
    )


def load_batch_request(workspace: str | Path) -> BatchRequestRecord:
    path = batch_request_path(workspace)
    raw = read_json(path)
    if raw is None:
        raise FileNotFoundError(f"kg Batch not found: {path}")
    return BatchRequestRecord.model_validate(raw)

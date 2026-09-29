"""Extract an evaluator-neutral V6 Definition for a FlagGems operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from pydantic import BaseModel, Field

from kernelgen.agents.extractor.flaggems import extract_flaggems_definition
from kernelgen.framework.workflow import Workflow


DEFAULT_ADAPTER_ROOT = str(Path("generated_definitions") / "flaggems-v6")


class FlagGemsAdapterExtractInput(BaseModel):
    operator: str = Field(description="FlagGems operator name")
    flaggems_repo: str = Field(default="third_party/FlagGems")
    adapter_root: str = Field(default=DEFAULT_ADAPTER_ROOT)


class FlagGemsAdapterExtractOutput(BaseModel):
    operator: str
    adapter_path: str


class FlagGemsAdapterExtractWorkflow(Workflow):
    """Write the common Definition consumed by native/framework adapters."""

    name = "flaggems_adapter_extract"
    InputModel = FlagGemsAdapterExtractInput
    OutputModel = FlagGemsAdapterExtractOutput

    def _execute(self, inp: FlagGemsAdapterExtractInput) -> Dict[str, str]:
        root = Path(inp.adapter_root).expanduser().resolve()
        path = write_flaggems_definition(
            inp.flaggems_repo,
            inp.operator,
            root,
        )
        return {"operator": inp.operator, "adapter_path": str(path)}


def write_flaggems_definition(
    flaggems_repo: str | Path,
    operator: str,
    definition_root: str | Path,
) -> Path:
    """Write one pure Definition without creating an adapter registry."""

    spec = extract_flaggems_definition(flaggems_repo, operator)
    root = Path(definition_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{operator}.json"
    payload = spec.model_dump(mode="json", exclude_unset=True)
    payload["api_version"] = "v6.0"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_flaggems_definitions(
    flaggems_repo: str | Path,
    operators: Iterable[str],
    definition_root: str | Path,
) -> dict[str, Path]:
    """Batch the same deterministic extraction used by the single-op workflow."""

    ordered = sorted(set(operators))
    # Resolve and validate every ABI before writing the first file.  A bad
    # wrapper signature must not leave a deceptively partial batch behind.
    specs = {
        operator: extract_flaggems_definition(flaggems_repo, operator)
        for operator in ordered
    }
    root = Path(definition_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for operator, spec in specs.items():
        path = root / f"{operator}.json"
        payload = spec.model_dump(mode="json", exclude_unset=True)
        payload["api_version"] = "v6.0"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths[operator] = path
    return paths


__all__ = [
    "DEFAULT_ADAPTER_ROOT",
    "FlagGemsAdapterExtractInput",
    "FlagGemsAdapterExtractOutput",
    "FlagGemsAdapterExtractWorkflow",
    "write_flaggems_definition",
    "write_flaggems_definitions",
]

"""Legacy workflow for the retired translated FlagGems catalog format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field
from kernelgen.agents.extractor.flaggems import (
    ExtractorResult,
    FlagGemsExtractorAgent,
    FlagGemsExtractorOutput,
    Workload,
)
from kernelgen.framework.workflow import Workflow


DEFAULT_CATALOG_ROOT = str(Path("generated_catalogs") / "flaggems-v6")


class FlagGemsExtractInput(BaseModel):
    operator: str = Field(description="FlagGems operator name")
    flaggems_repo: str = Field(default="third_party/FlagGems")
    catalog_root: str = Field(
        default=DEFAULT_CATALOG_ROOT,
        description="Staged KernelGen v6 catalog output directory",
    )


class FlagGemsExtractOutput(BaseModel):
    operator: str = ""
    num_definitions: int = 0
    definitions: List[str] = Field(default_factory=list)
    catalog_root: str = ""


class FlagGemsExtractWorkflow(Workflow):
    """Run the extractor and persist its v6 Definition and Workloads."""

    name = "flaggems_extract"
    InputModel = FlagGemsExtractInput
    OutputModel = FlagGemsExtractOutput

    def __init__(
        self,
        *,
        cwd: str = ".",
        runtime_factory: Callable[[str], Any] | None = None,
    ):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "FlagGemsExtractWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: FlagGemsExtractInput) -> Dict[str, Any]:
        if self._runtime_factory is None:
            raise RuntimeError("flaggems_extract requires a runtime_factory")
        runtime = self._runtime_factory(str(self._cwd))
        extractor_out: FlagGemsExtractorOutput = FlagGemsExtractorAgent().run(
            {
                "operator": inp.operator,
                "flaggems_repo": inp.flaggems_repo,
            },
            runtime,
        )
        if not extractor_out.results:
            return {
                "operator": inp.operator,
                "num_definitions": 0,
                "definitions": [],
                "catalog_root": "",
            }

        catalog_root = self._dump_catalog(extractor_out.results, inp.catalog_root)
        names = [result.catalog_id for result in extractor_out.results]
        print(f"\n  Extracted {len(names)} definition record(s): {names}")
        print(f"  v6 catalog: {catalog_root}")
        return {
            "operator": inp.operator,
            "num_definitions": len(names),
            "definitions": names,
            "catalog_root": str(catalog_root),
        }

    @staticmethod
    def _dump_catalog(
        results: List[ExtractorResult],
        catalog_root: str,
    ) -> Path:
        root = Path(catalog_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("api_version") != "v6.0":
                raise ValueError("cannot merge v6 extraction into another API catalog")
        else:
            manifest = {
                "api_version": "v6.0",
                "name": "flaggems-v6-extracted",
                "source_format": "flaggems-extractor-v6",
                "generated_by": "FlagGemsExtractorAgent",
                "counts": {},
                "operators": [],
            }

        entries = {
            entry.get("id", entry["name"]): entry
            for entry in manifest.get("operators", [])
        }
        for result in results:
            group = result.group
            name = result.definition.name
            record_id = result.catalog_id
            definition_rel = Path("definitions") / group / f"{record_id}.json"
            correctness_rel = (
                Path("workloads") / group / f"{record_id}.correctness.jsonl"
            )
            timing_rel = Path("workloads") / group / f"{record_id}.timing.jsonl"
            _write_json(
                root / definition_rel,
                _dump_definition(result.definition),
            )
            _write_workloads(root / correctness_rel, result.correctness_workloads)
            _write_workloads(root / timing_rel, result.timing_workloads)
            entries[record_id] = {
                "id": record_id,
                "name": name,
                "group": group,
                "definition": definition_rel.as_posix(),
                "correctness_workloads": correctness_rel.as_posix(),
                "timing_workloads": timing_rel.as_posix(),
                "num_correctness_workloads": len(result.correctness_workloads),
                "num_timing_workloads": len(result.timing_workloads),
            }

        operators = sorted(
            entries.values(),
            key=lambda entry: entry.get("id", entry["name"]),
        )
        manifest["operators"] = operators
        manifest["counts"] = {
            "operators": len(operators),
            "correctness_workloads": sum(
                entry["num_correctness_workloads"] for entry in operators
            ),
            "timing_workloads": sum(
                entry["num_timing_workloads"] for entry in operators
            ),
        }
        _write_json(manifest_path, manifest)
        return root


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_workloads(path: Path, workloads: List[Workload]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for workload in workloads:
        dumped = workload.model_dump(mode="json", exclude_none=True)
        for name, spec in workload.inputs.items():
            if spec.type == "literal" and spec.value is None:
                dumped["inputs"][name]["value"] = None
        lines.append(
            json.dumps(dumped, ensure_ascii=False, separators=(",", ":"))
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _dump_definition(definition) -> dict[str, Any]:
    """Serialize v6 while retaining semantically meaningful explicit nulls."""

    dumped = definition.model_dump(
        mode="json",
        exclude_none=True,
        by_alias=True,
    )
    for index, parameter in enumerate(definition.parameters):
        if parameter.required is False and parameter.default is None:
            dumped["parameters"][index]["default"] = None
    for index, case in enumerate(definition.effects.cases):
        dumped_case = dumped["effects"]["cases"][index]
        for name, predicate in case.when.items():
            dumped_predicate = dumped_case["when"][name]
            if "is_value" in predicate.model_fields_set:
                dumped_predicate["is"] = predicate.is_value
            if "is_not_value" in predicate.model_fields_set:
                dumped_predicate["is_not"] = predicate.is_not_value
    return dumped

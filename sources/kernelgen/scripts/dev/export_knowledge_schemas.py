"""Export committed JSON Schemas from the KB V1 Pydantic models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Type

from pydantic import BaseModel

from kernelgen.knowledge.models import (
    CandidateConcept,
    Concept,
    ObservationRecord,
    GitSourceManifest,
    KnowledgeBundle,
    OperatorSignature,
    QueryContext,
    SourcePackage,
    StaticIngestionPlan,
    TargetContext,
)


SCHEMAS: Dict[str, Type[BaseModel]] = {
    "candidate.schema.json": CandidateConcept,
    "concept.schema.json": Concept,
    "observation.schema.json": ObservationRecord,
    "git_source_manifest.schema.json": GitSourceManifest,
    "knowledge_bundle.schema.json": KnowledgeBundle,
    "operator_signature.schema.json": OperatorSignature,
    "query_context.schema.json": QueryContext,
    "source_package.schema.json": SourcePackage,
    "static_ingestion.schema.json": StaticIngestionPlan,
    "target_context.schema.json": TargetContext,
}


def export_schemas(output_dir: Path, *, check: bool = False) -> list[str]:
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    changed = []
    for filename, model in sorted(SCHEMAS.items()):
        rendered = (
            json.dumps(
                model.model_json_schema(mode="validation"),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        path = output_dir / filename
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != rendered:
            changed.append(filename)
            if not check:
                path.write_text(rendered, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("kb/schemas"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = export_schemas(args.output, check=args.check)
    if args.check and changed:
        print(json.dumps({"valid": False, "outdated": changed}, indent=2))
        return 1
    print(json.dumps({"valid": True, "written": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

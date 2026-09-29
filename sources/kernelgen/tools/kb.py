"""Unified command line interface for KernelGen knowledge-base operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.run_facts import (
    KernelGenRunFactReader,
)
from kernelgen.knowledge.round_index import SQLiteRoundIndex
from kernelgen.knowledge.run_archive import KernelGenRunArchive
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.source_index import SQLiteSourceIndex
from kernelgen.knowledge.metrics import build_knowledge_metrics
from kernelgen.knowledge.publishing.lifecycle import (
    delete_concept,
    deprecate_concept,
    revalidation_candidates,
    rollback_publish_batch,
)
from kernelgen.knowledge.publishing.publisher import BatchPublisher
from kernelgen.knowledge.publishing.static_import import StaticKnowledgeImporter
from kernelgen.knowledge.layout import CatalogLayout
from kernelgen.knowledge.models import TargetContext
from kernelgen.knowledge.validation import validate_knowledge_base


def rebuild_indexes(kb: Path, run_archive: Path) -> dict:
    catalog = FilesystemCatalog(kb)
    layout = CatalogLayout(kb)
    concepts = catalog.iter_concepts()
    observations = catalog.iter_observations()
    retrievals = catalog.iter_retrieval_events()
    snapshot = catalog.snapshot()
    usage_snapshot = catalog.usage_snapshot()
    SQLiteKnowledgeIndex(layout.index).rebuild(
        concepts,
        snapshot,
        observations=observations,
        retrievals=retrievals,
        usage_snapshot=usage_snapshot,
    )
    SQLiteRoundIndex(layout.round_index).rebuild(run_archive)
    source_index = SQLiteSourceIndex(layout.source_index, kb).rebuild()
    return {
        "snapshot": snapshot,
        "usage_snapshot": usage_snapshot,
        "concepts": len(concepts),
        "observations": len(observations),
        "retrievals": len(retrievals),
        "knowledge_index": str(layout.index),
        "source_index": str(layout.source_index),
        "source_files": source_index["files"],
        "source_chunks": source_index["chunks"],
        "round_index": str(layout.round_index),
    }


def publish_workspace_batch(
    kb: Path,
    run_archive: Path,
    workspaces: list[Path],
    *,
    run_id: str,
    batch_id: str,
):
    return BatchPublisher(
        kb,
        KernelGenRunFactReader(KernelGenRunArchive(run_archive)),
    ).publish(
        workspaces,
        run_id=run_id,
        batch_id=batch_id,
    )


def _default_archive(kb: Path) -> Path:
    return kb.resolve().parent / "run-archive"


def _validate(arguments: argparse.Namespace) -> int:
    report = validate_knowledge_base(arguments.kb)
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return 0 if report.valid else 1


def _rebuild(arguments: argparse.Namespace) -> int:
    report = rebuild_indexes(
        arguments.kb,
        arguments.run_archive or _default_archive(arguments.kb),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _import_static(arguments: argparse.Namespace) -> int:
    importer = StaticKnowledgeImporter.from_plan_file(
        arguments.kb,
        arguments.plan,
    )
    candidates = importer.build_candidates()
    if not arguments.publish:
        print(
            json.dumps(
                {
                    "valid": True,
                    "mode": "dry-run",
                    "source_package": importer.package.id,
                    "source_revision": importer.package.revision,
                    "candidate_ids": [
                        candidate.candidate_id for candidate in candidates
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    result = importer.publish(
        batch_id=arguments.batch_id or importer.default_batch_id(),
    )
    print(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result.status in {"published", "noop"} else 1


def _publish(arguments: argparse.Namespace) -> int:
    result = publish_workspace_batch(
        arguments.kb,
        arguments.run_archive or _default_archive(arguments.kb),
        arguments.workspaces,
        run_id=arguments.run_id,
        batch_id=arguments.batch_id,
    )
    print(result.model_dump_json(indent=2))
    return 0


def _revalidate(arguments: argparse.Namespace) -> int:
    target = TargetContext.model_validate_json(
        arguments.target_context.read_text(encoding="utf-8")
    )
    candidates = revalidation_candidates(arguments.kb, target)
    print(
        json.dumps(
            {
                "target": target.model_dump(mode="json"),
                "count": len(candidates),
                "candidates": candidates,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _metrics(arguments: argparse.Namespace) -> int:
    report = build_knowledge_metrics(
        arguments.kb,
        arguments.run_archive or _default_archive(arguments.kb),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _rollback(arguments: argparse.Namespace) -> int:
    report = rollback_publish_batch(
        arguments.kb,
        arguments.batch_id,
        apply=arguments.apply,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _deprecate(arguments: argparse.Namespace) -> int:
    report = deprecate_concept(
        arguments.kb,
        arguments.concept_id,
        reason=arguments.reason,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _delete(arguments: argparse.Namespace) -> int:
    report = delete_concept(
        arguments.kb,
        arguments.concept_id,
        reason=arguments.reason,
        apply=arguments.apply,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _add_kb_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kb", type=Path, default=Path("kb"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    _add_kb_argument(validate)
    validate.set_defaults(handler=_validate)

    rebuild = commands.add_parser("rebuild")
    _add_kb_argument(rebuild)
    rebuild.add_argument("--run-archive", type=Path)
    rebuild.set_defaults(handler=_rebuild)

    import_static = commands.add_parser("import")
    _add_kb_argument(import_static)
    import_static.add_argument("--plan", type=Path, required=True)
    import_static.add_argument(
        "--publish",
        action="store_true",
        help="publish after validation; default is dry-run",
    )
    import_static.add_argument("--batch-id", default="")
    import_static.set_defaults(handler=_import_static)

    publish = commands.add_parser("publish")
    _add_kb_argument(publish)
    publish.add_argument("--run-archive", type=Path)
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--batch-id", required=True)
    publish.add_argument("workspaces", nargs="+", type=Path)
    publish.set_defaults(handler=_publish)

    revalidate = commands.add_parser("revalidate")
    _add_kb_argument(revalidate)
    revalidate.add_argument("--target-context", type=Path, required=True)
    revalidate.set_defaults(handler=_revalidate)

    metrics = commands.add_parser("metrics")
    _add_kb_argument(metrics)
    metrics.add_argument("--run-archive", type=Path)
    metrics.set_defaults(handler=_metrics)

    deprecate = commands.add_parser(
        "deprecate",
        help="mark one Concept deprecated and rebuild the index",
    )
    _add_kb_argument(deprecate)
    deprecate.add_argument("--concept-id", required=True)
    deprecate.add_argument("--reason", required=True)
    deprecate.set_defaults(handler=_deprecate)

    delete = commands.add_parser(
        "delete",
        help="plan or apply hard deletion of an empty deprecated Concept",
    )
    _add_kb_argument(delete)
    delete.add_argument("--concept-id", required=True)
    delete.add_argument("--reason", required=True)
    delete.add_argument(
        "--apply",
        action="store_true",
        help="apply the deletion; default is a dry-run",
    )
    delete.set_defaults(handler=_delete)

    rollback = commands.add_parser("rollback")
    _add_kb_argument(rollback)
    rollback.add_argument("--batch-id", required=True)
    rollback.add_argument(
        "--apply",
        action="store_true",
        help=(
            "apply an Observation-only rollback; Concept changes require "
            "Catalog git revert"
        ),
    )
    rollback.set_defaults(handler=_rollback)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

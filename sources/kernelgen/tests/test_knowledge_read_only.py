"""Production contracts for immutable V1 Knowledge Catalog access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kernelgen.workflows.knowledge_bridge as bridge_module
from kernelgen.knowledge.bootstrap import build_workspace_services
from kernelgen.knowledge.config import KnowledgeConfig, KnowledgeMode
from kernelgen.knowledge.context import (
    KnowledgeWorkspaceMaterializer,
    build_operator_signature,
    build_query_context,
    build_target_context,
)
from kernelgen.knowledge.layout import CatalogLayout, KnowledgeLayout
from kernelgen.knowledge.source_index import SQLiteSourceIndex
from kernelgen.tests._knowledge_runtime_support import (
    _DEFINITION,
    _seed,
    _service_status,
)
from kernelgen.workflows.knowledge_bridge import KernelGenKnowledgeBridge


def _tree_state(root: Path) -> list[tuple]:
    """Capture exact temporary-tree content and write-relevant metadata."""

    paths = [root, *sorted(root.rglob("*"))]
    return [
        (
            path.relative_to(root).as_posix(),
            path.is_dir(),
            path.stat().st_mode,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in paths
    ]


def _materialize_access(
    catalog_root: Path,
    workspace: Path,
    config: KnowledgeConfig,
) -> None:
    workspace.mkdir(parents=True)
    KnowledgeWorkspaceMaterializer(
        catalog_root=catalog_root,
        mode=config.mode,
        run_id="read-only-contract",
        operator_signature=build_operator_signature(_DEFINITION),
        target_context=build_target_context(
            target_hardware="A100",
            implementation_language="triton",
            service_status=_service_status(),
        ),
        derived_root=config.resolved_derived_root,
        index_rebuild_enabled=config.index_rebuild_enabled,
    ).materialize(workspace)


def _query(workspace: Path):
    return build_workspace_services(workspace).query.execute(
        build_query_context(
            workspace,
            phase="initial",
            task="architecture_selection",
            question="How should optimization use two_stage_reduction?",
        )
    )


def test_read_only_query_leaves_prebuilt_catalog_tree_exactly_unchanged(
    tmp_path,
):
    catalog = tmp_path / "catalog"
    _seed(catalog)

    writer = tmp_path / "writer"
    _materialize_access(
        catalog,
        writer,
        KnowledgeConfig(catalog_root=catalog),
    )
    assert _query(writer).direct

    reader = tmp_path / "reader"
    _materialize_access(
        catalog,
        reader,
        KnowledgeConfig(
            mode=KnowledgeMode.READ_ONLY_V1,
            catalog_root=catalog,
        ),
    )
    before = _tree_state(catalog)

    assert _query(reader).direct

    assert _tree_state(catalog) == before


def test_read_only_query_fails_closed_when_catalog_index_is_missing(tmp_path):
    catalog = tmp_path / "catalog"
    _seed(catalog)
    reader = tmp_path / "reader"
    _materialize_access(
        catalog,
        reader,
        KnowledgeConfig(
            mode=KnowledgeMode.READ_ONLY_V1,
            catalog_root=catalog,
        ),
    )
    before = _tree_state(catalog)

    with pytest.raises(RuntimeError, match="missing or stale"):
        _query(reader)

    assert _tree_state(catalog) == before
    assert not CatalogLayout(catalog).index.exists()


def test_read_only_query_rebuilds_only_in_external_derived_root(tmp_path):
    catalog = tmp_path / "catalog"
    _seed(catalog)
    derived = tmp_path / "derived"
    reader = tmp_path / "reader"
    _materialize_access(
        catalog,
        reader,
        KnowledgeConfig(
            mode=KnowledgeMode.READ_ONLY_V1,
            catalog_root=catalog,
            derived_root=derived,
        ),
    )
    before = _tree_state(catalog)

    assert _query(reader).direct

    assert (derived / "knowledge.db").is_file()
    assert _tree_state(catalog) == before
    assert not CatalogLayout(catalog).derived.exists()


def test_read_only_source_index_fails_closed_without_prebuilt_index(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    index = SQLiteSourceIndex(
        CatalogLayout(catalog).source_index,
        catalog,
        read_only=True,
    )
    before = _tree_state(catalog)

    with pytest.raises(RuntimeError, match="source index is missing or stale"):
        index.ensure_current()

    assert _tree_state(catalog) == before


def test_read_only_epoch_records_workspace_noop_without_catalog_write(
    tmp_path,
    monkeypatch,
):
    catalog = tmp_path / "catalog"
    _seed(catalog)
    config = KnowledgeConfig(
        mode=KnowledgeMode.READ_ONLY_V1,
        catalog_root=catalog,
    )
    monkeypatch.setattr(
        bridge_module,
        "_service_status",
        lambda _: _service_status(),
    )
    bridge = KernelGenKnowledgeBridge(
        config=config,
        definition=_DEFINITION,
        target_hardware="A100",
        implementation_language="triton",
        eval_server_url="http://unused.invalid",
        run_id="read-only-epoch",
    )
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    bridge.materializer().materialize(workspace)
    before = _tree_state(catalog)

    result = bridge.publish_epoch([workspace], epoch_num=1)

    assert result.status == "noop"
    payload = json.loads(
        KnowledgeLayout(workspace).publish_result.read_text(encoding="utf-8")
    )
    assert payload["status"] == "noop"
    assert payload["reviewer_mode"] == "off"
    assert _tree_state(catalog) == before

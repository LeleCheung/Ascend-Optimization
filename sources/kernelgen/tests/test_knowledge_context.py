"""Knowledge context and target matching behavior tests."""

from __future__ import annotations

from kernelgen.tests._knowledge_runtime_support import *  # noqa: F403
from kernelgen.tests._knowledge_runtime_support import (
    _DEFINITION,
    _evidence_state,
    _lifecycle_candidate,
    _materialize,
    _reference_body,
    _seed,
    _service_status,
    _target_devices_match,
)
from kernelgen.knowledge.config import KnowledgeReviewerMode


def test_knowledge_config_supports_explicit_access_modes(tmp_path):
    config = KnowledgeConfig(catalog_root=tmp_path / "catalog")

    assert list(KnowledgeMode) == [
        KnowledgeMode.READ_WRITE_V1,
        KnowledgeMode.READ_ONLY_V1,
    ]
    assert config.mode == KnowledgeMode.READ_WRITE_V1
    assert config.reviewer_mode == KnowledgeReviewerMode.OFF
    assert config.reads_v1 is True
    assert config.writes_v1 is True
    assert config.index_rebuild_enabled is True

    read_only = KnowledgeConfig(
        mode=KnowledgeMode.READ_ONLY_V1,
        catalog_root=tmp_path / "catalog",
    )
    assert read_only.reads_v1 is True
    assert read_only.writes_v1 is False
    assert read_only.resolved_derived_root == (
        tmp_path / "catalog" / ".derived"
    ).resolve()
    assert read_only.index_rebuild_enabled is False

    external = KnowledgeConfig(
        mode=KnowledgeMode.READ_ONLY_V1,
        catalog_root=tmp_path / "catalog",
        derived_root=tmp_path / "indexes",
    )
    assert external.resolved_derived_root == tmp_path / "indexes"
    assert external.index_rebuild_enabled is True

    with pytest.raises(ValueError, match="reviewer mode=off"):
        KnowledgeConfig(
            mode=KnowledgeMode.READ_ONLY_V1,
            reviewer_mode=KnowledgeReviewerMode.SHADOW,
            catalog_root=tmp_path / "catalog",
        )
    with pytest.raises(ValueError, match="outside catalog_root"):
        KnowledgeConfig(
            mode=KnowledgeMode.READ_ONLY_V1,
            catalog_root=tmp_path / "catalog",
            derived_root=tmp_path / "catalog" / "cache",
        )
    assert list(KnowledgeReviewerMode) == [
        KnowledgeReviewerMode.OFF,
        KnowledgeReviewerMode.SHADOW,
        KnowledgeReviewerMode.ENFORCE,
    ]

    for reviewer_mode in KnowledgeReviewerMode:
        assert KnowledgeConfig(
            catalog_root=tmp_path / reviewer_mode.value,
            reviewer_mode=reviewer_mode,
        ).reviewer_mode == reviewer_mode

    with pytest.raises(ValueError):
        KnowledgeConfig(
            catalog_root=tmp_path / "invalid-reviewer",
            reviewer_mode="publish_directly",
        )

    for legacy_mode in ("legacy", "shadow", "read_v1"):
        with pytest.raises(ValueError):
            KnowledgeConfig(
                mode=legacy_mode,
                catalog_root=tmp_path / legacy_mode,
            )


def test_workspace_state_accepts_both_v1_access_modes():
    state = WorkspaceKnowledgeState(
        catalog_ref="/catalog",
        mode="read_write_v1",
        run_id="run-1",
        workspace_id="agent0",
        operator_signature_ref=(
            ".kernelgen/knowledge/operator-signature.json"
        ),
        target_context_ref=".kernelgen/knowledge/target-context.json",
    )

    assert state.mode == "read_write_v1"
    assert state.index_rebuild_enabled is True
    read_only = WorkspaceKnowledgeState.model_validate(
        {
            **state.model_dump(mode="json"),
            "mode": "read_only_v1",
            "index_rebuild_enabled": False,
        }
    )
    assert read_only.mode == "read_only_v1"
    assert read_only.index_rebuild_enabled is False
    for removed_mode in ("legacy", "shadow", "read_v1"):
        with pytest.raises(ValueError):
            WorkspaceKnowledgeState(
                catalog_ref="/catalog",
                mode=removed_mode,
                run_id="run-1",
                workspace_id="agent0",
                operator_signature_ref=(
                    ".kernelgen/knowledge/operator-signature.json"
                ),
                target_context_ref=(
                    ".kernelgen/knowledge/target-context.json"
                ),
            )


def test_materializer_does_not_delete_catalog_at_workspace_kb_path(tmp_path):
    catalog_root = tmp_path / "kb"
    concept = _seed(catalog_root)

    _materialize(catalog_root, tmp_path)

    assert FilesystemCatalog(catalog_root).get_concept(concept.id) == concept
    state = WorkspaceKnowledgeState.model_validate_json(
        KnowledgeLayout(tmp_path).state.read_text(encoding="utf-8")
    )
    assert Path(state.catalog_ref) == catalog_root.resolve()


def test_operator_and_target_context_are_deterministic():
    first_signature = build_operator_signature(_DEFINITION)
    second_signature = build_operator_signature(_DEFINITION)
    assert first_signature == second_signature
    assert "reduction" in first_signature.motifs

    first_target = build_target_context(
        target_hardware="A100",
        implementation_language="triton",
        service_status=_service_status(),
    )
    second_target = build_target_context(
        target_hardware="A100",
        implementation_language="triton",
        service_status=_service_status(),
    )
    assert first_target == second_target
    assert first_target.backend == "cuda"
    assert first_target.source == "eval_service"

    expanded_signature = build_operator_signature(
        {
            "name": "sort_broadcast_transpose",
            "op_type": "conversion",
            "reference": (
                "values = torch.sort(torch.broadcast_to(x, shape)).values"
                ".transpose(0, 1)"
            ),
            "inputs": {},
            "outputs": {},
        }
    )
    assert {"sort", "broadcast", "conversion"}.issubset(
        expanded_signature.motifs
    )

    ascend_target = build_target_context(
        target_hardware="Ascend910B",
        implementation_language="triton",
        service_status={
            "backend": "npu",
            "target": {
                "backend": "ascend",
                "vendor": "huawei",
                "device": "910B4-1",
                "architecture": "DAV_2201",
            },
            "software": {
                "language": "triton",
                "language_version": "3.2.0",
                "compiler": "triton-ascend",
                "compiler_version": "3.2.0",
                "runtime": "cann",
                "runtime_version": "8.5.0",
                "driver_version": "25.2.0",
            },
        },
    )
    assert ascend_target.backend == "ascend"
    assert ascend_target.vendor == "huawei"
    assert ascend_target.device == "Ascend910B"
    assert ascend_target.architecture == "DAV_2201"
    assert ascend_target.software.language_version == "3.2.0"
    assert ascend_target.software.compiler == "triton-ascend"
    assert ascend_target.software.compiler_version == "3.2.0"
    assert ascend_target.software.runtime == "cann"
    assert ascend_target.software.runtime_version == "8.5.0"
    assert ascend_target.software.driver_version == "25.2.0"

    enum_like_definition = {
        **_DEFINITION,
        "inputs": {"x": {"shape": [16], "dtype": "DType.FLOAT32"}},
    }
    assert build_operator_signature(enum_like_definition).dtypes == [
        "float32"
    ]


def test_ledger_target_accepts_same_ascend_family_alias_only():
    assert _target_devices_match("Ascend910B", "910B4-1")
    assert _target_devices_match("ASCEND910B", "Ascend 910B")
    assert not _target_devices_match("Ascend910B", "Ascend910A")
    assert not _target_devices_match("A100", "H100")
    alias_match = match_scope(
        Scope(
            target=TargetScope(
                level="device",
                devices=["910B4-1"],
            )
        ),
        QueryContext(
            phase="initial",
            task="constraint_check",
            question="device alias compatibility",
            operator_signature=build_operator_signature(_DEFINITION),
            target_context=build_target_context(
                target_hardware="Ascend 910B",
                implementation_language="triton",
                service_status=_service_status(
                    backend="ascend",
                    device="910B4-1",
                ),
            ),
        ),
    )
    assert alias_match.level == "direct"
    assert alias_match.matched_on == ["target:device:Ascend910B"]


def test_device_scope_checks_backend_and_architecture():
    scope = Scope(
        target=TargetScope(
            level="device",
            backend="ascend",
            architecture="DAV_2201",
            devices=["Ascend910B"],
        )
    )
    signature = build_operator_signature(_DEFINITION)

    matching = match_scope(
        scope,
        QueryContext(
            phase="initial",
            task="constraint_check",
            question="matching Ascend device",
            operator_signature=signature,
            target_context=build_target_context(
                target_hardware="910B4-1",
                implementation_language="triton",
                service_status={
                    "target": {
                        "backend": "ascend",
                        "device": "910B4-1",
                        "architecture": "DAV_2201",
                    }
                },
            ),
        ),
    )
    wrong_backend = match_scope(
        scope,
        QueryContext(
            phase="initial",
            task="constraint_check",
            question="same label on wrong backend",
            operator_signature=signature,
            target_context=build_target_context(
                target_hardware="Ascend910B",
                implementation_language="triton",
                service_status={
                    "target": {
                        "backend": "cuda",
                        "device": "Ascend910B",
                        "architecture": "DAV_2201",
                    }
                },
            ),
        ),
    )

    assert matching.level == "direct"
    assert matching.matched_on == [
        "target:architecture:DAV_2201",
        "target:backend:ascend",
        "target:device:Ascend910B",
    ]
    assert wrong_backend.level == "incompatible"
    assert wrong_backend.conflicts == ["target:backend:cuda"]


def test_query_prefers_device_variant_over_backend_variant(tmp_path):
    catalog_root = tmp_path / "catalog"
    generic = _seed(catalog_root, level="backend", backend="cuda")
    specific = generic.model_copy(
        update={
            "id": "kg:method:two-stage-reduction-a100",
            "scope": Scope(
                target=TargetScope(
                    level="device",
                    backend="cuda",
                    devices=["A100"],
                ),
                operator=generic.scope.operator,
            ),
        }
    )
    specific = specific.model_copy(
        update={
            "managed": specific.managed.model_copy(
                update={"content_hash": canonical_concept_hash(specific)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(specific)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)

    bundle = build_workspace_services(workspace).query.execute(
        build_query_context(
            workspace,
            phase="initial",
            task="architecture_selection",
            question="How should optimization use two_stage_reduction?",
        )
    )

    assert [item.concept_ref for item in bundle.direct] == [
        "kg:method:two-stage-reduction-a100"
    ]

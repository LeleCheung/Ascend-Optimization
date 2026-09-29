import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen_server import Definition, Workload
from kernelgen_server.schema import (
    AdapterCapabilities,
    AdapterCase,
    AdapterManifest,
    CandidateContract,
    CaseList,
)

from kernelgen.data.evaluation_snapshot import (
    CatalogEvaluationSnapshot,
    EVALUATION_SNAPSHOT_RELATIVE_PATH,
    build_catalog_evaluation_snapshot,
    freeze_catalog_evaluation_snapshot,
    load_catalog_evaluation_snapshot,
    snapshot_optimization_context,
)
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.data.tool_context import ToolContext
from kernelgen.tools.kernelgen_server_adapter import prepare_evaluation_bundle
from kernelgen.workflows.optimization.kernelgen import KernelGenInput
from kernelgen.workflows.optimization.kernelgen.epoch import build_coder_input
from kernelgen.workflows.optimization.kernelgen.preparation import freeze_run_evaluation_contract


def _write_catalog(root: Path, *, scalar: int = 1) -> None:
    operator_root = root / "ops" / "pointwise" / "identity"
    operator_root.mkdir(parents=True, exist_ok=True)
    definition = Definition(
        name="identity",
        description=f"identity-{scalar}",
        parameters=[{"name": "x", "required": True}],
        outputs=["output"],
    )
    correctness = Workload(
        name="corr-0",
        inputs={"x": {"type": "scalar", "value": scalar}},
        seed=42,
    )
    timing = Workload(
        name="time-0",
        inputs={"x": {"type": "scalar", "value": scalar + 1}},
        seed=42,
    )
    (operator_root / "definition.json").write_text(
        definition.model_dump_json(exclude_unset=True), encoding="utf-8"
    )
    (operator_root / "oracle.py").write_text(
        'REFERENCE_DEVICE = "target"\n\ndef run(x):\n    return x\n',
        encoding="utf-8",
    )
    (operator_root / "correctness.jsonl").write_text(
        correctness.model_dump_json() + "\n", encoding="utf-8"
    )
    (operator_root / "timing.jsonl").write_text(
        timing.model_dump_json() + "\n", encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "name": "fixture",
                "evaluator": "native",
                "layout": "per-operator",
                "generated_by": f"fixture-{scalar}",
                "operators": [
                    {
                        "name": "identity",
                        "group": "pointwise",
                        "definition": "definitions/identity.json",
                        "correctness_workloads": (
                            "workloads/identity.correctness.jsonl"
                        ),
                        "timing_workloads": "workloads/identity.timing.jsonl",
                        "num_correctness_workloads": 1,
                        "num_timing_workloads": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def fixture_catalog(tmp_path, monkeypatch):
    catalog_root = tmp_path / "catalog"
    _write_catalog(catalog_root)
    from kernelgen.data import evaluation_snapshot as snapshot_module

    monkeypatch.setattr(
        snapshot_module,
        "resolve_builtin_catalog_path",
        lambda catalog_name: catalog_root,
    )
    from kernelgen.data import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "resolve_builtin_catalog_path",
        lambda catalog_name: catalog_root,
    )
    return catalog_root


def test_freeze_reuses_run_contract_after_live_catalog_changes(
    tmp_path,
    fixture_catalog,
):
    workspace = tmp_path / "run"
    first, path = freeze_catalog_evaluation_snapshot(
        workspace, "fixture", "identity"
    )
    _write_catalog(fixture_catalog, scalar=9)

    resumed, resumed_path = freeze_catalog_evaluation_snapshot(
        workspace, "fixture", "identity"
    )

    assert resumed_path == path
    assert resumed == first
    assert resumed.catalog_generated_by == "fixture-1"
    assert resumed.correctness_workloads[0]["inputs"]["x"]["value"] == 1
    assert load_catalog_evaluation_snapshot(path) == first


def test_snapshot_preserves_explicit_none_defaults(tmp_path, fixture_catalog):
    operator = fixture_catalog / "ops/pointwise/identity"
    path = operator / "definition.json"
    definition = json.loads(path.read_text())
    definition["parameters"].append({"name": "scale", "required": False, "default": None})
    path.write_text(json.dumps(definition))
    (operator / "oracle.py").write_text('REFERENCE_DEVICE = "target"\ndef run(x, scale=None): return x\n')
    for name in ("correctness", "timing"):
        path = operator / f"{name}.jsonl"
        workload = json.loads(path.read_text())
        workload["inputs"]["scale"] = {"type": "literal", "value": None}
        path.write_text(json.dumps(workload) + "\n")
    snapshot, path = freeze_catalog_evaluation_snapshot(tmp_path / "run", "fixture", "identity")
    loaded = load_catalog_evaluation_snapshot(path)
    assert loaded == snapshot
    parameters = loaded.definition["parameters"]
    assert "default" not in parameters[0]
    assert "default" in parameters[1] and parameters[1]["default"] is None
    assert loaded.correctness_workloads[0]["inputs"]["scale"] == {"type": "literal", "value": None}
    assert loaded.native_operator()[0].parameters[1].default is None


def test_snapshot_drives_prompt_and_rejects_a_changed_bound_catalog(
    tmp_path,
    fixture_catalog,
    monkeypatch,
):
    from kernelgen_client import http as client

    monkeypatch.setattr(
        client,
        "inspect",
        lambda request, server_url: AdapterManifest(
            kind="native",
            benchmark_fingerprint="fixture-fingerprint",
            candidate_contract=CandidateContract(signature="(x)"),
            capabilities=AdapterCapabilities(preflight=True, profile=True),
            case_list=CaseList(
                adapter_kind="native",
                operator="identity",
                benchmark_fingerprint="fixture-fingerprint",
                cases=[AdapterCase(case_id="time-0", ordinal=0)],
            ),
        ),
    )
    workspace = tmp_path / "run"
    snapshot, path = freeze_catalog_evaluation_snapshot(
        workspace, "fixture", "identity"
    )
    prompt_definition, prompt_workloads = snapshot_optimization_context(snapshot)
    _write_catalog(fixture_catalog, scalar=9)
    kernel = workspace / "main.py"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text("def run(x): return x", encoding="utf-8")
    context = SimpleNamespace(
        definition="identity",
        catalog_name="fixture",
        evaluation_snapshot_path=str(path),
        implementation_language=SimpleNamespace(value="python"),
        eval_server_url="http://server:8000",
    )

    assert prompt_definition.description == "identity-1"
    assert [item["uuid"] for item in prompt_workloads] == ["corr-0", "time-0"]
    with pytest.raises(RuntimeError, match="catalog changed"):
        prepare_evaluation_bundle(kernel, context)


def test_kernel_gen_reuses_one_root_snapshot_across_epochs(
    tmp_path,
    fixture_catalog,
):
    initial = build_catalog_evaluation_snapshot(
        "kernelswift", "identity"
    )
    definition, _ = snapshot_optimization_context(initial)
    inp = KernelGenInput(
        definition=definition,
        target_hardware="A100",
        catalog_name="kernelswift",
    )

    first_inp, snapshot = freeze_run_evaluation_contract(tmp_path / "run", inp)
    first = build_coder_input(first_inp, {}, evaluation_snapshot=snapshot)
    _write_catalog(fixture_catalog, scalar=9)
    resumed_inp, snapshot = freeze_run_evaluation_contract(tmp_path / "run", inp)
    resumed = build_coder_input(resumed_inp, {}, evaluation_snapshot=snapshot)

    assert first["definition"]["description"] == "identity-1"
    assert resumed["definition"] == first["definition"]
    assert resumed["workloads"] == first["workloads"]
    assert (
        tmp_path / "run" / EVALUATION_SNAPSHOT_RELATIVE_PATH
    ).is_file()


def test_snapshot_rejects_a_different_workspace_identity(
    tmp_path,
    fixture_catalog,
):
    workspace = tmp_path / "run"
    freeze_catalog_evaluation_snapshot(workspace, "fixture", "identity")

    with pytest.raises(ValueError, match="different request"):
        freeze_catalog_evaluation_snapshot(workspace, "other", "identity")


def test_snapshot_allows_omitted_optional_parameters():
    snapshot = CatalogEvaluationSnapshot(
        catalog_name="fixture",
        catalog_api_version="v6.2",
        definition_name="identity",
        definition=Definition(
            name="identity",
            parameters=[
                {"name": "x", "required": True},
                {"name": "alpha", "required": False, "default": 1},
            ],
            outputs=["output"],
        ).model_dump(mode="json", exclude_none=True),
        correctness_workloads=[
            Workload(
                name="corr-0",
                inputs={"x": {"type": "scalar", "value": 1}},
            ).model_dump(mode="json"),
        ],
        timing_workloads=[],
    )

    assert snapshot.definition_name == "identity"


def test_measured_legacy_kernelswift_run_cannot_silently_adopt_new_contract(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "run"
    workspace.mkdir()
    (workspace / ".ledger.json").write_text(
        json.dumps({"rounds": [{"round_num": 1}]}), encoding="utf-8"
    )
    from kernelgen.data import evaluation_snapshot as snapshot_module

    monkeypatch.setattr(
        snapshot_module,
        "build_catalog_evaluation_snapshot",
        lambda *args: pytest.fail("legacy run must fail before reading live catalog"),
    )

    with pytest.raises(RuntimeError, match="cannot safely resume"):
        freeze_catalog_evaluation_snapshot(
            workspace,
            "kernelswift",
            "identity",
        )


def test_measured_multi_epoch_kernelswift_run_requires_a_root_snapshot(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "run"
    ledger = workspace / "1R" / "agent0" / ".ledger.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"rounds": [{"round_num": 1}]}), encoding="utf-8"
    )
    from kernelgen.data import evaluation_snapshot as snapshot_module

    monkeypatch.setattr(
        snapshot_module,
        "build_catalog_evaluation_snapshot",
        lambda *args: pytest.fail("legacy run must fail before reading live catalog"),
    )

    with pytest.raises(RuntimeError, match="cannot safely resume"):
        freeze_catalog_evaluation_snapshot(
            workspace,
            "kernelswift",
            "identity",
        )


def test_tool_context_persists_the_orchestrator_owned_snapshot_path(tmp_path):
    snapshot_path = tmp_path / EVALUATION_SNAPSHOT_RELATIVE_PATH
    context = ToolContext(
        definition="identity",
        target_hardware="A100",
        implementation_language=ImplementationLanguage.TRITON,
        catalog_name="fixture",
        evaluation_snapshot_path=str(snapshot_path),
    )

    written = context.write(tmp_path)
    loaded = ToolContext.model_validate_json(written.read_text(encoding="utf-8"))

    assert loaded.evaluation_snapshot_path == str(snapshot_path)

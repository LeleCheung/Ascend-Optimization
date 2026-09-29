"""Tests for exact-scope current-best solution slots."""

from __future__ import annotations

import json
import subprocess

import pytest

import kernelgen.workflows.knowledge_bridge as knowledge_bridge_module
from kernelgen.data.ledger import Ledger
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.knowledge.config import KnowledgeMode
from kernelgen.knowledge.context import (
    KnowledgeWorkspaceMaterializer,
    build_operator_signature,
    build_target_context,
)
from kernelgen.knowledge.layout import KnowledgeLayout
from kernelgen.knowledge.solutions import SolutionRegistry
from kernelgen.knowledge.validation import validate_knowledge_base
from kernelgen.tests.helpers import experiment_plan
from kernelgen.workflows.knowledge_bridge import KernelGenKnowledgeBridge


_DEFINITION = {
    "name": "sum_rows",
    "op_type": "reduction",
    "axes": {"reduction_size": {"type": "const", "value": 4096}},
    "inputs": {"x": {"shape": [16, 4096], "dtype": "float16"}},
    "outputs": {"y": {"shape": [16], "dtype": "float32"}},
    "reference": "def run(x): return x.sum(-1)",
}
_BENCHMARK_ID = "flaggems-v5-v5.1"


def _target(device: str = "A100", architecture: str = "sm80"):
    return build_target_context(
        target_hardware=device,
        implementation_language="triton",
        service_status={
            "backend": "cuda",
            "target": {
                "backend": "cuda",
                "architecture": architecture,
                "device": device,
            },
        },
    )


def _promote(
    registry: SolutionRegistry,
    *,
    code: str,
    geo_mean: float,
    round_num: int,
):
    return registry.promote(
        signature=build_operator_signature(_DEFINITION),
        target=_target(),
        benchmark_id=_BENCHMARK_ID,
        run_id=f"run-{round_num}",
        workspace_id=f"workspace-{round_num}",
        round_num=round_num,
        archive_ref=f"run://run-{round_num}",
        code=code,
        geo_mean=geo_mean,
        start_mode="fresh",
    )


def test_solution_slot_keeps_only_the_strictly_best_result(tmp_path):
    registry = SolutionRegistry(tmp_path / "catalog")
    signature = build_operator_signature(_DEFINITION)
    target = _target()

    first = _promote(
        registry,
        code="def kernel(x): return x",
        geo_mean=1.10,
        round_num=1,
    )
    assert first is not None
    assert _promote(
        registry,
        code="def kernel(x): return x + 1",
        geo_mean=1.10,
        round_num=2,
    ) is None

    improved = _promote(
        registry,
        code="def kernel(x): return x + 2",
        geo_mean=1.20,
        round_num=3,
    )
    assert improved is not None

    seed = registry.resolve(
        signature,
        target,
        benchmark_id=_BENCHMARK_ID,
    )
    assert seed is not None
    assert seed.code == "def kernel(x): return x + 2"
    assert seed.manifest.round_num == 3
    assert len(list((tmp_path / "catalog").glob("**/manifest.json"))) == 1


def test_solution_promotion_rejects_dirty_target_slot(tmp_path):
    catalog = tmp_path / "catalog"
    registry = SolutionRegistry(catalog)
    assert _promote(
        registry,
        code="def kernel(x): return x",
        geo_mean=1.10,
        round_num=1,
    ) is not None

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(catalog),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init")
    git("add", "-A")
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "seed",
    )
    code_path = next(catalog.glob("**/best_kernel.py"))
    code_path.write_text("dirty\n", encoding="utf-8")

    assert _promote(
        registry,
        code="def kernel(x): return x - 1",
        geo_mean=1.00,
        round_num=2,
    ) is None

    with pytest.raises(
        RuntimeError,
        match="pre-existing dirty paths: .*best_kernel.py",
    ):
        _promote(
            registry,
            code="def kernel(x): return x + 1",
            geo_mean=1.20,
            round_num=2,
        )
    assert code_path.read_text(encoding="utf-8") == "dirty\n"


def test_solution_scope_is_enforced_without_hash_validation(tmp_path):
    catalog = tmp_path / "catalog"
    registry = SolutionRegistry(catalog)
    _promote(
        registry,
        code="def kernel(x): return x",
        geo_mean=1.10,
        round_num=1,
    )
    signature = build_operator_signature(_DEFINITION)

    assert registry.resolve(
        signature,
        _target("H100"),
        benchmark_id=_BENCHMARK_ID,
    ) is None
    assert registry.resolve(
        signature,
        _target("A100", "sm90"),
        benchmark_id=_BENCHMARK_ID,
    ) is None

    manifest_path = next(catalog.glob("**/manifest.json"))
    assert "solution_sha256" not in json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    code_path = next(catalog.glob("**/best_kernel.py"))
    code_path.write_text("updated", encoding="utf-8")
    seed = registry.resolve(
        signature,
        _target(),
        benchmark_id=_BENCHMARK_ID,
    )
    assert seed is not None
    assert seed.code == "updated"
    assert validate_knowledge_base(catalog).valid


def test_workspace_state_records_fork_lineage(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    KnowledgeWorkspaceMaterializer(
        catalog_root=tmp_path / "catalog",
        mode=KnowledgeMode.READ_WRITE_V1,
        run_id="run-1",
        operator_signature=build_operator_signature(_DEFINITION),
        target_context=_target(),
        start_mode="fork",
        parent_solution_ref="solution://cuda--a100--triton/sum_rows/bench",
    ).materialize(workspace)

    state = json.loads(
        KnowledgeLayout(workspace).state.read_text(encoding="utf-8")
    )
    assert state["start_mode"] == "fork"
    assert state["parent_solution_ref"].startswith("solution://")


@pytest.mark.parametrize("best_export", ["intact", "missing", "stale"])
def test_bridge_promotes_and_resolves_exact_fork_seed(
    tmp_path,
    monkeypatch,
    best_export,
):
    monkeypatch.setattr(
        knowledge_bridge_module,
        "_service_status",
        lambda _: {
            "backend": "cuda",
            "target": {
                "backend": "cuda",
                "architecture": "sm80",
                "device": "A100",
            },
        },
    )
    config = KnowledgeConfig(catalog_root=tmp_path / "catalog")
    workspace = tmp_path / "run-1" / "1R" / "agent0"
    workspace.mkdir(parents=True)
    fresh = KernelGenKnowledgeBridge(
        config=config,
        definition=_DEFINITION,
        target_hardware="A100",
        implementation_language="triton",
        eval_server_url="http://eval.invalid",
        run_id="run-1",
        benchmark_id=_BENCHMARK_ID,
        start_mode="fresh",
    )
    fresh.materializer().materialize(workspace)
    Ledger(workspace).record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.25,
            "num_workloads": 2,
            "num_passed": 2,
            "per_workload": [
                {
                    "uuid": "correctness-0",
                    "phase": "correctness",
                    "status": "PASSED",
                },
                {
                    "uuid": "timing-0",
                    "phase": "timing",
                    "status": "PASSED",
                    "speedup": 1.25,
                },
            ],
        },
        "def run(x): return x.sum(-1)",
        experiment_plan(1),
        definition_name="sum_rows",
        target_hardware="A100",
    )

    export_path = Ledger(workspace).best_kernel_path
    if best_export == "missing":
        export_path.unlink()
    elif best_export == "stale":
        export_path.write_text("stale_unmeasured_code", encoding="utf-8")

    promoted = fresh.promote_solution(workspace)
    assert promoted is not None
    assert promoted.benchmark_id == _BENCHMARK_ID
    assert fresh.promote_solution(workspace) is None

    fork = KernelGenKnowledgeBridge(
        config=config,
        definition=_DEFINITION,
        target_hardware="A100",
        implementation_language="triton",
        eval_server_url="http://eval.invalid",
        run_id="run-2",
        benchmark_id=_BENCHMARK_ID,
        start_mode="fork",
    )
    seed = fork.resolve_fork_seed()

    assert seed.code == "def run(x): return x.sum(-1)"
    assert fork.parent_solution_ref == promoted.solution_ref

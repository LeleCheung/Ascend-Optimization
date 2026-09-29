import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kernelgen.tools.kernel_todo_v2_pipeline import (
    KernelTodoV2PipelineResults,
    PipelineStatus,
    StageName,
    StageResult,
    load_pipeline_results,
    write_pipeline_results,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "run_kernel_todo_v2_kernelgen.py"
)
SPEC = importlib.util.spec_from_file_location("run_kernel_todo_v2_kernelgen", SCRIPT)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _stage(
    attempt_id: str,
    *,
    verdict: str = "passed",
    geo_mean: float | None = 0.7,
    reason: str = "",
) -> StageResult:
    return StageResult(
        attempt_id=attempt_id,
        verdict=verdict,
        geo_mean=geo_mean,
        evidence=[f"{attempt_id}.json"],
        reason=reason,
    )


def _results(operators: list[str]) -> str:
    rows = "\n".join(
        f"| `{operator}` | 通过 | 通过 | 待二阶段优化 | 0.700x | "
        "未发现 | — | 待运行 Native KernelGen。 | — |"
        for operator in operators
    )
    return (
        "| 总算子数 | reference通过数 | Gems 可计时数 | 第一阶段达标数 | "
        "待二阶段优化数 | 第二阶段达标数 | 最终失败数 | 阻塞数 |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {len(operators)} | {len(operators)} | {len(operators)} | 0 | "
        f"{len(operators)} | 0 | 0 | 0 |\n\n"
        "| 算子名 | Reference | Gems 可计时 | 优化状态 | 加速比 | "
        "Hack 情况 | 原因 | 后续方向 | code_path |\n"
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |\n"
        f"{rows}\n"
    )


def _fixture(tmp_path: Path, operators: list[tuple[str, str]]):
    pipeline_path = tmp_path / "pipeline_results.json"
    pipeline = KernelTodoV2PipelineResults(chip="muxi")
    for source, definition in operators:
        pipeline.upsert_simple_opt(
            source_operator=source,
            definition_name=definition,
            result=_stage(f"simple-{definition}"),
        )
        pipeline.update_stage(
            source_operator=source,
            name=StageName.NATIVE_BASELINE,
            result=_stage(f"native-{definition}"),
        )
    write_pipeline_results(pipeline_path, pipeline)
    results_path = tmp_path / "results.md"
    results_path.write_text(
        _results([source for source, _ in operators]),
        encoding="utf-8",
    )
    native_catalog = tmp_path / "native-catalog"
    native_catalog.mkdir()
    (native_catalog / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "evaluator": "native",
                "layout": "per-operator",
            }
        ),
        encoding="utf-8",
    )
    evidence_args = []
    seed_args = []
    for source, definition in operators:
        operator_root = native_catalog / "ops" / definition
        operator_root.mkdir(parents=True)
        for name in (
            "definition.json",
            "oracle.py",
            "correctness.jsonl",
            "timing.jsonl",
        ):
            (operator_root / name).write_text("evidence\n", encoding="utf-8")
        evidence = tmp_path / "profiles" / f"{definition}.json"
        evidence.parent.mkdir(exist_ok=True)
        evidence.write_text('{"passed": true}\n', encoding="utf-8")
        evidence_args.extend(["--profile-evidence", f"{source}={evidence}"])
        baseline = tmp_path / "baselines" / f"{definition}.py"
        baseline.parent.mkdir(exist_ok=True)
        baseline.write_text(
            "def run(input):\n    return input\n",
            encoding="utf-8",
        )
        seed_args.extend(["--seed-code-path", f"{source}={baseline}"])
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    run_root = tmp_path / "run"
    argv = [
        "--chip",
        "muxi",
        "--attempt-id",
        "campaign-1",
        *evidence_args,
        *seed_args,
        "--server-url",
        "http://kgs.test",
        "--expected-backend",
        "metax",
        "--target-hardware",
        "MetaX C550",
        "--native-catalog-root",
        str(native_catalog),
        "--knowledge-catalog-path",
        str(knowledge),
        "--run-root",
        str(run_root),
        "--results-md",
        str(results_path),
        "--pipeline-json",
        str(pipeline_path),
        "--kernelgen-results-md",
        str(tmp_path / "kernelgen_results.md"),
        "--max-operators",
        "2",
        "--agents-per-operator",
        "2",
        "--epochs",
        "1",
    ]
    return launcher._parse_args(argv)


class FakeClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.calls = 0

    def get(self, path: str, *, timeout: float):
        assert path == "/status"
        self.calls += 1
        return {
            "api_version": "v6.2",
            "backend": "metax",
            "scheduler": {
                "device_slots": 2,
                "healthy": 2,
                "available": 2,
                "active": 0,
                "waiting": 0,
                "checking": 0,
                "broken": 0,
            },
        }


def test_dry_run_selects_pending_second_stage_and_builds_smoke_then_campaign(
    tmp_path: Path,
):
    args = _fixture(tmp_path, [("gelu", "gelu"), ("foo.out", "foo_out")])
    args.dry_run = True

    exit_code, manifest = launcher.run(
        args,
        client_factory=lambda url: (_ for _ in ()).throw(AssertionError()),
    )

    assert exit_code == 0
    assert manifest["smoke_operator"] == "gelu"
    assert [item["definition_name"] for item in manifest["operators"]] == [
        "foo_out",
        "gelu",
    ]
    smoke = manifest["smoke_command"]
    campaign = manifest["campaign_command"]
    assert smoke[smoke.index("--max-round") + 1] == "1"
    assert smoke[smoke.index("--agents-per-operator") + 1] == "1"
    assert smoke[smoke.index("--knowledge-mode") + 1] == "read_only_v1"
    assert campaign[campaign.index("--knowledge-mode") + 1] == "read_write_v1"
    assert "--no-profile" not in smoke + campaign
    assert smoke.count("--seed-code-path") == 1
    assert campaign.count("--seed-code-path") == 2
    assert all("seed_code_path" in item for item in manifest["operators"])
    assert any(
        item.startswith("gelu=")
        for item in (
            smoke[index + 1]
            for index, item in enumerate(smoke)
            if item == "--seed-code-path"
        )
    )
    assert json.loads(
        (args.run_root / "campaign_manifest.json").read_text(encoding="utf-8")
    ) == manifest


def test_missing_profile_evidence_blocks_before_launch(tmp_path: Path):
    args = _fixture(tmp_path, [("gelu", "gelu")])
    args.profile_evidence = []

    with pytest.raises(launcher.LaunchError, match="profiling evidence is missing"):
        launcher.run(args)


def test_missing_seed_code_blocks_before_launch(tmp_path: Path):
    args = _fixture(tmp_path, [("gelu", "gelu")])
    args.seed_code_path = []

    with pytest.raises(
        launcher.LaunchError,
        match="seed code is missing",
    ):
        launcher.run(args)


def test_smoke_failure_does_not_start_formal_campaign_or_update_pipeline(
    tmp_path: Path,
):
    args = _fixture(tmp_path, [("gelu", "gelu")])
    commands = []
    client = FakeClient(args.server_url)

    def runner(command, log_path):
        commands.append(command)
        return 7

    exit_code, manifest = launcher.run(
        args,
        command_runner=runner,
        client_factory=lambda url: client,
    )

    assert exit_code == 1
    assert manifest["status"] == "SMOKE_FAILED"
    assert len(commands) == 1
    assert client.calls == 2
    pipeline = load_pipeline_results(args.pipeline_json, chip="muxi")
    assert pipeline.operators["gelu"].kernelgen_native is None


def test_formal_campaign_collects_results_and_updates_all_views(tmp_path: Path):
    args = _fixture(tmp_path, [("gelu", "gelu"), ("foo.out", "foo_out")])
    commands = []
    client = FakeClient(args.server_url)
    evidence = tmp_path / "campaign-ledger.json"
    evidence.write_text("{}\n", encoding="utf-8")

    def runner(command, log_path):
        commands.append(command)
        log_path.write_text("completed\n", encoding="utf-8")
        return 0

    def collector(workspace, definition, epoch):
        if definition == "gelu":
            return launcher.CollectedResult(
                definition_name=definition,
                complete=True,
                geo_mean=1.25,
                best_code="def run(input):\n    return input\n",
                evidence=[evidence],
            )
        return launcher.CollectedResult(
            definition_name=definition,
            complete=True,
            geo_mean=None,
            best_code="",
            evidence=[evidence],
            reason="no correct timed candidate",
        )

    exit_code, manifest = launcher.run(
        args,
        command_runner=runner,
        client_factory=lambda url: client,
        result_collector=collector,
    )

    assert exit_code == 1
    assert manifest["status"] == "COMPLETED_WITH_FAILURES"
    assert len(commands) == 2
    assert client.calls == 3
    pipeline = load_pipeline_results(args.pipeline_json, chip="muxi")
    gelu = pipeline.operators["gelu"]
    failed = pipeline.operators["foo.out"]
    assert gelu.kernelgen_native.current.geo_mean == 1.25
    assert gelu.status() == PipelineStatus.SECOND_STAGE_RUNNING
    assert failed.kernelgen_native.current.verdict.value == "failed"
    assert failed.status() == PipelineStatus.FAILED
    best = args.run_root / "best/gelu.py"
    assert best.read_text(encoding="utf-8").startswith("def run")
    results = args.results_md.read_text(encoding="utf-8")
    assert "| `gelu` | 通过 | 通过 | 二阶段处理中 |" in results
    assert "| `foo.out` | 通过 | 通过 | 失败 |" in results
    assert "Native best code" in results
    assert "KernelGen Native 加速比" in args.kernelgen_results_md.read_text(
        encoding="utf-8"
    )
    report = json.loads(
        (args.run_root / "campaign_results.json").read_text(encoding="utf-8")
    )
    assert [item["verdict"] for item in report["results"]] == [
        "failed",
        "passed",
    ]


def test_incomplete_operator_is_recorded_as_blocked(tmp_path: Path):
    args = _fixture(tmp_path, [("gelu", "gelu")])
    client = FakeClient(args.server_url)

    def runner(command, log_path):
        log_path.write_text("incomplete\n", encoding="utf-8")
        return 0

    def collector(workspace, definition, epoch):
        return launcher.CollectedResult(
            definition_name=definition,
            complete=False,
            geo_mean=None,
            best_code="",
            evidence=[],
            reason="final epoch manifest is missing",
        )

    exit_code, _ = launcher.run(
        args,
        command_runner=runner,
        client_factory=lambda url: client,
        result_collector=collector,
    )

    assert exit_code == 1
    pipeline = load_pipeline_results(args.pipeline_json, chip="muxi")
    record = pipeline.operators["gelu"]
    assert record.kernelgen_native.current.verdict.value == "blocked"
    assert record.status() == PipelineStatus.BLOCKED
    assert "| `gelu` | 通过 | 通过 | 阻塞 |" in args.results_md.read_text(
        encoding="utf-8"
    )


def test_started_run_root_is_not_overwritten(tmp_path: Path):
    args = _fixture(tmp_path, [("gelu", "gelu")])
    args.run_root.mkdir()
    (args.run_root / "campaign_manifest.json").write_text(
        json.dumps({"dry_run": False, "status": "SMOKE_FAILED"}),
        encoding="utf-8",
    )

    with pytest.raises(launcher.LaunchError, match="already contains a started"):
        launcher.run(args)

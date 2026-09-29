"""Workflow postconditions for durable Coder ledger state."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.framework import FakeRuntime
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow
from kernelgen.tests.helpers import experiment_plan, round_conclusion


_DEFINITION = {
    "name": "abl_t1_gelu",
    "op_type": "elementwise",
    "axes": {},
    "inputs": {"input_0": {"shape": [128], "dtype": "float32"}},
    "outputs": {"output": {"shape": [128], "dtype": "float32"}},
    "reference": "def run(input_0): return input_0",
}

_INPUT = {
    "definition": _DEFINITION,
    "target_hardware": "A100",
    "profile_enabled": True,
}


def _eval():
    return {
        "status": "PASSED",
        "geo_mean": 1.1,
        "min_speedup": 1.1,
        "worst_workload_uuid": "u0",
        "latency_ms": 0.1,
        "abs_err": 0.0,
        "rel_err": 0.0,
        "per_workload": [{"uuid": "u0", "status": "PASSED", "speedup": 1.1}],
    }


class _TestWorkflow(SingleCoderOptimizationWorkflow):
    def _resolve_target_context(self, inp):
        from kernelgen.data.target_context import build_target_context

        return build_target_context(
            target_hardware=inp.target_hardware,
            implementation_language=inp.implementation_language.value,
            service_status={
                "status": "ok",
                "backend": "cuda",
                "target": {
                    "backend": "cuda",
                    "vendor": "nvidia",
                    "architecture": "ampere",
                    "device": "NVIDIA A100-SXM4-40GB",
                },
                "software": {"language": "triton"},
            },
        )


def _record_profile_fixture(workspace, profiled_rounds, round_num):
    profiled_rounds.append(round_num)
    analysis = (
        workspace
        / ".kernelgen"
        / "profile-analysis"
        / f"round-{round_num:04d}.json"
    )
    analysis.parent.mkdir(parents=True, exist_ok=True)
    analysis.write_text(
        '{"status":"failed","error":"fixture"}',
        encoding="utf-8",
    )
    Ledger(workspace).attach_profile_analysis(
        round_num,
        status="failed",
        analysis_path=str(analysis.relative_to(workspace)),
        summary="failed: fixture",
    )


def _run(workspace: Path):
    runtime = FakeRuntime(['{"status":"PASSED","summary":"done"}'])
    return _TestWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)


@pytest.fixture(autouse=True)
def _disable_distillation(monkeypatch):
    from kernelgen.workflows.optimization.single_coder import workflow

    monkeypatch.setattr(
        workflow,
        "run_distillation",
        lambda workspace, inp, runtime: None,
    )


def test_workflow_profiles_final_best_before_distill(tmp_path, monkeypatch):
    from kernelgen.workflows.optimization.single_coder import profiling

    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        profile_enabled=True,
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(
        1,
        round_conclusion(1),
        StopConfig(max_round=1),
    )
    runtime = FakeRuntime(['{"status":"PASSED","summary":"done"}'])
    profiled_rounds = []
    monkeypatch.setattr(
        profiling,
        "_run_profile_analyzer",
        lambda runtime, round_num, *, knowledge_enabled: (
            _record_profile_fixture(tmp_path, profiled_rounds, round_num)
        ),
    )
    workflow = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    )

    result = workflow.run(_INPUT, None)

    assert result.status == "PASSED"
    assert profiled_rounds == [1]
    assert Ledger(tmp_path).get_round(1).profile.status == "failed"


def test_workflow_profiles_latest_best_not_older_pending_round(
    tmp_path,
    monkeypatch,
):
    from kernelgen.workflows.optimization.single_coder import profiling

    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "first-best",
        experiment_plan(1),
        profile_enabled=True,
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(1, round_conclusion(1))
    improved = _eval()
    improved["geo_mean"] = 1.2
    ledger.record_eval(
        improved,
        "final-best",
        experiment_plan(2),
        profile_enabled=True,
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(
        2,
        round_conclusion(2),
        StopConfig(max_round=2),
    )
    runtime = FakeRuntime(['{"status":"PASSED","summary":"done"}'])
    profiled_rounds = []
    monkeypatch.setattr(
        profiling,
        "_run_profile_analyzer",
        lambda runtime, round_num, *, knowledge_enabled: (
            _record_profile_fixture(tmp_path, profiled_rounds, round_num)
        ),
    )
    workflow = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    )

    result = workflow.run(_INPUT, None)

    reloaded = Ledger(tmp_path)
    assert result.status == "PASSED"
    assert profiled_rounds == [2]
    assert reloaded.get_round(1).profile.status == "pending"
    assert reloaded.get_round(2).profile.status == "failed"


def test_workflow_records_profile_failure_without_failing_run(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        profile_enabled=True,
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    snapshot = tmp_path / ".kernelgen" / "evals" / "round-0001"
    snapshot.mkdir(parents=True)
    (snapshot / "identity.json").write_text(
        json.dumps(
                {
                    "evaluation_fingerprint": "workflow-fingerprint",
                    "solution_sha256": "workflow-solution-sha",
                    "server_backend": "cuda",
                    "workload_uuids": ["u0"],
                    "workload_sha256": ["workflow-workload-sha"],
                    "profile_workload_uuids": ["u0"],
                }
        ),
        encoding="utf-8",
    )
    for filename, payload in (
        ("result.json", _eval()),
        ("solution.json", {}),
        ("definition.json", {}),
        ("workloads.json", []),
    ):
        (snapshot / filename).write_text(json.dumps(payload), encoding="utf-8")
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="workflow-fingerprint",
        snapshot_path=str(snapshot.relative_to(tmp_path)),
    )
    ledger.finalize_round(
        1,
        round_conclusion(1),
        StopConfig(max_round=1),
    )

    result = _run(tmp_path)

    profile = Ledger(tmp_path).get_round(1).profile
    assert result.status == "PASSED"
    assert profile.status == "failed"
    analysis = json.loads((tmp_path / profile.analysis_path).read_text())
    assert "profile analyzer role" in analysis["error"]


def test_workflow_rejects_missing_conclusion(tmp_path):
    Ledger(tmp_path).record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    try:
        _run(tmp_path)
    except RuntimeError as exc:
        assert "finalize_round" in str(exc)
    else:
        raise AssertionError("workflow must reject a Coder that skips finalize_round")


def test_workflow_recovers_pending_conclusion_in_same_session(tmp_path):
    Ledger(tmp_path).record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )

    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "pending-session"
            return '{"status":"PASSED","summary":"returned too early"}'

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            ledger = Ledger(tmp_path)
            ledger.finalize_round(
                1,
                round_conclusion(1),
                StopConfig(max_round=1),
            )
            return '{"status":"PASSED","summary":"finalized"}'

    runtime = _ResumableRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert [kind for kind, _ in runtime.calls] == ["invoke", "resume"]
    recovery_prompt = runtime.calls[1][1]
    assert "finalize_round for exactly round 1" in recovery_prompt
    assert "Do not use Python or direct file edits" in recovery_prompt
    assert Ledger(tmp_path).stopped_round().round_num == 1


def test_workflow_resumes_persisted_pending_session_after_restart(tmp_path):
    Ledger(tmp_path).record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )

    class _RestoredRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def restore_session(self):
            self.calls.append(("restore", None))
            self.last_session_id = "persisted-pending-session"
            return self.last_session_id

        def invoke(self, prompt, *, model="inherit", agent=None):
            raise AssertionError("restart must not create a new provider session")

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            assert self.last_session_id == "persisted-pending-session"
            Ledger(tmp_path).finalize_round(
                1,
                round_conclusion(1),
                StopConfig(max_round=1),
            )
            return '{"status":"PASSED","summary":"finalized after restart"}'

    runtime = _RestoredRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert [kind for kind, _ in runtime.calls] == ["restore", "resume"]
    assert "finalize_round for exactly round 1" in runtime.calls[1][1]


def test_workflow_resumes_persisted_continue_session_after_restart(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "round 1 code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(1, round_conclusion(1))
    assert ledger.get_round(1).next_verdict.should_continue

    class _RestoredRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def restore_session(self):
            self.calls.append(("restore", None))
            self.last_session_id = "persisted-continue-session"
            return self.last_session_id

        def invoke(self, prompt, *, model="inherit", agent=None):
            raise AssertionError("restart must not create a new provider session")

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            assert self.last_session_id == "persisted-continue-session"
            improved = _eval()
            improved["geo_mean"] = 1.2
            resumed = Ledger(tmp_path)
            resumed.record_eval(
                improved,
                "round 2 code",
                experiment_plan(2),
                definition_name="abl_t1_gelu",
                target_hardware="A100",
            )
            resumed.finalize_round(
                2,
                round_conclusion(2),
                StopConfig(max_round=2),
            )
            return '{"status":"PASSED","summary":"continued after restart"}'

    runtime = _RestoredRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert result.best_geo_mean == 1.2
    assert [kind for kind, _ in runtime.calls] == ["restore", "resume"]
    assert "verdict for round 1 is CONTINUE" in runtime.calls[1][1]


def test_workflow_advances_recovery_to_a_new_pending_round(tmp_path):
    Ledger(tmp_path).record_eval(
        _eval(),
        "round 1 code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )

    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "advancing-pending-session"
            return '{"status":"PASSED","summary":"returned before round 1 conclusion"}'

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            ledger = Ledger(tmp_path)
            if len(self.calls) == 2:
                assert "finalize_round for exactly round 1" in prompt
                ledger.finalize_round(1, round_conclusion(1))
                improved = _eval()
                improved["geo_mean"] = 1.2
                ledger.record_eval(
                    improved,
                    "round 2 code",
                    experiment_plan(2),
                    definition_name="abl_t1_gelu",
                    target_hardware="A100",
                )
                return (
                    '{"status":"PASSED",'
                    '"summary":"continued but returned before round 2 conclusion"}'
                )

            assert "finalize_round for exactly round 2" in prompt
            ledger.finalize_round(
                2,
                round_conclusion(2),
                StopConfig(max_round=2),
            )
            return '{"status":"PASSED","summary":"finalized round 2"}'

    runtime = _ResumableRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert result.rounds == 2
    assert result.best_geo_mean == 1.2
    assert [kind for kind, _ in runtime.calls] == [
        "invoke",
        "resume",
        "resume",
    ]
    assert Ledger(tmp_path).stopped_round().round_num == 2


def test_workflow_retries_pending_conclusion_until_it_is_finalized(tmp_path):
    Ledger(tmp_path).record_eval(
        _eval(),
        "round 1 code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )

    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "retry-pending-session"
            return "I still need to finalize the measured round."

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            assert "finalize_round for exactly round 1" in prompt
            if len(self.calls) == 2:
                return "I will finalize it next."
            Ledger(tmp_path).finalize_round(
                1,
                round_conclusion(1),
                StopConfig(max_round=1),
            )
            return '{"status":"PASSED","summary":"finalized after retry"}'

    runtime = _ResumableRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert [kind for kind, _ in runtime.calls] == [
        "invoke",
        "resume",
        "resume",
    ]
    assert Ledger(tmp_path).stopped_round().round_num == 1


@pytest.mark.parametrize(
    "early_output",
    [
        "Let me read the relevant knowledge before building the baseline.",
        '{"status":"NO_MEASURED_ROUNDS","summary":"returned too early"}',
    ],
)
def test_workflow_resumes_zero_round_early_return_in_same_session(
    tmp_path,
    early_output,
):
    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "zero-round-session"
            return early_output

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            assert "zero measured rounds" in prompt
            assert "same conversation and workspace" in prompt
            assert "PREVIOUS OUTPUT FAILED VALIDATION" not in prompt
            assert "Re-emit the FULL corrected JSON object" not in prompt
            ledger = Ledger(tmp_path)
            ledger.record_eval(
                _eval(),
                "baseline code",
                experiment_plan(1),
                definition_name="abl_t1_gelu",
                target_hardware="A100",
            )
            ledger.finalize_round(
                1,
                round_conclusion(1),
                StopConfig(max_round=1),
            )
            return '{"status":"PASSED","summary":"continued to stop"}'

    runtime = _ResumableRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert result.rounds == 1
    assert [kind for kind, _ in runtime.calls] == ["invoke", "resume"]


def test_workflow_fails_loudly_after_zero_round_recovery_budget(tmp_path):
    class _AlwaysPrematureRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "always-premature-session"
            return "I will inspect the candidate next."

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            return "I still need to inspect the candidate."

    runtime = _AlwaysPrematureRuntime()
    with pytest.raises(
        RuntimeError,
        match="no Coder invocation remains for same-session recovery",
    ):
        _TestWorkflow(
            cwd=str(tmp_path),
            runtime_factory=lambda _: runtime,
        ).run({**_INPUT, "max_coder_sessions": 2}, None)

    assert [kind for kind, _ in runtime.calls] == ["invoke", "resume"]
    assert "PREVIOUS OUTPUT FAILED VALIDATION" not in runtime.calls[1][1]


def test_workflow_reuses_terminal_ledger_without_invoking_coder(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(
        1,
        round_conclusion(1),
        StopConfig(max_round=1),
    )
    runtime = FakeRuntime([])

    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert result.best_geo_mean == 1.1
    assert "authoritative Coder ledger" in result.summary
    assert runtime.calls == []


def test_workflow_reuses_terminal_ledger_without_valid_best(tmp_path):
    ledger = Ledger(tmp_path)
    failed = _eval()
    failed.update({"status": "PARTIAL_PASS", "geo_mean": None})
    ledger.record_eval(
        failed,
        "incorrect code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(
        1,
        round_conclusion(1),
        StopConfig(max_round=1),
    )
    runtime = FakeRuntime([])

    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "FAILED"
    assert result.best_geo_mean is None
    assert runtime.calls == []


def test_workflow_continues_after_malformed_report_with_continue_verdict(
    tmp_path,
):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "round 1 code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(1, round_conclusion(1))

    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "continue-after-contract-error"
            return "not json"

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            if "authoritative finalize_round verdict" not in prompt:
                return "still not json"
            resumed = Ledger(tmp_path)
            improved = _eval()
            improved["geo_mean"] = 1.2
            resumed.record_eval(
                improved,
                "round 2 code",
                experiment_plan(2),
                definition_name="abl_t1_gelu",
                target_hardware="A100",
            )
            resumed.finalize_round(
                2,
                round_conclusion(2),
                StopConfig(max_round=2),
            )
            return '{"status":"PASSED","summary":"continued to stop"}'

    runtime = _ResumableRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert result.best_geo_mean == 1.2
    assert [kind for kind, _ in runtime.calls] == ["invoke", "resume"]
    assert "authoritative finalize_round verdict" in runtime.calls[-1][1]


def test_workflow_retries_unchanged_continue_round_before_session_limit(
    tmp_path,
):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "round 1 code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(1, round_conclusion(1))
    improved = _eval()
    improved["geo_mean"] = 1.2
    ledger.record_eval(
        improved,
        "round 2 code",
        experiment_plan(2),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(2, round_conclusion(2))
    assert ledger.get_round(2).next_verdict.should_continue

    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "retry-continue-session"
            return "I will start the next optimization round."

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            assert "authoritative finalize_round verdict for round 2" in prompt
            if len(self.calls) == 2:
                return "I am about to evaluate the next candidate."
            final_eval = _eval()
            final_eval["geo_mean"] = 1.3
            resumed = Ledger(tmp_path)
            resumed.record_eval(
                final_eval,
                "round 3 code",
                experiment_plan(3),
                definition_name="abl_t1_gelu",
                target_hardware="A100",
            )
            resumed.finalize_round(
                3,
                round_conclusion(3),
                StopConfig(max_round=3),
            )
            return '{"status":"PASSED","summary":"continued to round 3"}'

    runtime = _ResumableRuntime()
    result = _TestWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: runtime,
    ).run(_INPUT, None)

    assert result.status == "PASSED"
    assert result.rounds == 3
    assert result.best_geo_mean == 1.3
    assert [kind for kind, _ in runtime.calls] == [
        "invoke",
        "resume",
        "resume",
    ]
    assert Ledger(tmp_path).stopped_round().round_num == 3


def test_workflow_accepts_terminal_round(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    ledger.finalize_round(
        1,
        round_conclusion(1),
        StopConfig(max_round=1),
    )
    result = _run(tmp_path)
    assert result.status == "PASSED"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            import traceback

            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)

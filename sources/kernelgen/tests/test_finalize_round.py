"""Unit tests for the post-measurement finalize_round tool."""

import contextlib
import io
import json

from kernelgen.data.ledger import Ledger
from kernelgen.data.round_conclusion import RoundConclusion
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.tools.finalize_round import main


def _eval(geo=1.2):
    return {
        "api_version": "v5.1",
        "status": "PASSED",
        "geo_mean": geo,
        "min_speedup": geo,
        "worst_workload_uuid": "u0",
        "latency_ms": 0.1,
        "abs_err": 0.0,
        "rel_err": 0.0,
        "num_workloads": 1,
        "num_passed": 1,
        "per_workload": [
            {
                "uuid": "u0",
                "axes": {},
                "status": "PASSED",
                "speedup": geo,
                "latency_ms": 0.1,
                "reference_latency_ms": 0.12,
            }
        ],
    }


def _concept_use(concept_ref="kg:method:two-stage-reduction"):
    return {
        "concept_ref": concept_ref,
        "query_event_id": "query:" + "a" * 32,
        "role": "implementation",
        "disposition": "adopted",
        "application_note": "implemented the two-stage reduction",
        "affected_parts": ["kernel reduction"],
    }


def _run(argv):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main(argv)
    return code, json.loads(output.getvalue())


def test_round_conclusion_schema_forbids_measurement_fields():
    model = RoundConclusion.model_validate(round_conclusion(1))
    assert model.optimization_level == "L2_memory"
    try:
        RoundConclusion.model_validate({**round_conclusion(1), "geo_mean": 999.0})
    except ValueError as exc:
        assert "geo_mean" in str(exc)
    else:
        raise AssertionError("conclusion must reject authoritative measurement fields")


def test_round_conclusion_discards_legacy_placeholders():
    model = RoundConclusion.model_validate(
        {
            **round_conclusion(1),
            "architecture_tag": "legacy",
            "suggestion_followed": "yes",
            "key_numbers": {"tile": 128},
            "composable": True,
            "composition_group": "tiling",
            "diagnostic_results": "legacy",
        }
    )

    assert set(model.model_dump()) == {
        "round_num",
        "expectation_status",
        "root_cause",
        "perf_gap_analysis",
        "next_suggestion",
        "optimization_level",
        "debug_lesson",
        "strategy_evolution",
        "knowledge_assessments",
    }


def test_finalize_round_happy(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "code_v1", experiment_plan(1))
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(round_conclusion(1)))
    code, output = _run(["--ledger-dir", str(tmp_path), str(conclusion_path)])
    assert code == 0
    assert output["recorded"] is True
    assert output["conclusion_recorded"] is True
    assert output["round_finalized"] is True
    assert output["status"] == "CONTINUE"
    assert output["continue"] is True
    assert output["candidate_action"] == "KEEP"
    assert output["candidate_path"] == "tmp/main.py"
    assert output["instruction"].startswith(
        "KEEP applied: tmp/main.py was atomically overwritten with the "
        "evaluated round 1 solution."
    )
    assert "Any edits made after eval_round for round 1 were discarded." in output[
        "instruction"
    ]
    assert "CONTINUE optimizing." in output["instruction"]
    assert (tmp_path / "tmp" / "main.py").read_text() == "code_v1"
    record = Ledger(tmp_path).get_round(1)
    assert record.conclusion.root_cause == "cause 1"
    assert record.next_verdict.should_continue is True
    assert record.profile.status == "not_required"
    assert record.evaluation.geo_mean == 1.2


def test_finalize_round_turns_continue_into_cooperative_cancel(tmp_path):
    Ledger(tmp_path).record_eval(_eval(), "code", experiment_plan(1))
    control = WorkspaceRunControl(tmp_path)
    control.request_cancel("requested from operator console")
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(round_conclusion(1)))

    code, output = _run(
        ["--ledger-dir", str(tmp_path), str(conclusion_path)]
    )

    assert code == 0
    assert output["status"] == "RUN_STOPPED"
    assert output["continue"] is False
    assert output["cancel_requested"] is True
    assert output["reason_code"] == "user_cancelled"
    assert output["reason"] == "requested from operator console"
    record = Ledger(tmp_path).get_round(1)
    assert record.conclusion is not None
    assert record.next_verdict is not None
    assert record.next_verdict.code == "user_cancelled"
    assert control.progress().stage == "CANCELLING"
    assert any(
        event.event_type == "CANCEL_SAFE_POINT_REACHED"
        for event in control.read_events()
    )


def test_finalize_round_requires_assessment_for_each_applied_knowledge(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(1.0), "baseline", experiment_plan(1))
    ledger.finalize_round(1, round_conclusion(1))

    concept_ref = "kg:method:two-stage-reduction"
    ledger.record_eval(
        _eval(),
        "code_v1",
        experiment_plan(2, knowledge_uses=[_concept_use(concept_ref)]),
    )
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(round_conclusion(2)))

    code, output = _run(
        ["--ledger-dir", str(tmp_path), str(conclusion_path)]
    )
    assert code == 2
    assert "knowledge_assessments must match" in output["error"]

    conclusion_path.write_text(
        json.dumps(
            round_conclusion(
                2,
                knowledge_assessments=[
                    {
                        "concept_ref": "kg:method:unrelated",
                        "assessment": "confirmed",
                        "rationale": "unrelated knowledge",
                    }
                ],
            )
        )
    )
    code, output = _run(
        ["--ledger-dir", str(tmp_path), str(conclusion_path)]
    )
    assert code == 2
    assert "knowledge_assessments must match" in output["error"]

    conclusion_path.write_text(
        json.dumps(
            round_conclusion(
                2,
                knowledge_assessments=[
                    {
                        "concept_ref": concept_ref,
                        "assessment": "confirmed",
                        "rationale": "the measured reduction latency improved",
                    }
                ],
            )
        )
    )
    code, output = _run(
        ["--ledger-dir", str(tmp_path), str(conclusion_path)]
    )
    assert code == 0
    assessment = Ledger(tmp_path).get_round(2).conclusion.knowledge_assessments[0]
    assert assessment.assessment == "confirmed"


def test_finalize_round_replays_identical_request(tmp_path):
    Ledger(tmp_path).record_eval(_eval(), "code", experiment_plan(1))
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(round_conclusion(1)))
    first_code, first = _run(["--ledger-dir", str(tmp_path), str(conclusion_path)])
    (tmp_path / "tmp" / "main.py").write_text("DRIFTED")
    second_code, second = _run(["--ledger-dir", str(tmp_path), str(conclusion_path)])
    assert first_code == 0 and first["conclusion_recorded"] is True
    assert first["idempotent_replay"] is False
    assert second_code == 0
    assert second["recorded"] is True
    assert second["idempotent_replay"] is True
    assert second["continue"] == first["continue"]
    assert second["reason_code"] == first["reason_code"]
    assert second["candidate_action"] == first["candidate_action"]
    assert (tmp_path / "tmp" / "main.py").read_text() == "code"


def test_finalize_round_rejects_conflicting_replay(tmp_path):
    Ledger(tmp_path).record_eval(_eval(), "code", experiment_plan(1))
    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps(round_conclusion(1)))
    first_code, _ = _run(["--ledger-dir", str(tmp_path), str(first_path)])
    conflicting = {
        **round_conclusion(1),
        "root_cause": "different conclusion",
    }
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps(conflicting))

    second_code, second = _run(
        ["--ledger-dir", str(tmp_path), str(second_path)]
    )

    assert first_code == 0
    assert second_code == 2
    assert second["error"] == "ROUND_CONCLUSION_CONFLICT"
    assert Ledger(tmp_path).get_round(1).conclusion.root_cause == "cause 1"


def test_finalize_round_missing_required(tmp_path):
    Ledger(tmp_path).record_eval(_eval(), "code", experiment_plan(1))
    invalid = round_conclusion(1)
    del invalid["root_cause"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid))
    code, output = _run(["--ledger-dir", str(tmp_path), str(path)])
    assert code == 2
    assert "root_cause" in output["error"]


def test_finalize_round_nonbaseline_requires_perf_gap(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "v1", experiment_plan(1))
    ledger.finalize_round(1, round_conclusion(1))
    ledger.record_eval(_eval(1.3), "v2", experiment_plan(2))
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(round_conclusion(2, perf_gap_analysis="")))
    code, output = _run(["--ledger-dir", str(tmp_path), str(path)])
    assert code == 2
    assert "perf_gap_analysis" in output["error"]


def test_finalize_round_rejects_unknown_round_and_bad_json(tmp_path):
    Ledger(tmp_path).record_eval(_eval(), "code", experiment_plan(1))
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(round_conclusion(5)))
    code, output = _run(["--ledger-dir", str(tmp_path), str(unknown)])
    assert code == 2 and "no eval round 5" in output["error"]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    code, output = _run(["--ledger-dir", str(tmp_path), str(bad)])
    assert code == 2 and "invalid JSON" in output["error"]


def test_finalize_round_returns_max_round_stop(tmp_path):
    from kernelgen.data.stop_policy import StopConfig

    ledger = Ledger(tmp_path)
    ledger.set_stop_config(StopConfig(max_round=1))
    failed = _eval(geo=None)
    failed["status"] = "RUNTIME_ERROR"
    failed["num_passed"] = 0
    failed["per_workload"][0]["status"] = "RUNTIME_ERROR"
    ledger.record_eval(failed, "bad", experiment_plan(1))
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(round_conclusion(1)))

    code, output = _run([
        "--ledger-dir",
        str(tmp_path),
        str(conclusion_path),
    ])

    assert code == 0
    assert output["status"] == "RUN_STOPPED"
    assert output["continue"] is False
    assert output["reason_code"] == "max_round_reached"
    assert output["candidate_action"] == "REPAIR"
    assert Ledger(tmp_path).stopped_round().round_num == 1


def test_finalize_round_stops_after_suspected_device_error(tmp_path):
    from kernelgen.data.stop_policy import StopConfig

    ledger = Ledger(tmp_path)
    ledger.set_stop_config(
        StopConfig(
            max_round=15,
            min_rounds=10,
            soft_stop_disabled=True,
        )
    )
    suspected = _eval(geo=None)
    suspected["status"] = "SUSPECTED_DEVICE_ERROR"
    suspected["num_passed"] = 0
    suspected["per_workload"] = []
    suspected["log"] = "same request failed on two different device slots"
    ledger.record_eval(suspected, "candidate", experiment_plan(1))
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(
        json.dumps(
            round_conclusion(
                1,
                expectation_status="not_evaluable",
            )
        )
    )

    code, output = _run([
        "--ledger-dir",
        str(tmp_path),
        str(conclusion_path),
    ])

    assert code == 0
    assert output["status"] == "RUN_STOPPED"
    assert output["continue"] is False
    assert output["reason_code"] == "suspected_device_error"
    assert "operator review" in output["reason"]
    stopped = Ledger(tmp_path).stopped_round()
    assert stopped is not None
    assert stopped.round_num == 1
    assert stopped.next_verdict.should_continue is False


def test_finalize_round_restores_best_candidate(tmp_path):
    ledger = Ledger(tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("BEST")
    ledger.record_eval(
        _eval(1.4),
        "BEST",
        experiment_plan(1),
        candidate_path="candidate.py",
    )
    ledger.finalize_round(1, round_conclusion(1))

    candidate.write_text("WORSE")
    ledger.record_eval(
        _eval(1.1),
        "WORSE",
        experiment_plan(2),
        candidate_path="candidate.py",
    )
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(round_conclusion(2)))

    code, output = _run([
        "--ledger-dir",
        str(tmp_path),
        str(conclusion_path),
    ])

    assert code == 0
    assert output["candidate_action"] == "REVERT"
    assert output["candidate_path"] == "candidate.py"
    assert candidate.read_text() == "BEST"


def test_finalize_round_reverts_hack_even_when_it_is_faster(tmp_path):
    ledger = Ledger(tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("REAL_KERNEL")
    ledger.record_eval(
        _eval(1.4),
        "REAL_KERNEL",
        experiment_plan(1),
        candidate_path="candidate.py",
    )
    ledger.finalize_round(1, round_conclusion(1))

    candidate.write_text("TORCH_FALLBACK")
    hacked = _eval(100.0) | {
        "is_hack": True,
        "hack_reason": "forbidden Torch API: main.py:1 torch.mm",
    }
    ledger.record_eval(
        hacked,
        "TORCH_FALLBACK",
        experiment_plan(2),
        candidate_path="candidate.py",
    )
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(round_conclusion(2)))

    code, output = _run([
        "--ledger-dir",
        str(tmp_path),
        str(conclusion_path),
    ])

    assert code == 0
    assert output["candidate_action"] == "REVERT"
    assert output["best_round"] == 1
    assert candidate.read_text() == "REAL_KERNEL"
    assert (tmp_path / ".best_kernel.py").read_text() == "REAL_KERNEL"


def test_record_eval_rejects_candidate_path_outside_workspace(tmp_path):
    try:
        Ledger(tmp_path).record_eval(
            _eval(),
            "code",
            experiment_plan(1),
            candidate_path="../outside.py",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("candidate_path escaped the workspace")

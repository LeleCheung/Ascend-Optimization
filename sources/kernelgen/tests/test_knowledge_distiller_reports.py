"""Tests for the per-run Distiller Markdown contract."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kernelgen.agents.knowledge_distiller import (
    DistillOutput,
    KnowledgeDistillerAgent,
)
from kernelgen.agents.knowledge_distiller.report_format import (
    render_detailed_report,
    render_experience_report,
)
from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.data.optimization_history import (
    EvaluationRecord,
    OptimizationHistory,
    RoundRecord,
    SolutionRecord,
)
from kernelgen.framework import FakeRuntime
from kernelgen.workflows.optimization.single_coder.distillation import (
    read_profile_analysis_files,
)


EXPERIENCE = """\
## Lessons

- [validated] Keep the passing configuration.
  - Evidence: R1
  - Scope: this definition
  - Action: retain the baseline

## Remaining Bottlenecks

- Dispatch overhead remains.

## Next Experiments

- Change one launch parameter; expect lower latency.
"""

DETAILED = """\
## Strategy Evolution

R1 established the first passing implementation.

## Cross-Round Analysis

### Baseline

- Rounds: R1
- Status: validated
- Analysis: the implementation passed evaluation.
- Uncertainty: performance attribution remains open.

## Failure Analysis

- No failed measured round.

## Open Questions

- Which launch parameter controls the remaining overhead?
"""

CONCEPT_BODY = """\
## Claim

Keep the measured implementation.

## Evidence

R1 directly measured the claim.

## Applicability

This definition and workload.

## Action

Retain the measured structure.

## Limits

No portability claim is made.
"""

DIAGNOSTIC_BODY = """\
## Symptom

R2 recovered the measured numerical failure.

## Likely Causes

The boundary mask did not preserve documented semantics.

## Candidate Techniques

Apply the documented mask rule.

## Diagnosis Checklist

1. Compare the failing lanes against the documented mask.

## Caveats

Confirm the rule against the exact compiler revision.
"""


def _history() -> OptimizationHistory:
    plan = ExperimentPlan.model_validate(
        {
            "kind": "baseline",
            "strategy": "establish baseline",
            "hypothesis": "the implementation is correct",
            "expected_effect": {
                "metric": "correctness",
                "direction": "establish_baseline",
                "mechanism": "implement the canonical definition",
            },
            "code_changes": "write baseline",
            "source": {"kind": "baseline"},
            "knowledge_uses": [],
        }
    )
    return OptimizationHistory(
        definition_name="abl_t1_gelu",
        target_hardware="ASCEND910B",
        implementation_language="triton",
        best_geo_mean=1.25,
        best_round=1,
        best_code="def run(): pass",
        rounds=[
            RoundRecord(
                round_num=1,
                plan=plan,
                solution=SolutionRecord(
                    sha256="a" * 64,
                    code="def run(): pass",
                ),
                evaluation=EvaluationRecord(
                    status="PASSED",
                    geo_mean=1.25,
                ),
            )
        ],
    )


def _reply(*, experience=EXPERIENCE, detailed=DETAILED, concepts=None) -> str:
    return json.dumps(
        {
            "candidate_experience": experience,
            "candidate_detailed": detailed,
            "candidate_concepts": concepts or [],
            "skip_reason": "",
        }
    )


def _concept(**updates):
    value = {
        "proposed_kind": "experience",
        "claim_key": "test.candidate",
        "title": "Keep the measured candidate",
        "summary": "The candidate is supported by one measured round.",
        "domains": ["test"],
        "retrieval": {
            "phases": ["post_evaluation"],
            "tasks": ["next_experiment"],
        },
        "body": CONCEPT_BODY,
        "observation_intents": [
            {
                "round_num": 1,
                "claim_stance": "supports",
                "rationale": "R1 measured the claim.",
            }
        ],
    }
    value.update(updates)
    return value


def test_distill_output_rejects_unstructured_markdown():
    with pytest.raises(ValidationError, match="exactly these H2 sections"):
        DistillOutput.model_validate_json(
            _reply(experience="## Free Form\n\nA lesson from R1.")
        )


def test_distill_output_accepts_long_reports():
    long_experience = EXPERIENCE.replace(
        "## Remaining Bottlenecks",
        f"{'additional evidence ' * 300}\n\n## Remaining Bottlenecks",
    )
    long_detailed = DETAILED.replace(
        "## Failure Analysis",
        f"{'additional analysis ' * 1_000}\n\n## Failure Analysis",
    )

    output = DistillOutput.model_validate_json(
        _reply(experience=long_experience, detailed=long_detailed)
    )

    assert len(output.candidate_experience) > 4_000
    assert len(output.candidate_detailed) > 16_000


def test_distiller_repairs_round_reference_outside_ledger():
    runtime = FakeRuntime(
        [
            _reply(experience=EXPERIENCE.replace("R1", "R9")),
            _reply(),
        ]
    )
    output = KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "PASSED", "geo_mean": 1.25},
                }
            ],
        },
        runtime,
    )

    assert output.candidate_experience == EXPERIENCE.strip()
    assert len(runtime.calls) == 2
    assert "absent from ledger: R9" in runtime.calls[1]["prompt"]


def test_distiller_repairs_swapped_retrieval_axes():
    runtime = FakeRuntime(
        [
            _reply(
                concepts=[
                    {
                        "proposed_kind": "experience",
                        "claim_key": "test.swapped_retrieval_axes",
                        "title": "Keep retrieval axes distinct",
                        "summary": "Known phase and task values can be repaired.",
                        "domains": ["test"],
                        "scope_hints": {"motifs": ["elementwise"]},
                        "retrieval": {
                            "phases": ["diagnosis"],
                            "tasks": ["post_error"],
                        },
                        "body": CONCEPT_BODY,
                        "observation_intents": [
                            {
                                "round_num": 1,
                                "stance": "supports",
                                "rationale": "R1 supplies the measured observation.",
                            }
                        ],
                    }
                ]
            )
        ]
    )

    output = KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "PASSED", "geo_mean": 1.25},
                }
            ],
        },
        runtime,
    )

    retrieval = output.candidate_concepts[0].retrieval
    assert retrieval.phases == ["post_error"]
    assert retrieval.tasks == ["diagnosis"]
    assert "scope" not in output.candidate_concepts[0].model_dump()
    assert output.candidate_concepts[0].scope_hints.motifs == ["elementwise"]
    assert len(runtime.calls) == 1


def test_distiller_adds_failed_status_as_symptom():
    runtime = FakeRuntime(
        [
            _reply(
                concepts=[
                    {
                        "proposed_kind": "experience",
                        "claim_key": "test.incorrect_numerical",
                        "title": "Reject the incorrect formula",
                        "summary": "The measured formula failed correctness.",
                        "domains": ["test"],
                        "scope": {"target": {"level": "exact"}},
                        "retrieval": {
                            "phases": ["post_error"],
                            "tasks": ["diagnosis"],
                        },
                        "body": CONCEPT_BODY,
                        "observation_intents": [
                            {
                                "round_num": 1,
                                "stance": "refutes",
                                "rationale": "R1 failed numerical correctness.",
                            }
                        ],
                    }
                ]
            )
        ]
    )

    output = KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "INCORRECT_NUMERICAL"},
                }
            ],
        },
        runtime,
    )

    assert output.candidate_concepts[0].retrieval.symptoms == [
        "precision_error"
    ]


def test_distiller_retries_candidate_body_with_untracked_round():
    invalid_body = CONCEPT_BODY.replace(
        "R1 directly measured the claim.",
        "R1 and R2 directly measured the claim.",
    )
    concept = {
        "proposed_kind": "experience",
        "claim_key": "test.round_provenance",
        "title": "Track candidate evidence rounds",
        "summary": "Body citations must match observation intents.",
        "domains": ["test"],
        "scope": {"target": {"level": "exact"}},
        "retrieval": {
            "phases": ["post_evaluation"],
            "tasks": ["next_experiment"],
        },
        "body": invalid_body,
        "observation_intents": [
            {
                "round_num": 1,
                "stance": "supports",
                "rationale": "R1 measured the claim.",
            }
        ],
    }
    runtime = FakeRuntime(
        [
            _reply(concepts=[concept]),
            json.dumps({**concept, "body": CONCEPT_BODY}),
        ]
    )

    output = KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "PASSED"},
                }
            ],
        },
        runtime,
    )

    assert output.candidate_concepts[0].body == CONCEPT_BODY.strip()
    assert len(runtime.calls) == 2
    assert "rounds without intents R2" in runtime.calls[1]["prompt"]
    assert "Repair exactly one KernelGen CandidateDraft" in runtime.calls[1]["prompt"]


@pytest.mark.parametrize(
    ("invalid_updates", "repaired_updates", "error_text"),
    [
        pytest.param(
            {
                "relations": [
                    {
                        "type": "relates",
                        "target": "kg:method:known-target",
                    }
                ]
            },
            {
                "relations": [
                    {
                        "type": "related",
                        "target": "kg:method:known-target",
                    }
                ]
            },
            "Input should be",
            id="relation-enum",
        ),
        pytest.param(
            {"observation_intents": []},
            {},
            "requires observation_intents or source_refs",
            id="missing-support",
        ),
        pytest.param(
            {
                "relations": [
                    {
                        "type": "related",
                        "target": "kg:method:not-retrieved",
                    }
                ]
            },
            {"relations": []},
            "not retrieved in this workspace",
            id="unavailable-relation",
        ),
    ],
)
def test_distiller_repairs_only_invalid_candidate(
    invalid_updates,
    repaired_updates,
    error_text,
):
    valid = _concept(claim_key="test.valid_candidate")
    invalid = _concept(
        claim_key="test.invalid_candidate",
        **invalid_updates,
    )
    repaired = _concept(
        claim_key="test.invalid_candidate",
        **repaired_updates,
    )
    runtime = FakeRuntime(
        [
            _reply(concepts=[valid, invalid]),
            json.dumps(repaired),
        ]
    )

    output = KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "PASSED"},
                }
            ],
            "known_concept_ids": ["kg:method:known-target"],
        },
        runtime,
    )

    assert output.candidate_experience == EXPERIENCE.strip()
    assert output.candidate_detailed == DETAILED.strip()
    assert [item.claim_key for item in output.candidate_concepts] == [
        "test.valid_candidate",
        "test.invalid_candidate",
    ]
    assert len(runtime.calls) == 2
    assert error_text in runtime.calls[1]["prompt"]
    assert "claim_stance" in runtime.calls[1]["prompt"]
    assert "relative only to the candidate Concept claim" in runtime.calls[1]["prompt"]
    assert "disproves its own optimization hypothesis" in runtime.calls[1]["prompt"]


def test_distiller_skips_only_the_candidate_that_cannot_be_repaired(capsys):
    valid = _concept(claim_key="test.valid_candidate")
    invalid = _concept(
        claim_key="test.missing_support",
        observation_intents=[],
    )
    runtime = FakeRuntime(
        [
            _reply(concepts=[valid, invalid]),
            json.dumps(invalid),
            json.dumps(invalid),
        ]
    )

    output = KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "PASSED"},
                }
            ],
        },
        runtime,
    )

    assert output.candidate_experience == EXPERIENCE.strip()
    assert output.candidate_detailed == DETAILED.strip()
    assert [item.claim_key for item in output.candidate_concepts] == [
        "test.valid_candidate"
    ]
    assert len(runtime.calls) == 3
    assert "Skipped invalid candidate 2" in capsys.readouterr().out


def test_distiller_receives_backend_native_profile_analysis_paths():
    runtime = FakeRuntime([_reply()])
    KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "PASSED", "geo_mean": 1.25},
                }
            ],
            "best_round": 1,
            "profile_analysis_files": [
                {
                    "round_num": 1,
                    "path": ".kernelgen/profile-analysis/round-0001.json",
                }
            ],
        },
        runtime,
    )

    prompt = runtime.calls[0]["prompt"]
    assert "<profile_analysis_files>" in prompt
    assert ".kernelgen/profile-analysis/round-0001.json" in prompt
    assert "use Read to inspect every file" in prompt
    assert "<profile_analyses>" not in prompt
    assert '"profiler": "msprof"' not in prompt
    assert '"findings"' not in prompt


def test_workflow_passes_profile_paths_without_analysis_payload(tmp_path):
    relative_path = ".kernelgen/profile-analysis/round-0001.json"
    analysis_path = tmp_path / relative_path
    analysis_path.parent.mkdir(parents=True)
    analysis_path.write_text(
        json.dumps({"sentinel": "must-not-enter-prompt"}),
        encoding="utf-8",
    )
    ledger = SimpleNamespace(
        history=SimpleNamespace(
            rounds=[
                SimpleNamespace(
                    round_num=1,
                    profile=SimpleNamespace(analysis_path=relative_path),
                )
            ]
        )
    )

    assert read_profile_analysis_files(tmp_path, ledger) == [
        {"round_num": 1, "path": relative_path}
    ]


def test_distiller_defines_claim_stance_relative_to_candidate_claim():
    runtime = FakeRuntime([_reply()])
    KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "PASSED", "geo_mean": 1.0},
                }
            ],
        },
        runtime,
    )

    prompt = runtime.calls[0]["prompt"]
    assert "claim_stance" in prompt
    assert "relative only to the candidate Concept claim" in prompt
    assert "disproves its own optimization hypothesis" in prompt
    contract = prompt.split("--- OUTPUT CONTRACT ---", 1)[1]
    assert "claim_stance: Literal" in contract
    assert "\n          stance:" not in contract


def test_distiller_exposes_applied_source_outcome_for_promotion():
    source_ref = {
        "resource": "source:compiler-docs",
        "revision": "abc123",
        "locator": "mask.md:L10-L20",
    }
    candidate = {
        "proposed_kind": "diagnostic",
        "claim_key": "diagnostic.mask.correctness_recovery",
        "title": "Preserve documented mask semantics",
        "summary": "The documented mask rule recovered correctness.",
        "domains": ["correctness"],
        "scope": {"target": {"level": "exact"}},
        "retrieval": {
            "phases": ["post_evaluation"],
            "tasks": ["diagnosis"],
            "symptoms": ["INCORRECT_NUMERICAL"],
        },
        "body": DIAGNOSTIC_BODY,
        "observation_intents": [
            {
                "round_num": 2,
                "stance": "supports",
                "rationale": "R2 recovered correctness after applying the rule.",
            }
        ],
        "source_refs": [source_ref],
    }
    runtime = FakeRuntime([_reply(concepts=[candidate])])

    output = KnowledgeDistillerAgent().run(
        {
            "definition_name": "abl_t1_gelu",
            "op_type": "elementwise",
            "target_hardware": "ASCEND910B",
            "rounds": [
                {
                    "round_num": 1,
                    "plan": {},
                    "solution": {"code": "def run(): pass"},
                    "evaluation": {"status": "INCORRECT_NUMERICAL"},
                },
                {
                    "round_num": 2,
                    "experiment_parent_round_num": 1,
                    "plan": {
                        "strategy": "apply documented mask semantics",
                        "knowledge_uses": [
                            {
                                "source_ref": source_ref,
                                "role": "constraint",
                                "disposition": "adopted",
                                "application_note": "mask out invalid lanes",
                                "affected_parts": ["boundary_mask"],
                            }
                        ],
                    },
                    "solution": {"code": "def run(): return 1"},
                    "evaluation": {
                        "status": "PASSED",
                        "geo_mean": 1.0,
                        "comparison": {
                            "performance_baseline_round_num": None,
                            "geo_mean_delta_pct": None,
                        },
                    },
                },
            ],
            "available_source_refs": [
                source_ref,
                {
                    "resource": "source:read-only",
                    "revision": "def456",
                    "locator": "unused.md:L1-L5",
                },
            ],
        },
        runtime,
    )

    promoted = output.candidate_concepts[0]
    assert promoted.source_refs[0].resource == "source:compiler-docs"
    assert promoted.observation_intents[0].round_num == 2
    prompt = runtime.calls[0]["prompt"]
    block = prompt.split("<source_application_results>", 1)[1].split(
        "</source_application_results>",
        1,
    )[0]
    assert '"effect": "correctness_recovered"' in block
    assert '"usage_mode": "single"' in block
    assert "source:compiler-docs" in block
    assert "source:read-only" not in block
    assert "Combined success is contextual evidence" in prompt


def test_python_injects_authoritative_ledger_metadata():
    history = _history()

    experience = render_experience_report(EXPERIENCE, history)
    detailed = render_detailed_report(DETAILED, history)

    assert experience.startswith("# Run Experience\n\n## Run Metadata")
    assert detailed.startswith("# Run Analysis\n\n## Run Metadata")
    for report in (experience, detailed):
        assert "| Definition | abl_t1_gelu |" in report
        assert "| Hardware | ASCEND910B |" in report
        assert "| Best round | R1 |" in report
        assert "| Best geo mean | 1.25x |" in report
        assert "| Final status | PASSED |" in report

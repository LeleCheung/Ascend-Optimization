import json

import pytest
from pydantic import ValidationError

from kernelgen.agents.merge import MergeAgent, MergeInput, MergeOutput
from kernelgen.framework import AgentContractError, FakeRuntime


def test_judge_merge_prompt_uses_unified_envelope_and_contract():
    runtime = FakeRuntime([])
    prompt = MergeAgent().preprocess(
        MergeInput(
            mode="judge_merge",
            candidate="new measured result",
            existing="existing result",
            label="experience",
            definition_name="attention",
            op_type="attention",
            target_hardware="H100",
        ),
        runtime,
    )

    assert "<context>" in prompt
    assert "Mode: judge_merge" in prompt
    assert "<task>" in prompt
    assert "<documents>" in prompt
    assert "<existing>" in prompt
    assert "<candidate>" in prompt
    assert "<merge_instruction>" in prompt
    assert "--- OUTPUT CONTRACT ---" in prompt
    assert "verdict: Literal" in prompt
    assert "You are a senior GPU optimization knowledge curator" in prompt


def test_n_way_prompt_uses_same_envelope():
    runtime = FakeRuntime([])
    prompt = MergeAgent().preprocess(
        MergeInput(
            mode="n_way_merge",
            texts=["first", "second"],
            merge_instruction="Preserve workload conditions.",
            label="experience",
        ),
        runtime,
    )

    assert "Mode: n_way_merge" in prompt
    assert '<document index="1">' in prompt
    assert '<document index="2">' in prompt
    assert "Preserve workload conditions." in prompt
    assert "--- OUTPUT CONTRACT ---" in prompt


@pytest.mark.parametrize(
    "payload",
    [
        {"verdict": "OTHER", "merged_text": ""},
        {"verdict": "MERGE", "merged_text": ""},
        {"verdict": "KEEP", "merged_text": "unexpected"},
        {"verdict": "DISCARD", "merged_text": "unexpected"},
    ],
)
def test_merge_output_rejects_invalid_verdict_payload(payload):
    with pytest.raises(ValidationError):
        MergeOutput.model_validate(payload)


def test_merge_input_rejects_mode_mismatch():
    with pytest.raises(ValidationError):
        MergeInput(mode="judge_merge", candidate="", existing="existing")
    with pytest.raises(ValidationError):
        MergeInput(mode="n_way_merge", texts=[], merge_instruction="merge")
    with pytest.raises(ValidationError):
        MergeInput(mode="n_way_merge", texts=["one"], candidate="unexpected")


def test_n_way_merge_retries_non_merge_verdict():
    runtime = FakeRuntime(
        [
            json.dumps({"verdict": "KEEP", "merged_text": ""}),
            json.dumps({"verdict": "MERGE", "merged_text": "combined"}),
        ]
    )
    agent = MergeAgent()
    agent._runtime = runtime

    result = agent.run(
        {
            "mode": "n_way_merge",
            "texts": ["one", "two"],
            "merge_instruction": "Combine them.",
        }
    )

    assert result.verdict == "MERGE"
    assert len(runtime.calls) == 2
    assert "n_way_merge requires verdict='MERGE'" in runtime.calls[1]["prompt"]


def test_judge_merge_fails_after_invalid_outputs():
    runtime = FakeRuntime(
        [
            json.dumps({"verdict": "KEEP", "merged_text": "unexpected"}),
            json.dumps({"verdict": "KEEP", "merged_text": "unexpected"}),
            json.dumps({"verdict": "KEEP", "merged_text": "unexpected"}),
        ]
    )
    agent = MergeAgent()
    agent._runtime = runtime

    with pytest.raises(AgentContractError):
        agent.run({"mode": "judge_merge", "candidate": "candidate"})
    assert len(runtime.calls) == 3

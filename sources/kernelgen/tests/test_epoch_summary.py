from kernelgen.agents.epoch_summary import (
    EpochSummaryAgent,
    EpochSummaryInput,
    EpochSummaryOutput,
)
from kernelgen.framework import FakeRuntime


def test_epoch_summary_prompt_contains_one_best_kernel_and_bounds_experience():
    prompt = EpochSummaryAgent().preprocess(
        EpochSummaryInput(
            definition_name="decode_mla",
            op_type="attention",
            target_hardware="Ascend910B",
            n_agents=2,
            fixed_best_geo=79.72,
            fixed_best_agent="agent1",
            fixed_best_round=18,
            fixed_best_kernel="BEST_KERNEL_SENTINEL",
            agent_results=[
                {
                    "agent_id": "agent0",
                    "status": "PASSED",
                    "best_geo": 70.0,
                    "strategy": "agent zero",
                    "trajectory": "R1 | PASSED | 70x",
                    "new_experience": "A" * 5000 + "EXPERIENCE_END",
                },
                {
                    "agent_id": "agent1",
                    "status": "PASSED",
                    "best_geo": 79.72,
                    "strategy": "agent one",
                    "trajectory": "R1 | PASSED | 79.72x",
                    "new_experience": "short lesson",
                },
            ],
        ),
        FakeRuntime([]),
    )

    assert prompt.count("BEST_KERNEL_SENTINEL") == 1
    assert prompt.count("<authoritative_best_kernel>") == 1
    assert "agent=agent1, round=18" in prompt
    assert "EXPERIENCE_END" not in prompt
    assert "[distilled experience truncated to synthesis budget]" in prompt


def test_epoch_summary_output_does_not_reselect_authoritative_best():
    properties = EpochSummaryOutput.model_json_schema()["properties"]

    assert "selected_author" not in properties
    assert "selected_agent" not in properties
    assert set(properties) == {"next_directions", "synthesis_report"}

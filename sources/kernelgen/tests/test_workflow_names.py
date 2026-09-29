"""Source names change; existing run identifiers and serialized contracts do not."""

import json

import pytest

from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.workflows.optimization.single_coder import (
    KERNEL_OPTIMIZATION_OUTPUT_FILENAME, SingleCoderOptimizationOutput, SingleCoderOptimizationWorkflow,
)
from kernelgen.workflows.operator_development import (
    DummyWorkflowCall, OperatorDevelopmentWorkflow, campaign_status, run_campaign,
)


def test_optimizer_keeps_persisted_identity_and_output_filename(tmp_path):
    assert SingleCoderOptimizationWorkflow.name == "optimize_definition"
    assert KERNEL_OPTIMIZATION_OUTPUT_FILENAME == "optimize_definition_output.json"
    result = SingleCoderOptimizationOutput.model_validate({
        "definition_name": "identity", "status": "PASSED", "best_code": "def run(x): return x",
        "best_geo_mean": 1.2, "workspace": str(tmp_path), "rounds": 1,
    })
    workflow = SingleCoderOptimizationWorkflow(cwd=str(tmp_path))
    workflow._save_output(result)
    assert json.loads((tmp_path / "optimize_definition_output.json").read_text()) == result.model_dump(mode="json")
    assert not (tmp_path / "kernel_optimization_output.json").exists()


def test_single_coder_public_name_and_shared_callers(tmp_path):
    from kernelgen.workflows.optimization import single_coder
    from kernelgen.workflows.legacy.simple_opt import workflow as simple_opt
    from kernelgen.workflows.optimization.kernelgen import epoch
    from kernelgen.workflows import extract_opt

    assert single_coder.SingleCoderOptimizationWorkflow is SingleCoderOptimizationWorkflow
    assert SingleCoderOptimizationWorkflow.__name__ == "SingleCoderOptimizationWorkflow"
    assert simple_opt.SingleCoderOptimizationWorkflow is SingleCoderOptimizationWorkflow
    assert epoch.SingleCoderOptimizationWorkflow is SingleCoderOptimizationWorkflow
    assert extract_opt.SingleCoderOptimizationWorkflow is SingleCoderOptimizationWorkflow
    assert isinstance(SingleCoderOptimizationWorkflow.bind(str(tmp_path), None), SingleCoderOptimizationWorkflow)


@pytest.mark.parametrize("initial,expected", [
    ("complete", []),
    ("waiting", ["code_review", "local_ci", "pr_submit", "pr_review_fix", "pr_followup"]),
    ("cancelled", ["local_ci", "pr_submit", "pr_review_fix", "pr_followup"]),
])
def test_development_resume_preserves_plan_receipts_and_cancel_generation(tmp_path, initial, expected):
    runner = DummyWorkflowCall(
        wait_at="code_review" if initial == "waiting" else None,
        cancel_at="code_review" if initial == "cancelled" else None,
    )
    inp = {"operator": "add", "dummy": True}
    OperatorDevelopmentWorkflow(cwd=tmp_path, workflow_call=runner).run(inp)
    assert OperatorDevelopmentWorkflow.name == "operator_lifecycle"
    plan = tmp_path / ".kernelgen/operator-lifecycle.json"
    assert json.loads(plan.read_text()) == {
        "schema_version": "1.1", "operator": "add", "simulated": True,
        "stages": ["pytest_generate", "pytest_review", "optimize", "code_review", "local_ci", "pr_submit", "pr_review_fix", "pr_followup"],
        "optimize": {"mode": "simple_opt"},
    }
    frozen = {p: p.read_bytes() for p in [plan, *tmp_path.glob("stages/*/attempts/*/result.json")]}
    control = WorkspaceRunControl(tmp_path)
    generation = control.cancellation_state().generation
    calls = []

    def resume(ctx):
        calls.append(ctx.stage)
        return DummyWorkflowCall()(ctx)

    result = OperatorDevelopmentWorkflow(cwd=tmp_path, workflow_call=resume).run({**inp, "resume": True})
    assert result.state == "SUCCEEDED"
    assert calls == expected
    assert all(p.read_bytes() == value for p, value in frozen.items())
    assert control.cancellation_state().generation == generation + (initial == "cancelled")
    assert not (tmp_path / ".kernelgen/lifecycle-owner.json").exists()


def test_campaign_kind_remains_readable_by_existing_consumers(tmp_path):
    run_campaign(tmp_path, ["add"], dummy=True)
    assert campaign_status(tmp_path)["kind"] == "operator_lifecycle_campaign"

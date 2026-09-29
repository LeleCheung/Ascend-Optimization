"""The formal eval tool cannot bypass autonomous preflight."""

from types import SimpleNamespace

from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.data.ledger import Ledger
from kernelgen.data.tool_context import ToolContext
from kernelgen.mcp_server import server
from kernelgen.tests.helpers import experiment_plan
from kernelgen.tools import profile_round
from kernelgen.tools.profile_round import EvaluationBundle


class _Model(SimpleNamespace):
    def model_dump(self, mode=None):
        del mode
        return self.data


def test_eval_round_rejects_candidate_without_preflight_receipt(tmp_path, monkeypatch):
    candidate = tmp_path / "tmp" / "main.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("def run(x, out):\n    out.copy_(x)\n", encoding="utf-8")
    definition = _Model(
        data={"name": "copy", "inputs": {"x": {}}, "outputs": {"out": {}}},
        name="copy",
        inputs={"x": {}},
        outputs={"out": {}},
    )
    workload = _Model(data={"uuid": "w0", "axes": {}, "inputs": {}}, uuid="w0")
    bundle = EvaluationBundle(
        kernel_code=candidate.read_text(encoding="utf-8"),
        solution=_Model(data={"name": "candidate"}),
        definition=definition,
        workloads=[SimpleNamespace(workload=workload)],
    )
    context = ToolContext(
        definition="copy",
        target_hardware="Ascend910B",
        eval_server_url="http://eval:8000",
    )
    context.write(tmp_path)
    monkeypatch.setenv("KERNELGEN_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        profile_round,
        "prepare_evaluation_bundle",
        lambda kernel, loaded_context: bundle,
    )

    result = server.eval_round(
        ExperimentPlan.model_validate(experiment_plan(1)),
        "tmp/main.py",
    )

    assert result["status"] == "PREFLIGHT_REQUIRED"
    assert result["required_tool"] == "preflight_kernel"
    assert Ledger(tmp_path).history.rounds == []

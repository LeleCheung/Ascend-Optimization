import pytest

from kernelgen.cli.api import build_request
from kernelgen.framework.run_options import resolve_run_options
from kernelgen.workflows.optimization import OperatorOptimizeInput, OperatorOptimizeWorkflow


@pytest.mark.parametrize("mode", ["simple_opt", "kernelgen"])
@pytest.mark.parametrize("device,backend", [("A100", "cuda"), ("Ascend910B", "ascend")])
@pytest.mark.parametrize("constraint", [None, "A100"])
def test_target_survives_request_roundtrip(tmp_path, monkeypatch, mode, device, backend, constraint):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    extra = {"n_parallel": 1} if mode == "kernelgen" else {}
    values = resolve_run_options({"mode": mode, "catalog_name": "kernelgenbench", "target_hardware": constraint, **extra})
    request = build_request(values, definition="kernelgenbench_square", workspace=tmp_path / "run")
    request = type(request).model_validate_json(request.model_dump_json())
    assert request.target_hardware == constraint
    inp = OperatorOptimizeInput.model_validate(request.workflow_input)
    from kernelgen.tools import kernelgen_server_adapter as adapter
    status = {"api_version": "v6.2", "backend": backend, "timing": "triton",
              "target": {"device": device}, "software": {},
              "capabilities": {"operator_contract": {"enabled": True}}}
    monkeypatch.setattr(adapter, "get_service_status", lambda url: status)
    workflow = OperatorOptimizeWorkflow(cwd=tmp_path / "run")
    monkeypatch.setattr(workflow, "run_build", lambda resolved, **kw: resolved)
    if constraint and constraint != device:
        with pytest.raises(ValueError, match="TARGET_HARDWARE_MISMATCH"):
            workflow._execute(inp)
    else:
        resolved = workflow._execute(inp)
        assert resolved.optimization.target_hardware == device
        assert resolved.target_snapshot["target"]["device"] == device


def test_missing_server_device_does_not_fall_back(tmp_path, monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter as adapter
    inp = OperatorOptimizeInput(operator="square", catalog_name="kernelgenbench",
                               optimization={"definition_name": "square"})
    monkeypatch.setattr(adapter, "get_service_status", lambda url: {"backend": "cuda"})
    with pytest.raises(ValueError, match="TARGET_DEVICE_MISSING"):
        OperatorOptimizeWorkflow(cwd=tmp_path)._execute(inp)

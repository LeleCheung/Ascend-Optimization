"""All new entry points share defaults; persisted explicit options remain fixed."""
from kernelgen.data.constants import DEFAULT_CATALOG_NAME
from kernelgen.framework.catalog_options import catalog_input
from kernelgen.framework.run_options import resolve_run_options
from kernelgen.workflows.optimization import OperatorOptimizeInput
from kernelgen.workflows.optimization.kernelgen import KernelGenInput
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationInput


def test_python_and_cli_have_the_same_default_optimization_plan():
    direct = OperatorOptimizeInput(operator="negative")
    cli = catalog_input(resolve_run_options({}), "negative")
    assert direct.catalog_name == cli.catalog_name == DEFAULT_CATALOG_NAME
    for inp in (direct, cli):
        assert inp.optimization.mode == "kernelgen"
        assert inp.optimization.definition_name == "negative"
        assert inp.optimization.n_parallel == inp.optimization.n_epoch == 1
        assert inp.optimization.max_round == 10
        assert inp.optimization.profile_enabled is True
        assert inp.optimization.target_hardware is None
    assert KernelGenInput.model_fields["n_parallel"].default == 1
    assert KernelGenInput.model_fields["max_round"].default == 10
    assert SingleCoderOptimizationInput.model_fields["max_round"].default == 10


def test_explicit_historical_settings_are_not_replaced():
    inp = OperatorOptimizeInput(operator="negative", optimization={
        "mode": "kernelgen", "n_parallel": 3, "n_epoch": 2,
        "max_round": 15, "profile_enabled": False,
    })
    restored = OperatorOptimizeInput.model_validate_json(inp.model_dump_json())
    assert restored == inp
    assert restored.optimization.n_parallel == 3
    assert restored.optimization.n_epoch == 2
    assert restored.optimization.max_round == 15
    assert restored.optimization.profile_enabled is False


def test_batch_layers_override_shared_defaults():
    values = resolve_run_options({}, {"max_round": 12}, {"n_parallel": 2})
    assert values["mode"] == "kernelgen"
    assert values["max_round"] == 12 and values["n_parallel"] == 2
    assert values["n_epoch"] == 1


def test_explicit_simple_opt_and_local_catalog_remain_supported(tmp_path):
    values = resolve_run_options({"mode": "simple_opt", "catalog_path": tmp_path})
    inp = catalog_input(values, "negative")
    assert inp.catalog_path == tmp_path and inp.catalog_name is None
    assert inp.optimization.mode == "simple_opt" and inp.optimization.max_round == 10

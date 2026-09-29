import pytest

from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot


def snapshot(reference, *, names=("correctness", "timing")):
    workloads = [{"name": name, "inputs": {"case": {"shape": [2, 3]}}, "seed": 42}
                 for name in names]
    return CatalogEvaluationSnapshot(
        catalog_name="native-fixture", catalog_api_version="v6.2", definition_name="identity",
        definition={"name": "identity", "parameters": [{"name": "x", "required": True}],
                    "outputs": ["out"], "reference": reference},
        correctness_workloads=workloads[:1], timing_workloads=workloads[1:],
    )


def test_input_factory_recipe_survives_snapshot_round_trip():
    result = snapshot("def run(x): return x\ndef gen_inputs(ctx, device): return {'x': ctx['inputs']['case']}\n")
    reloaded = CatalogEvaluationSnapshot.model_validate_json(result.model_dump_json())
    assert reloaded == result
    assert reloaded.correctness_workloads[0]["inputs"] == {"case": {"shape": [2, 3]}}
    assert reloaded.native_operator()[0].parameters[0].name == "x"


@pytest.mark.parametrize("reference", [
    "def run(x): return x\n",
    "# gen_inputs is not actually defined\ndef run(x): return x\n",
    "def run(x):\n    def gen_inputs(ctx, device): return {}\n    return x\n",
])
def test_without_factory_invalid_direct_arguments_still_rejected(reference):
    with pytest.raises(ValueError, match="inputs violate Definition"):
        snapshot(reference)


def test_factory_does_not_disable_unique_workload_names():
    with pytest.raises(ValueError, match="globally unique"):
        snapshot("def run(x): return x\ndef gen_inputs(ctx, device): return {}\n", names=("same", "same"))

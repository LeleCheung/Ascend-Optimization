from kernelgen.agents._definition_prompt import render_definition_block
from kernelgen.framework.models import DefinitionModel


def test_render_definition_block_accepts_native_output_name_list():
    definition = DefinitionModel.model_construct(
        name="native_layer_norm",
        op_type="native",
        inputs={"input": {"shape": [2, 4], "dtype": "float32"}},
        outputs=["out", "mean", "rstd"],
        reference="return input",
    )

    prompt = render_definition_block(definition, destination_passing_style=True)

    assert "  out: [] ()" in prompt
    assert "  mean: [] ()" in prompt
    assert "  rstd: [] ()" in prompt
    assert "exactly 4 positional parameters" in prompt


def test_render_definition_block_preserves_output_metadata_dict():
    definition = DefinitionModel(
        name="square",
        op_type="native",
        inputs={"input": {"shape": [2, 4], "dtype": "float32"}},
        outputs={"out": {"shape": [2, 4], "dtype": "float32"}},
        reference="return input",
    )

    prompt = render_definition_block(definition, destination_passing_style=True)

    assert "  out: [2, 4] (float32)" in prompt


def test_render_definition_block_accepts_native_list_and_scalar_inputs():
    definition = DefinitionModel.model_construct(
        name="native_layer_norm",
        op_type="native",
        inputs={
            "input": {"shape": [2, 4], "dtype": "float32"},
            "normalized_shape": [4],
            "eps": 1e-5,
        },
        outputs={"out": {"shape": [2, 4], "dtype": "float32"}},
        reference="return input",
    )

    prompt = render_definition_block(definition, destination_passing_style=False)

    assert "  normalized_shape: [4] (list)" in prompt
    assert "  eps: 1e-05 (float)" in prompt

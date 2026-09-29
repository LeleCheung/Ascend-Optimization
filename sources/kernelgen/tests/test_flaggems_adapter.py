from pathlib import Path

from kernelgen.agents.extractor.flaggems import extract_flaggems_definition
from kernelgen.agents.extractor.flaggems.adapter import (
    _CANONICAL_SIGNATURES,
    _OUTPUTS,
    _effects,
    _parameters,
)
from kernelgen.workflows.flaggems_adapter_extract import (
    FlagGemsAdapterExtractWorkflow,
)


def _repo(tmp_path: Path, signature: str) -> Path:
    ops = tmp_path / "src" / "flag_gems" / "ops"
    tests = tmp_path / "tests"
    benchmark = tmp_path / "benchmark"
    conf = tmp_path / "conf"
    fused = tmp_path / "src" / "flag_gems" / "fused"
    for directory in (ops, tests, benchmark, conf, fused):
        directory.mkdir(parents=True, exist_ok=True)
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.addmm_ import addmm_\n__all__ = ['addmm_']\n",
        encoding="utf-8",
    )
    (fused / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (ops / "addmm_.py").write_text(signature, encoding="utf-8")
    (tests / "test_addmm_.py").write_text("# accuracy\n", encoding="utf-8")
    (benchmark / "test_addmm_.py").write_text("# performance\n", encoding="utf-8")
    (conf / "operators.yaml").write_text(
        "ops:\n  - id: addmm_\n    for: [addmm_]\n",
        encoding="utf-8",
    )
    return tmp_path


def test_addmm_extraction_emits_only_common_definition(tmp_path):
    repo = _repo(
        tmp_path,
        "def addmm_(self, mat1, mat2, *, beta=1, alpha=1): return self\n",
    )
    spec = extract_flaggems_definition(repo, "addmm_")

    assert spec.name == "addmm_"
    assert [parameter.name for parameter in spec.parameters] == [
        "self",
        "mat1",
        "mat2",
        "beta",
        "alpha",
    ]
    assert [parameter.kind for parameter in spec.parameters[-2:]] == [
        "keyword_only",
        "keyword_only",
    ]
    payload = spec.model_dump(mode="json", exclude_unset=True)
    assert "framework" not in payload
    assert "correctness_file" not in payload
    assert "performance_file" not in payload
    assert "definition" not in payload
    assert "correctness_workloads" not in payload


def test_definition_reflects_the_checked_out_public_signature(tmp_path):
    repo = _repo(
        tmp_path,
        "def addmm_(self, mat1, mat2, beta=1, alpha=1): return self\n",
    )
    spec = extract_flaggems_definition(repo, "addmm_")
    assert [parameter.kind for parameter in spec.parameters[-2:]] == [
        "positional_or_keyword",
        "positional_or_keyword",
    ]


def test_workflow_writes_common_definition_without_registry_fields(tmp_path):
    repo = _repo(
        tmp_path / "repo",
        "def addmm_(self, mat1, mat2, *, beta=1, alpha=1): return self\n",
    )
    output = FlagGemsAdapterExtractWorkflow().run(
        {
            "operator": "addmm_",
            "flaggems_repo": str(repo),
            "adapter_root": str(tmp_path / "out"),
        }
    )
    payload = Path(output.adapter_path).read_text(encoding="utf-8")
    assert output.adapter_path.endswith("addmm_.json")
    assert '"api_version": "v6.0"' in payload
    assert '"name": "addmm_"' in payload
    assert '"framework"' not in payload
    assert '"correctness_file"' not in payload


def test_wrapper_canonical_signatures_expose_real_public_abi():
    for operator, expected in {
        "digamma_": ["self"],
        "i0_": ["self"],
        "functional_sym_constrain_range": ["size", "min", "max", "dep_token"],
    }.items():
        parameters = _parameters(_CANONICAL_SIGNATURES[operator])
        assert [parameter.name for parameter in parameters] == expected


def test_nonzero_numpy_declares_dynamic_tensor_list_as_one_output():
    assert _OUTPUTS["nonzero_numpy"] == ["out"]


def test_sdpa_forward_preserves_nine_tuple_public_contract(tmp_path):
    repo = _repo(tmp_path, "def addmm_(self, mat1, mat2): return self\n")
    (repo / "conf/operators.yaml").write_text(
        "ops:\n  - id: scaled_dot_product_cudnn_attention\n"
        "    for: [_scaled_dot_product_cudnn_attention]\n")
    ops = repo / "src/flag_gems/ops"
    (ops / "__init__.py").write_text(
        "from flag_gems.ops._scaled_dot_product_cudnn_attention import _scaled_dot_product_cudnn_attention\n"
        "__all__ = ['_scaled_dot_product_cudnn_attention']\n")
    (ops / "_scaled_dot_product_cudnn_attention.py").write_text(
        "def _scaled_dot_product_cudnn_attention(query, key, value, attn_bias=None, "
        "compute_log_sumexp=True, dropout_p=0.0, is_causal=False, "
        "return_debug_mask=False, *, scale=None):\n"
        "    return (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, "
        "philox_seed, philox_offset, debug_attn_mask)\n")
    for suite in ("tests", "benchmark"):
        (repo / suite / "test_scaled_dot_product_cudnn_attention.py").write_text("# fixture\n")
    spec = extract_flaggems_definition(repo, "scaled_dot_product_cudnn_attention")
    assert spec.outputs == ["output", "logsumexp", "cum_seq_q", "cum_seq_k", "max_q", "max_k",
                            "philox_seed", "philox_offset", "debug_attn_mask"]
    assert spec.parameters[-1].name == "scale"
    assert spec.parameters[-1].kind == "keyword_only"
    assert spec.effects.mutates == []


def test_ilshift_catalog_id_keeps_inplace_effects():
    parameters = _parameters(_CANONICAL_SIGNATURES["ilshift"])
    effects = _effects("ilshift", parameters, ["out"])
    assert [parameter.name for parameter in parameters] == ["self", "other"]
    assert effects.mutates == ["self"]
    assert effects.returns_alias_of == {"out": "self"}


def test_public_mutation_schemas_cover_dunder_list_and_void_ops():
    signatures = {
        "dunder_ior_scalar": _CANONICAL_SIGNATURES["dunder_ior_scalar"],
        "amp_foreach_non_finite_check_and_unscale_": (
            "def op(tensors, found_inf, inv_scale): pass"
        ),
        "fused_adam_": (
            "def op(self, grads, exp_avgs, exp_avg_sqs, "
            "max_exp_avg_sqs, state_steps): pass"
        ),
    }
    expected = {
        "dunder_ior_scalar": (["out"], ["self"], {"out": "self"}),
        "amp_foreach_non_finite_check_and_unscale_": (
            [],
            ["tensors", "found_inf"],
            {},
        ),
        "fused_adam_": (
            [],
            ["self", "grads", "exp_avgs", "exp_avg_sqs", "max_exp_avg_sqs"],
            {},
        ),
    }
    for operator, (outputs, mutates, aliases) in expected.items():
        parameters = _parameters(signatures[operator])
        effects = _effects(operator, parameters, outputs)
        assert effects.mutates == mutates
        assert effects.returns_alias_of == aliases


def test_framework_dtype_default_is_json_token():
    parameters = _parameters(
        "def randint(high, size, *, dtype=torch.int64): pass"
    )
    assert parameters[-1].name == "dtype"
    assert parameters[-1].default == "int64"

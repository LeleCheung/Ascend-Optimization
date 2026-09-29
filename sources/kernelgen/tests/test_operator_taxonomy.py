"""Canonical operator taxonomy and Scope semantics tests."""

from __future__ import annotations

import pytest
import yaml

from kernelgen.knowledge.context import build_operator_signature
from kernelgen.knowledge.models import QueryContext, Scope, TargetContext, TargetScope
from kernelgen.knowledge.scope import match_scope
from kernelgen.knowledge.taxonomy import (
    OperatorTaxonomy,
    OperatorTaxonomyError,
    UnmappedOperatorDefinitionError,
)
from kernelgen.knowledge.vocabulary import Vocabulary


def _taxonomy_payload() -> dict:
    return {
        "schema_version": "1.1",
        "taxonomy_version": "test-1",
        "scope_semantics": {
            "definition_ids": "exact_any_overlap",
            "families": "any_overlap",
            "motifs": "concept_required_subset_of_query",
            "dataflows": "any_overlap",
            "dtypes": "any_overlap",
            "layouts": "any_overlap",
            "aliases": "exact_token",
        },
        "activation": {"status": "active", "client_compatible": True},
        "families": {
            "attention": {
                "description": "attention",
                "aliases": ["attention operator", "注意力算子"],
                "status": "stable",
            }
        },
        "motifs": {
            "matrix_multiply": {
                "description": "matrix multiply",
                "aliases": ["matmul", "gemm", "矩阵乘法"],
                "kind": "compute",
                "status": "stable",
            },
            "softmax": {
                "description": "softmax",
                "aliases": ["online softmax", "softmax归一化"],
                "kind": "normalization",
                "status": "stable",
            },
        },
        "dataflows": {
            "qk_softmax_pv": {
                "family": "attention",
                "required_motifs": ["matrix_multiply", "softmax"],
                "stages": ["qk", "softmax", "pv"],
                "aliases": ["attention dataflow", "注意力数据流"],
                "status": "stable",
            }
        },
        "definitions": {
            "ks_attention": {
                "family": "attention",
                "motifs": ["matrix_multiply", "softmax"],
                "dataflow": ["qk_softmax_pv"],
                "dtypes": ["float16"],
                "layouts": ["contiguous"],
                "workload_features": {"heads": 8},
            }
        },
    }


def _write_taxonomy(root) -> None:
    (root / "operator_taxonomy.yaml").write_text(
        yaml.safe_dump(_taxonomy_payload(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_active_taxonomy_emits_canonical_signature_and_fails_unmapped(tmp_path):
    _write_taxonomy(tmp_path)

    signature = build_operator_signature(
        {
            "name": "ks_attention",
            "op_type": "legacy_attention_label",
            "inputs": {"q": {"dtype": "float16"}},
            "reference": "def run(q): return q",
        },
        catalog_root=tmp_path,
    )

    assert signature.op_type == "attention"
    assert signature.motifs == ["matrix_multiply", "softmax"]
    assert signature.dataflow == ["qk_softmax_pv"]
    assert signature.dtypes == ["float16"]
    assert signature.layouts == ["contiguous"]
    assert signature.workload_features == {"heads": 8}

    with pytest.raises(UnmappedOperatorDefinitionError, match="unmapped"):
        build_operator_signature(
            {"name": "new_definition", "inputs": {}, "outputs": {}},
            catalog_root=tmp_path,
        )


def test_preview_taxonomy_cannot_silently_activate():
    payload = _taxonomy_payload()
    payload["activation"] = {
        "status": "preview_only",
        "client_compatible": False,
    }

    with pytest.raises(OperatorTaxonomyError, match="not active"):
        OperatorTaxonomy.from_mapping(payload)
    assert OperatorTaxonomy.from_mapping(payload, allow_preview=True)


def test_taxonomy_aliases_expand_english_and_chinese_queries(tmp_path):
    _write_taxonomy(tmp_path)
    vocabulary = Vocabulary.from_catalog(tmp_path)

    english = vocabulary.expand("Ascend matmul lowering")
    chinese = vocabulary.expand("优化矩阵乘法的数据布局")

    assert {"matrix_multiply", "matmul", "gemm", "矩阵乘法"} <= set(english)
    assert {"matrix_multiply", "matmul", "gemm", "矩阵乘法"} <= set(chinese)


def test_scope_requires_all_concept_motifs_but_other_lists_overlap():
    signature = build_operator_signature(
        {
            "name": "legacy_matmul",
            "op_type": "attention",
            "inputs": {},
            "outputs": {},
            "reference": "def run(a, b): return a @ b",
            "dataflow": ["qk_softmax_pv", "alternate"],
        }
    )
    target = TargetContext(
        backend="cuda",
        device="A100",
        source="fixture",
    )
    scope = Scope(
        target=TargetScope(level="portable"),
        operator={
            "motifs": ["matrix_multiply", "softmax"],
            "dataflow": ["qk_softmax_pv"],
        },
    )

    missing = match_scope(
        scope,
        QueryContext(
            phase="initial",
            task="implementation",
            question="attention",
            operator_signature=signature,
            target_context=target,
        ),
    )
    complete = match_scope(
        scope,
        QueryContext(
            phase="initial",
            task="implementation",
            question="attention",
            operator_signature=signature.model_copy(
                update={"motifs": ["matrix_multiply", "softmax"]}
            ),
            target_context=target,
        ),
    )

    assert missing.level == "incompatible"
    assert missing.conflicts == ["operator:motif:missing:softmax"]
    assert complete.level == "direct"
    assert "operator:dataflow:qk_softmax_pv" in complete.matched_on

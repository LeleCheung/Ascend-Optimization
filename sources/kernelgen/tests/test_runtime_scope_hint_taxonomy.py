"""Publisher guardrails for active operator taxonomies."""

from __future__ import annotations

import yaml

from kernelgen.knowledge.models import RuntimeScopeHints
from kernelgen.knowledge.publishing.materialize import (
    _canonicalize_runtime_scope_hints,
)
from kernelgen.knowledge.taxonomy import OperatorTaxonomy


def _taxonomy_payload() -> dict:
    return {
        "schema_version": "1.1",
        "taxonomy_version": "test-runtime-hints-1",
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
        "families": {"map": {"aliases": [], "status": "stable"}},
        "motifs": {
            "elementwise": {
                "aliases": ["pointwise"],
                "kind": "compute",
                "status": "stable",
            },
            "sincos": {
                "aliases": ["sin_cos"],
                "kind": "compute",
                "status": "stable",
            },
        },
        "dataflows": {
            "elementwise_map": {
                "family": "map",
                "required_motifs": ["elementwise"],
                "stages": ["map"],
                "aliases": [],
                "status": "stable",
            }
        },
        "definitions": {
            "test_sincos": {
                "family": "map",
                "motifs": ["elementwise", "sincos"],
                "dataflow": ["elementwise_map"],
                "dtypes": ["float32"],
                "layouts": ["contiguous"],
                "workload_features": {},
            }
        },
    }


def test_runtime_scope_hints_use_only_active_taxonomy_motifs(tmp_path):
    (tmp_path / "operator_taxonomy.yaml").write_text(
        yaml.safe_dump(_taxonomy_payload(), sort_keys=False),
        encoding="utf-8",
    )
    taxonomy = OperatorTaxonomy.from_catalog(tmp_path)

    hints, unknown = _canonicalize_runtime_scope_hints(
        RuntimeScopeHints(
            motifs=["sin_cos", "elementwise", "invented-latency-bound"]
        ),
        taxonomy,
    )

    assert hints.motifs == ["elementwise", "sincos"]
    assert unknown == ["invented-latency-bound"]

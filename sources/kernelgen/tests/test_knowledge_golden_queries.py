"""Golden recall checks for the committed static knowledge catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.query import QueryKnowledge
from kernelgen.knowledge.sources import SourceSearcher
from kernelgen.knowledge.publishing.static_import import StaticKnowledgeImporter
from kernelgen.knowledge.models import (
    OperatorSignature,
    QueryContext,
    TargetContext,
)


def _query(
    kb: Path,
    tmp_path: Path,
    *,
    backend: str,
    architecture: str,
    device: str,
    phase: str,
    task: str,
    question: str,
    language: str = "triton",
    op_type: str = "elementwise",
    motifs: list[str] | None = None,
    dtypes: list[str] | None = None,
    findings: list[dict] | None = None,
):
    catalog = FilesystemCatalog(kb)
    service = QueryKnowledge(
        catalog,
        SQLiteKnowledgeIndex(tmp_path / f"{backend}.db"),
        FileRetrievalAudit(tmp_path / f"{backend}.retrieval.jsonl"),
    )
    context = QueryContext(
        phase=phase,
        task=task,
        round_num=1 if phase in {"post_profile", "post_error"} else None,
        question=question,
        operator_signature=OperatorSignature(
            definition_id="golden:elementwise:v1",
            definition_name="golden_elementwise",
            op_type=op_type,
            motifs=motifs or ["elementwise"],
            dataflow=["load_compute_store"],
            dtypes=dtypes or ["int64"],
        ),
        target_context=TargetContext(
            backend=backend,
            architecture=architecture,
            device=device,
            software={
                "language": language,
                "compiler": (
                    "triton-ascend"
                    if language == "triton" and backend == "ascend"
                    else ("triton" if language == "triton" else "")
                ),
            },
            source="fixture",
        ),
        findings=findings or [],
        errors=["synthetic compiler failure"] if phase == "post_error" else [],
        max_results=20,
    )
    return service.execute(context)


def test_ascend_scalar_fallback_query_recalls_reviewed_diagnostic(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="ascend",
        architecture="DAV_2201",
        device="Ascend910B",
        phase="post_profile",
        task="diagnosis",
        question="How should i64 comparison scalar fallback be diagnosed?",
        findings=[
            {
                "category": "pipeline",
                "label": "scalar_fallback",
                "confidence": "high",
            }
        ],
    )
    refs = {item.concept_ref for item in bundle.direct}
    assert "kg:diagnostic:triton-ascend-scalar-lowering-routing" in refs


def test_static_only_source_rejects_explicit_search():
    root = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match="static_only"):
        SourceSearcher(root / "kb").search(
            "coalesced access",
            package_ids=["source:cuda-best-practices"],
        )


def test_restricted_ascend_knowledge_is_not_direct_on_cuda(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="cuda",
        architecture="sm80",
        device="NVIDIA-A100-SXM4-80GB",
        phase="post_profile",
        task="diagnosis",
        question="How should i64 comparison scalar fallback be diagnosed?",
        findings=[
            {
                "category": "pipeline",
                "label": "scalar_fallback",
                "confidence": "high",
            }
        ],
    )
    returned = [
        item.concept_ref
        for item in [*bundle.direct, *bundle.analogies, *bundle.conflicts]
    ]
    assert not any("triton-ascend" in ref for ref in returned)


def test_dav2201_hardware_query_recalls_architecture_reference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="ascend",
        architecture="DAV_2201",
        device="Ascend910B",
        phase="initial",
        task="constraint_check",
        question="What are the DAV_2201 UB L1 and L0C memory limits?",
    )
    refs = {item.concept_ref for item in bundle.direct}
    assert "kg:reference:ascend-dav2201-memory-hierarchy" in refs
    result = next(
        item
        for item in bundle.direct
        if item.concept_ref == "kg:reference:ascend-dav2201-memory-hierarchy"
    )
    assert "backend:ascend" in result.matched_on
    assert "architecture:DAV_2201" in result.matched_on


def test_ascendc_double_buffer_query_recalls_reviewed_method(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="ascend",
        architecture="DAV_2201",
        device="Ascend910B",
        language="ascendc",
        phase="initial",
        task="implementation",
        question="How should TQue double buffer overlap MTE2 and Vector compute?",
    )
    refs = {item.concept_ref for item in bundle.direct}
    assert "kg:method:ascendc-double-buffer-pipeline" in refs


def test_ascendc_datacopypad_alignment_query_recalls_diagnostic(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="ascend",
        architecture="DAV_2201",
        device="Ascend910B",
        language="ascendc",
        phase="post_error",
        task="diagnosis",
        question="DataCopyPad UB address AIV error 80 alignment",
    )
    refs = {item.concept_ref for item in bundle.direct}
    assert "kg:diagnostic:ascendc-datacopypad-ub-address-alignment" in refs


def test_restricted_ascendc_knowledge_is_not_returned_on_cuda(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="cuda",
        architecture="sm80",
        device="NVIDIA-A100-SXM4-80GB",
        language="ascendc",
        phase="initial",
        task="implementation",
        question="TQue double buffer DataCopyPad",
    )
    returned = [
        item.concept_ref
        for item in [*bundle.direct, *bundle.analogies, *bundle.conflicts]
    ]
    assert not any("ascendc-" in ref for ref in returned)


def test_triton_ascend_int8_quantization_recalls_saturating_cast(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="ascend",
        architecture="DAV_2201",
        device="Ascend910B",
        language="triton",
        motifs=["elementwise", "quantization", "type_conversion"],
        dtypes=["float32", "int8"],
        phase="initial",
        task="implementation",
        question="Quantize float32 values to int8 without overflow using saturate",
    )
    refs = {item.concept_ref for item in bundle.direct}
    assert "kg:method:triton-ascend-int8-saturating-quantization-cast" in refs


def test_triton_ascend_int8_matmul_recalls_int32_accumulation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = _query(
        root / "kb",
        tmp_path,
        backend="ascend",
        architecture="DAV_2201",
        device="Ascend910B",
        language="triton",
        op_type="matmul",
        motifs=["matrix_multiply"],
        dtypes=["int8"],
        phase="initial",
        task="implementation",
        question="Implement int8 tl.dot matmul with an int32 accumulator",
    )
    refs = {item.concept_ref for item in bundle.direct}
    assert "kg:method:triton-ascend-int8-matmul-int32-accumulator" in refs
    assert "kg:reference:triton-ascend-int8-dot-contract" in refs


def test_triton_ascend_int8_dot_dtype_diagnostic_is_cuda_isolated(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ascend = _query(
        root / "kb",
        tmp_path / "ascend",
        backend="ascend",
        architecture="DAV_2201",
        device="Ascend910B",
        language="triton",
        op_type="matmul",
        motifs=["matrix_multiply"],
        dtypes=["int8"],
        phase="post_error",
        task="diagnosis",
        question="tl.dot int8 out_dtype compile error",
    )
    assert (
        "kg:diagnostic:triton-ascend-int8-dot-output-dtype"
        in {item.concept_ref for item in ascend.direct}
    )

    cuda = _query(
        root / "kb",
        tmp_path / "cuda",
        backend="cuda",
        architecture="sm80",
        device="NVIDIA-A100-SXM4-80GB",
        language="triton",
        op_type="matmul",
        motifs=["matrix_multiply"],
        dtypes=["int8"],
        phase="post_error",
        task="diagnosis",
        question="tl.dot int8 out_dtype compile error",
    )
    returned = [
        item.concept_ref
        for item in [*cuda.direct, *cuda.analogies, *cuda.conflicts]
    ]
    assert not any("triton-ascend-int8" in ref for ref in returned)


def test_committed_direct_concepts_match_resolved_source_material():
    root = Path(__file__).resolve().parents[1]
    kb = root / "kb"
    concepts = {
        concept.id: concept
        for concept in FilesystemCatalog(kb).iter_concepts()
    }
    plan_names = [
        "triton-ascend-a60006ac63f3.yaml",
        "cannbot-skills-7fedd2ff5e1f.yaml",
    ]
    checked = 0
    for plan_name in plan_names:
        importer = StaticKnowledgeImporter.from_plan_file(
            kb,
            kb / "sources" / "ingestion" / plan_name,
        )
        for candidate, entry in zip(
            importer.build_candidates(),
            importer.plan.entries,
            strict=True,
        ):
            if entry.mode != "direct":
                continue
            assert entry.proposed_id is not None
            concept = concepts[entry.proposed_id]
            assert concept.body == candidate.body
            assert "# Source material" in concept.body
            checked += 1
    assert checked == 54

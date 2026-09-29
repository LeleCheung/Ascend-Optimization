"""Resolve Catalog sources once; optimizers only receive frozen contracts."""

from kernelgen.data._atomic import atomic_write_json
from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot
from .artifacts import workflow_result


def review_source_files(source, workspace):
    """Derive readable files from frozen JSON; never resolve remote source paths locally."""
    from pathlib import Path
    from kernelgen_client.protocol.schema import TestReviewSources
    from .artifacts import file_digest

    source = Path(source)
    evidence = TestReviewSources.model_validate_json(source.read_text(encoding="utf-8"))
    directory = Path(workspace) / "review-sources"
    directory.mkdir(parents=True, exist_ok=False)
    paths, entries = [], []
    for number, item in enumerate(evidence.files, 1):
        # Remote paths are labels only, never filesystem destinations or imports.
        path = directory / f"{number:02}.txt"
        path.write_text(item.content, encoding="utf-8")
        paths.append(path)
        entries.append({"source_path": item.path, "read_path": str(path), "sha256": file_digest(path)})
    index = directory / "index.json"
    atomic_write_json(index, {"source_snapshot": str(source), "source_sha256": file_digest(source),
                             "framework_revision": evidence.framework_revision, "files": entries})
    return [index, *paths]


def prepare_test_evidence(inp, prepared, context):
    """Freeze the target-owned sources, not a same-named local checkout."""
    from kernelgen.data.evaluation_snapshot import load_catalog_evaluation_snapshot
    from kernelgen.tools.kernelgen_server_adapter import get_service_status
    from kernelgen_client import EvaluatorBinding, InspectRequest
    from kernelgen_client.http import get_operator_contract

    server = inp.optimization.eval_server_url
    capability = get_service_status(server).get("capabilities", {}).get("operator_contract", {})
    if capability.get("enabled") is not True or capability.get("test_sources") is not True:
        raise RuntimeError("KGS test-source export is required for review_tests; upgrade KGS or explicitly set skip_review")
    snapshot = load_catalog_evaluation_snapshot(prepared["snapshot"])
    binding = EvaluatorBinding(definition=context.operator, **(
        {"bundle_id": snapshot.bundle_id} if snapshot.bundle_id else {"catalog_name": snapshot.catalog_name}))
    contract = get_operator_contract(InspectRequest(binding=binding), server, include_test_sources=True)
    observed = CatalogEvaluationSnapshot.from_server_contract(contract, snapshot.benchmark_fingerprint)
    if (observed.definition != snapshot.definition or observed.evaluator_kind != snapshot.evaluator_kind
            or observed.correctness_workloads != snapshot.correctness_workloads
            or observed.timing_workloads != snapshot.timing_workloads):
        raise ValueError("test review contract differs from the frozen evaluation input")
    if contract.test_sources is None:
        raise ValueError("test review requires actual source evidence")
    path = context.workspace / "test-sources.json"
    atomic_write_json(path, contract.test_sources.model_dump(mode="json"))
    return path


def require_source_policy(definition, server):
    if not definition.source_policy_id:
        return
    from kernelgen.tools.kernelgen_server_adapter import get_service_status
    capability = get_service_status(server).get('capabilities', {}).get('native_source_policy', {})
    if (capability.get('enabled') is not True or capability.get('workload_conditions') is not True
            or capability.get('policy_id') != definition.source_policy_id):
        raise RuntimeError('KGS does not provide the Native Catalog source policy contract')
    if capability.get('flags') is None:
        raise RuntimeError('KGS source policy is unknown for the target vendor')


def prepare_installed_catalog(inp, context):
    from kernelgen_client import EvaluatorBinding, InspectRequest
    from kernelgen_client.http import get_operator_contract, inspect

    binding = EvaluatorBinding(catalog_name=inp.catalog_name, definition=context.operator)
    request = InspectRequest(binding=binding)
    server = inp.optimization.eval_server_url
    contract = get_operator_contract(request, server)
    require_source_policy(contract.definition, server)
    if contract.kind == "native" and (not contract.correctness_workloads or not contract.timing_workloads):
        raise ValueError("optimization requires both correctness and timing workloads")
    manifest = inspect(request, server)
    if manifest.kind != contract.kind:
        raise ValueError("KGS inspect and operator contract disagree on evaluator kind")
    snapshot = CatalogEvaluationSnapshot.from_server_contract(contract, manifest.benchmark_fingerprint)
    path = context.workspace / "evaluation-contract.json"
    atomic_write_json(path, snapshot.model_dump(mode="json"))
    return workflow_result({"snapshot": str(path), "source": "installed_catalog"}, [path])


def upload_catalog_snapshot(inp, extracted, context):
    """Bind prepared content before review; upload is storage, not execution."""
    from kernelgen_client import Catalog, EvaluatorBinding, InspectRequest
    from kernelgen_client.http import upload_operator_bundle, inspect, get_operator_contract

    server = inp.optimization.eval_server_url
    catalog = Catalog(extracted["catalog"])
    operator = catalog.load(context.operator)
    require_source_policy(operator.definition, server)
    bundle = upload_operator_bundle(extracted["operator_dir"], server)
    if bundle.bundle_id != extracted["bundle_id"]:
        raise ValueError("catalog content changed during upload")
    binding = EvaluatorBinding(bundle_id=bundle.bundle_id, definition=context.operator)
    request = InspectRequest(binding=binding)
    manifest = inspect(request, server)
    if manifest.kind != catalog.evaluator:
        raise ValueError("KGS inspect disagrees with the uploaded evaluator kind")
    if catalog.evaluator == "flaggems":
        contract = get_operator_contract(request, server)
        if contract.kind != "flaggems" or contract.definition != operator.definition:
            raise ValueError("KGS contract differs from the uploaded Gems Definition")
        return CatalogEvaluationSnapshot.from_server_contract(contract, manifest.benchmark_fingerprint)
    if not operator.correctness_workloads or not operator.timing_workloads:
        raise ValueError("optimization requires both correctness and timing workloads")
    return CatalogEvaluationSnapshot(
        catalog_name="uploaded-" + bundle.sha256, catalog_api_version="v6.2",
        definition_name=context.operator, definition=operator.definition.model_dump(mode="json", exclude_unset=True),
        correctness_workloads=[w.model_dump(mode="json", exclude_unset=True) for w in operator.correctness_workloads],
        timing_workloads=[w.model_dump(mode="json", exclude_unset=True) for w in operator.timing_workloads],
        bundle_id=bundle.bundle_id, benchmark_fingerprint=manifest.benchmark_fingerprint,
    )

"""Adapters for a fixed-environment, single-operator Catalog lifecycle."""

import hashlib
import os
from pathlib import Path

from kernelgen.data._atomic import atomic_write_json
from kernelgen.data.timeout_policy import TimeoutPolicy
from .artifacts import file_digest, workflow_result, snapshot_catalog


def receipt_data(context, stage):
    try:
        return context.outputs[stage]
    except KeyError as exc:
        raise ValueError(f"missing successful upstream workflow: {stage}") from exc


def prepare_catalog(inp, root, runtime_factory, context):
    from .sources import upload_catalog_snapshot
    if inp.catalog_name is not None:
        from .sources import prepare_installed_catalog
        result = prepare_installed_catalog(inp, context)
    else:
        prepared = snapshot_catalog(inp, context.workspace).output
        snapshot = upload_catalog_snapshot(inp, prepared, context)
        path = context.workspace / "evaluation-contract.json"
        atomic_write_json(path, snapshot.model_dump(mode="json"))
        result = workflow_result({**prepared, "snapshot": str(path)}, [*prepared["artifacts"], path])
    if not inp.skip_review:
        from .sources import prepare_test_evidence
        path = prepare_test_evidence(inp, result.output, context)
        result.output["test_evidence"] = str(path)
        result.output["artifacts"][str(path)] = file_digest(path)
    return result

def _review(runtime_factory, context, kind, digest, evidence):
    from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent
    report = ArtifactReviewerAgent().run({"kind": kind, "operator": context.operator,
                                          "subject_sha256": digest, "evidence_paths": [str(p) for p in evidence]},
                                         runtime_factory(str(context.workspace)))
    path = context.workspace / "review.json"
    policy = "advisory" if kind == "code" else "gate"
    atomic_write_json(path, {"subject_sha256": digest, "policy": policy, **report.model_dump(mode="json")})
    read = {str(Path(p).resolve()) for p in report.reviewed_files}
    missing = {str(Path(p).resolve()) for p in evidence} - read
    blocked = report.blocks_tests() if kind == "tests" else report.blocks_catalog()
    accepted = not missing and (policy == "advisory" or not blocked)
    return workflow_result({"review": str(path), "subject_sha256": digest, "policy": policy, "missing_evidence": sorted(missing)},
                        [path], state="SUCCEEDED" if accepted else "WAITING",
                        message=report.summary if accepted else "Review requires changes or missing evidence; no downstream execution")

def review_tests(inp, root, runtime_factory, context):
    prepared = receipt_data(context, "prepare_catalog")
    files = []
    if not inp.skip_review:
        from .sources import review_source_files
        evidence = [Path(prepared["snapshot"]), *review_source_files(prepared["test_evidence"], context.workspace)]
        digest = hashlib.sha256("".join(file_digest(p) for p in evidence).encode()).hexdigest()
        reviewed = _review(runtime_factory, context, "tests", digest, evidence)
        if reviewed.state != "SUCCEEDED":
            return reviewed
        files.extend(evidence)
        files.extend(reviewed.output["artifacts"])
    else:
        path = context.workspace / "review.json"
        atomic_write_json(path, {"policy": "gate", "status": "SKIPPED_EXPLICIT", "skip_review": True,
                                 "subject_sha256": file_digest(Path(prepared["snapshot"]))})
        files.append(path)
    result = validate_target(inp, context, prepared, files)
    result.output["test_review"] = "SKIPPED_EXPLICIT" if inp.skip_review else "ACCEPTED"
    return result

def validate_target(inp, context, prepared, files=()):
    """Deterministic target gate, independent of model review acceptance."""
    from kernelgen.data.evaluation_snapshot import load_catalog_evaluation_snapshot
    from types import SimpleNamespace
    from kernelgen_client import BoundEvaluateRequest, EvaluatorBinding, EvaluationSettings
    from .reference_readiness import reference_implementation
    from kernelgen_client.http import preflight, evaluate, OperationCancelledError
    from kernelgen.tools.kernelgen_server_adapter import get_service_status, tracked_server_operation

    server = inp.optimization.eval_server_url
    status = get_service_status(server)
    snapshot = load_catalog_evaluation_snapshot(prepared["snapshot"])
    files = [*files, Path(prepared["snapshot"])]
    binding = EvaluatorBinding(definition=context.operator,
        **({"bundle_id": snapshot.bundle_id} if snapshot.bundle_id else {"catalog_name": snapshot.catalog_name}))
    if snapshot.evaluator_kind == "flaggems":
        from kernelgen_client import ReferenceRequest
        from kernelgen_client.http import reference
        capability = status.get("capabilities", {}).get("benchmark_reference", {})
        if (capability.get("enabled") is not True or capability.get("level") != "core"
                or "flaggems" not in capability.get("evaluators", [])):
            raise RuntimeError("KGS benchmark core reference validation is required before optimization")
        request = ReferenceRequest(binding=binding, benchmark_fingerprint=snapshot.benchmark_fingerprint,
                                   settings={"timeout_seconds": inp.optimization.eval_timeout_seconds})
        try:
            with tracked_server_operation(context.control, status, server, "reference") as operation_id:
                result = reference(request, server,
                                   timeout=TimeoutPolicy(inp.optimization.eval_timeout_seconds).eval_transport_timeout_seconds,
                                   operation_id=operation_id)
        except OperationCancelledError:
            context.control.checkpoint("AFTER_REFERENCE")
            raise
        path = context.workspace / "benchmark-reference.json"
        atomic_write_json(path, result.model_dump(mode="json"))
        files.append(path)
        accepted = result.status == "PASSED"
        return workflow_result({"snapshot": prepared["snapshot"], "validation_scope": "benchmark_core",
                                "readiness": "benchmark_core_passed" if accepted else result.status,
                                "correctness": "candidate_preflight_pending"}, files,
                               state="SUCCEEDED" if accepted else "FAILED",
                               message=f"Benchmark core reference {result.status}; correctness reference is not validated here")
    definition, _, _ = snapshot.native_operator()
    operator = SimpleNamespace(definition=definition)
    options = inp.optimization
    request = BoundEvaluateRequest(binding=binding,
        implementation=reference_implementation(operator),
        settings=EvaluationSettings(warmup_ms=options.warmup_ms, benchmark_ms=options.benchmark_ms,
                                    num_trials=options.num_trials, timeout_seconds=options.eval_timeout_seconds))
    for operation, execute in (("preflight", preflight), ("evaluate", evaluate)):
        try:
            with tracked_server_operation(context.control, status, server, operation) as operation_id:
                result = execute(request, server, timeout=TimeoutPolicy(options.eval_timeout_seconds).eval_transport_timeout_seconds, operation_id=operation_id)
        except OperationCancelledError:
            context.control.checkpoint(f"AFTER_{operation.upper()}")
            raise
        if operation == "evaluate":
            result = result.model_dump(mode="json")
        path = context.workspace / f"{operation}.json"
        atomic_write_json(path, result)
        files.append(path)
        if result.get("status") != "PASSED":
            return workflow_result({}, files, state="FAILED", message=f"Target {operation} readiness failed")
        if operation == "evaluate" and (not result.get("geo_mean") or result.get("num_workloads", 0) <= 0):
            return workflow_result({}, files, state="FAILED", message="Reference readiness has no valid measured result")
    return workflow_result({"snapshot": prepared["snapshot"], "readiness": "PASSED"}, files)

def optimize(inp, root, runtime_factory, context):
    """Continue the optimizer's ledger; stage attempts only version receipts.

    Unlike extraction/review, this adapter deliberately does not execute in
    context.workspace. SingleCoderOptimizationWorkflow recovers from the stable
    workspace itself, including frozen evaluation input and stop decisions;
    no generic Stage resume flag or second round counter is needed.
    """
    from kernelgen.data.evaluation_snapshot import load_catalog_evaluation_snapshot, snapshot_optimization_context
    from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow
    from kernelgen.workflows.optimization.single_coder.inputs import build_optimizer_input
    from kernelgen.framework.local_state import process_start_identity
    from kernelgen.framework.worker_pool import WorkerLeasePool, LeaseRecord

    reviewed = receipt_data(context, "review_tests")
    snapshot = load_catalog_evaluation_snapshot(reviewed["snapshot"])
    definition, workloads = snapshot_optimization_context(snapshot)
    workspace = root / "stages" / "optimize" / "work"
    workspace.mkdir(parents=True, exist_ok=True)
    control = context.control.link_workspace(workspace, scope=context.control.scope)
    options = build_optimizer_input(inp.optimization,
                   definition=definition, workloads=workloads, catalog_name=snapshot.catalog_name,
                   evaluation_snapshot=snapshot.model_dump(mode="json"),
                   destination_passing_style=getattr(inp.optimization, "destination_passing_style", None) or False)
    from kernelgen.workflows.optimization.single_coder.reference import load_reference_code, _load_reference_text
    reference = load_reference_code(inp.optimization.reference_code_path, inp.optimization.reference_code_prompt_path)
    seed = _load_reference_text(inp.optimization.seed_code_path, 'seed_code_path')
    for name, digest in inp.code_inputs_sha256.items():
        if hashlib.sha256(_load_reference_text(getattr(inp.optimization,name),name).encode()).hexdigest() != digest:
            raise ValueError('reference or seed source changed after preparation')
    if any(reference.values()):
        options.update(reference)
    if seed:
        options.update(seed_code=seed, seed_is_validated_baseline=False)
    kernelgen = inp.optimization.mode == 'kernelgen'
    knowledge_run_id = f"{root.name}-{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]}"
    pool = WorkerLeasePool(inp.optimization.eval_server_url)
    lease = LeaseRecord(run_id=str(root), pid=os.getpid(), process_start=process_start_identity(os.getpid()),
                        workspace=workspace, worker_pool=pool.worker_pool, weight=inp.optimization.n_parallel if kernelgen else 1)
    try:
        from kernelgen.framework.run_control import RunState
        control.update_progress(state=RunState.QUEUED, stage="WAITING_CODER_LEASE",
                                message="Waiting for local Coder capacity")
        pool.acquire(lease, cancelled=lambda: (control.checkpoint("WAITING_CODER_LEASE"), False)[1])
        control.update_progress(state=RunState.RUNNING, stage="OPTIMIZING",
                                message="Coder capacity acquired")
        if kernelgen:
            from kernelgen.workflows.optimization.kernelgen import KernelGenInput, KernelGenWorkflow
            fields = inp.optimization.model_dump(include=set(KernelGenInput.model_fields), mode='python')
            fields.update(definition=definition,catalog_name=snapshot.catalog_name,
                evaluation_snapshot=snapshot.model_dump(mode='json'),initial_seed_code=seed,
                initial_seed_is_validated_baseline=False,**reference)
            fields["knowledge_run_id"] = inp.optimization.knowledge_run_id or knowledge_run_id
            # Completed/unfinished Coder ledgers retain their normal recovery semantics.
            # The original seed stays fixed for epoch 1; no historical score is imported.
            if inp.resume:
                fields.update(start_mode="resume", start_epoch=control.progress().current_epoch or 1,
                              initial_seed_code="", initial_seed_is_validated_baseline=False)
            workflow = KernelGenWorkflow(cwd=str(workspace), runtime_factory=runtime_factory,
                                        run_control=control, knowledge_config=inp.optimization.knowledge_config)
            output = (workflow.finalize_completed_epoch(fields, inp.optimization.finalize_epoch)
                      if inp.optimization.finalize_epoch is not None else workflow.run(fields))
        else:
            if inp.optimization.knowledge_catalog_path is not None:
                from kernelgen.workflows.optimization.knowledge import materialize_knowledge
                materialize_knowledge(inp.optimization, definition, workspace,
                                      run_id=knowledge_run_id,
                                      benchmark_id=f"{snapshot.catalog_name}-{snapshot.catalog_api_version}")
                options["knowledge_enabled"] = True
            output = SingleCoderOptimizationWorkflow(cwd=str(workspace), runtime_factory=runtime_factory,
                                                 run_control=control, run_mode="simple_opt").run(options)
    finally:
        pool.release(lease.run_id, lease.pid, lease.process_start)
    if kernelgen:
        from kernelgen.workflows.optimization.kernelgen.epoch import confirmed_workspace_best
        path = workspace/'kernelgen_output.json'
        atomic_write_json(path,output.model_dump(mode='json'))
        files = [path]
        winner = None
        for ledger in sorted(workspace.glob('*R/agent*/.ledger.json')):
            confirmed,best = confirmed_workspace_best(ledger.parent)
            if confirmed=='PASSED' and best['code']==output.best_code and best['geo_mean']==output.best_geo_mean:
                winner = ledger.parent
                break
        if winner is None:
            return workflow_result({'workspace':str(workspace)},files,state='FAILED',message='KernelGen has no confirmed winner')
        files.extend([winner/'.ledger.json',winner/'optimize_definition_output.json',winner/'.kernelgen/final-verification.json'])
        candidate = winner/'.best_kernel.py'
        if candidate.is_file():
            files.append(candidate)
        accepted = output.status=='PASSED' and candidate.is_file() and candidate.read_text()==output.best_code
        return workflow_result({'workspace':str(workspace),'candidate':str(candidate)},files,
            state='SUCCEEDED' if accepted else 'FAILED',message=f'KernelGen optimization {output.status}')
    files = [workspace / "optimize_definition_output.json", workspace / ".ledger.json"]
    candidate = workspace / ".best_kernel.py"
    if candidate.is_file():
        files.append(candidate)
    accepted = output.status == "PASSED" and candidate.is_file() and candidate.read_text() == output.best_code
    return workflow_result({"workspace": str(workspace), "candidate": str(candidate)}, files,
                        state="SUCCEEDED" if accepted else "FAILED", message=f"Optimization {output.status}")

def code_review(inp, root, runtime_factory, context):
    optimized = receipt_data(context, "optimize")
    extracted = receipt_data(context, "prepare_catalog")
    reviewed = receipt_data(context, "review_tests")
    candidate = Path(optimized["candidate"])
    evidence = list(dict.fromkeys(
        p for p in [*optimized["artifacts"], *extracted["artifacts"], *reviewed["artifacts"]]
        if not p.endswith(".tar") and p != extracted.get("test_evidence")
    ))
    return _review(runtime_factory, context, "code", file_digest(candidate), evidence)

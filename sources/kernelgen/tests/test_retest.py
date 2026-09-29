"""Independent confirmation: real snapshots/ledger, mocked remote device boundary."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernelgen_server import Definition, SourceFile, Implementation, Workload

from kernelgen.data.ledger import Ledger
from kernelgen.data.tool_context import ToolContext
from kernelgen.framework.run_control import RunCancelled, WorkspaceRunControl
from kernelgen.tests.helpers import experiment_plan
from kernelgen.tests.helpers import round_conclusion
from kernelgen.tools import retest
from kernelgen.tools.profile_round import EvaluationBundle, write_evaluation_snapshot


def measurement(reference=2.0, candidate=1.0):
    return {'status': 'PASSED', 'geo_mean': reference / candidate,
            'server_backend': 'cuda', 'num_workloads': 2, 'num_passed': 2,
            'per_workload': [
                {'uuid': 'c0', 'phase': 'correctness', 'status': 'PASSED', 'axes': {}},
                {'uuid': 't0', 'phase': 'timing', 'status': 'PASSED', 'axes': {'N': 16},
                 'reference_latency_ms': reference, 'latency_ms': candidate, 'speedup': reference / candidate},
            ]}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    code = 'def run(x): return x\n'
    status = {'api_version': 'v6.2', 'backend': 'cuda', 'timing': 'triton',
              'server_version': '6.3.1', 'target': {'device': 'NVIDIA H100'},
              'capabilities': {'candidate_admission': {'version': 1, 'stages': ['preflight'], 'policy_sha256': 'a' * 64}}}
    context = ToolContext(definition='op', target_hardware='NVIDIA H100',
                          destination_passing_style=False, catalog_name='fixture', profile_enabled=False)
    context.write(tmp_path)
    correctness = Workload(name='c0', inputs={'x': {'type': 'scalar', 'value': 1}})
    timing = Workload(name='t0', inputs={'x': {'type': 'scalar', 'value': 2}})
    bundle = EvaluationBundle(
        kernel_code=code, definition=Definition(name='op', parameters=[{'name': 'x', 'required': True}], outputs=['output']),
        solution=Implementation(name='kernelgen-candidate-op', definition='op', language='triton',
                                entrypoint='main.py::run', sources=[SourceFile(path='main.py', content=code)]),
        workloads=[correctness, timing], correctness_workloads=[correctness], timing_workloads=[timing],
        profile_workload_uuids=['t0'], benchmark_fingerprint='fixed-plan',
    )
    original = {**measurement(), 'evaluation_service': status}
    ledger = Ledger(tmp_path)
    ledger.record_eval(original, code, experiment_plan(), definition_name='op', target_hardware='NVIDIA H100')
    snapshot = write_evaluation_snapshot(tmp_path, 1, bundle, original, context)
    ledger.attach_evaluation_snapshot(1, evaluation_fingerprint=snapshot['evaluation_fingerprint'], snapshot_path=snapshot['snapshot_path'])
    monkeypatch.setattr(retest, 'prepare_evaluation_bundle', lambda kernel, ctx: bundle)
    monkeypatch.setattr(retest.adapter, 'get_service_status', lambda url: status)
    calls = []
    outputs = [measurement(2.1)]

    def evaluate(actual, ctx, *, run_control):
        assert actual is bundle
        calls.append(actual)
        output = outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return deepcopy(output), deepcopy(status)

    monkeypatch.setattr(retest.adapter, 'evaluate_bundle', evaluate)
    return SimpleNamespace(root=tmp_path, bundle=bundle, context=context, status=status,
                           calls=calls, outputs=outputs, original=original)


def test_final_retest_records_bilateral_times_without_mutating_search(prepared):
    p = prepared
    before = (p.root / '.ledger.json').read_bytes()
    result = retest.verify_final_best(p.root)
    assert result['status'] == 'PASSED'
    assert result['geo_mean'] == pytest.approx(2.1)
    assert (p.root / '.ledger.json').read_bytes() == before
    saved = json.loads((p.root / result['attempt_path']).read_text())
    assert saved['result']['per_workload'][1]['reference_latency_ms'] == 2.1
    assert saved['result']['per_workload'][1]['latency_ms'] == 1.0
    again = retest.verify_final_best(p.root)
    assert again['status'] == 'PASSED' and again['replayed']
    assert len(p.calls) == 1


def test_reference_spike_requires_two_consecutive_independent_measurements(prepared):
    p = prepared
    p.outputs[:] = [measurement(0.1), measurement(0.11)]
    first = retest.verify_final_best(p.root)
    assert first['status'] == 'NEEDS_RETEST' and first['reason_code'] == 'TIMING_DRIFT'
    assert first['geo_mean'] is None
    second = retest.request_retest(p.root, 1, 'reference t0 changed twentyfold', ['t0'])
    assert second['status'] == 'PASSED' and second['geo_mean'] == pytest.approx(0.11)
    assert second['compared_to'] == 'attempt-0001.json'
    assert retest.verify_final_best(p.root)['geo_mean'] == pytest.approx(0.11)
    assert Ledger(p.root).history.best_geo_mean == 2.0


def test_agent_retest_does_not_replace_mandatory_final_request(prepared):
    p = prepared
    p.outputs[:] = [measurement(2.1), measurement(1.9)]
    assert retest.request_retest(p.root, 1, 'checking t0 jitter', ['t0'])['status'] == 'PASSED'
    final = retest.verify_final_best(p.root)
    assert final['geo_mean'] == pytest.approx(1.9)  # Never select the favorable earlier 2.1x.
    assert len(p.calls) == 2


def test_correctness_failure_cannot_be_erased_by_retries(prepared):
    p = prepared
    bad = measurement()
    bad['status'] = 'FAILED'
    bad['per_workload'][0]['status'] = 'INCORRECT_NUMERICAL'
    p.outputs[:] = [bad]
    assert retest.verify_final_best(p.root)['status'] == 'FAILED'
    assert retest.request_retest(p.root, 1, 'retry incorrect c0', ['c0'])['reason_code'] == 'PRIOR_CORRECTNESS_FAILURE'
    assert len(p.calls) == 1


@pytest.mark.parametrize('mutation', ['missing_timing', 'nan', 'duplicate', 'all_skip', 'changed_axes', 'new_case'])
def test_incomplete_or_different_cases_never_confirm(prepared, mutation):
    p = prepared
    bad = measurement()
    if mutation == 'missing_timing': bad['per_workload'][1]['reference_latency_ms'] = None
    if mutation == 'nan': bad['per_workload'][1]['latency_ms'] = float('nan')
    if mutation == 'duplicate': bad['per_workload'].append(deepcopy(bad['per_workload'][1]))
    if mutation == 'all_skip': bad['per_workload'][0]['status'] = 'SKIPPED'
    if mutation == 'changed_axes': bad['per_workload'][1]['axes'] = {'N': 32}
    if mutation == 'new_case': bad['per_workload'][1]['uuid'] = 'other'
    p.outputs[:] = [bad]
    assert retest.verify_final_best(p.root)['status'] == 'NEEDS_RETEST'


@pytest.mark.parametrize('mutation', ['code', 'measurement', 'context', 'policy', 'catalog', 'legacy'])
def test_snapshot_and_environment_changes_fail_before_execution(prepared, mutation):
    p = prepared
    snapshot = p.root / Ledger(p.root).get_round(1).solution.snapshot_path
    if mutation == 'code': (snapshot / 'main.py').write_text('def run(x): return 0\n')
    if mutation == 'measurement': (snapshot / 'result.json').write_text(json.dumps(measurement(99)))
    if mutation == 'context': p.context.model_copy(update={'num_trials': 5}).write(p.root)
    if mutation == 'policy': p.status['capabilities']['candidate_admission']['policy_sha256'] = 'b' * 64
    if mutation == 'catalog': p.bundle.workloads[1].name = 'changed'
    if mutation == 'legacy':
        identity = json.loads((snapshot / 'identity.json').read_text())
        identity.pop('retest_contract')
        (snapshot / 'identity.json').write_text(json.dumps(identity))
    assert retest.verify_final_best(p.root)['status'] == 'NEEDS_RETEST'
    assert not p.calls


def test_agent_evidence_must_reference_measured_workloads_and_budget_is_bounded(prepared):
    p = prepared
    assert retest.request_retest(p.root, 1, 'unknown', ['fake'])['status'] == 'NEEDS_RETEST'
    assert retest.request_retest(p.root, 1, '', ['t0'])['status'] == 'NEEDS_RETEST'
    assert not p.calls
    p.outputs[:] = [measurement(), measurement(), measurement()]
    for _ in range(2):
        assert retest.request_retest(p.root, 1, 't0 reference timing looks noisy', ['t0'])['status'] == 'PASSED'
    assert retest.request_retest(p.root, 1, 'try again', ['t0'])['reason_code'] == 'RETEST_BUDGET_EXHAUSTED'
    assert retest.verify_final_best(p.root)['status'] == 'PASSED'
    assert len(p.calls) == 3


def test_failed_attempt_does_not_fall_back_to_older_passing_retest(prepared):
    p = prepared
    p.outputs[:] = [measurement(), RuntimeError('remote unavailable')]
    assert retest.verify_final_best(p.root)['status'] == 'PASSED'
    output = p.root / 'optimize_definition_output.json'
    output.write_text(json.dumps({'status': 'PASSED', 'best_geo_mean': 2.0}))
    assert retest.request_retest(p.root, 1, 'check t0 again', ['t0'])['status'] == 'NEEDS_RETEST'
    assert json.loads(output.read_text())['best_geo_mean'] is None
    assert json.loads(output.read_text())['status'] == 'NEEDS_RETEST'
    assert retest.verify_final_best(p.root)['status'] == 'NEEDS_RETEST'


def test_cancellation_does_not_start_device_work(prepared):
    p = prepared
    WorkspaceRunControl(p.root).request_cancel('stop')
    with pytest.raises(RunCancelled): retest.verify_final_best(p.root)
    assert not p.calls


def test_interrupted_final_attempt_is_not_silently_repeated(prepared):
    p = prepared
    directory = retest._attempt_root(p.root, retest._load_snapshot(p.root, 1))
    directory.mkdir(parents=True)
    (directory / 'attempt-0001.json').write_text(json.dumps({'purpose': 'final', 'filename': 'attempt-0001.json'}))
    assert retest.verify_final_best(p.root)['reason_code'] == 'INTERRUPTED_RETEST'
    assert not p.calls


def test_resubmitting_same_code_does_not_erase_correctness_failure(prepared):
    p = prepared
    bad = measurement()
    bad['status'] = 'FAILED'
    bad['per_workload'][0]['status'] = 'INCORRECT_NUMERICAL'
    p.outputs[:] = [bad]
    assert retest.verify_final_best(p.root)['status'] == 'FAILED'
    ledger = Ledger(p.root)
    ledger.finalize_round(1, round_conclusion(1))
    result = {**measurement(3.0), 'evaluation_service': p.status}
    ledger.record_eval(result, p.bundle.kernel_code, experiment_plan(2))
    snap = write_evaluation_snapshot(p.root, 2, p.bundle, result, p.context)
    ledger.attach_evaluation_snapshot(2, evaluation_fingerprint=snap['evaluation_fingerprint'], snapshot_path=snap['snapshot_path'])
    assert ledger.history.best_round == 2
    assert retest.verify_final_best(p.root)['reason_code'] == 'PRIOR_CORRECTNESS_FAILURE'
    assert len(p.calls) == 1


def test_new_candidate_sha_requires_new_final_verification(prepared):
    from dataclasses import replace
    p = prepared
    assert retest.verify_final_best(p.root)['status'] == 'PASSED'
    ledger = Ledger(p.root)
    ledger.finalize_round(1, round_conclusion(1))
    code = 'def run(x):\n    return x\n'
    bundle = replace(p.bundle, kernel_code=code, solution=p.bundle.solution.model_copy(update={
        'sources': [SourceFile(path='main.py', content=code)],
    }))
    p.bundle = bundle
    result = {**measurement(3), 'evaluation_service': p.status}
    ledger.record_eval(result, code, experiment_plan(2))
    snap = write_evaluation_snapshot(p.root, 2, bundle, result, p.context)
    ledger.attach_evaluation_snapshot(2, evaluation_fingerprint=snap['evaluation_fingerprint'], snapshot_path=snap['snapshot_path'])
    # The fake device boundary closes over the original bundle; replace it here.
    from unittest.mock import patch
    with patch.object(retest, 'prepare_evaluation_bundle', return_value=bundle), patch.object(
        retest.adapter, 'evaluate_bundle', return_value=(measurement(2.9), p.status),
    ) as evaluate:
        assert retest.verify_final_best(p.root)['geo_mean'] == pytest.approx(2.9)
        evaluate.assert_called_once()


@pytest.mark.parametrize('status', ['TIMEOUT', 'SUSPECTED_DEVICE_ERROR'])
def test_device_incident_cannot_trigger_automatic_retry(prepared, status):
    p = prepared
    p.outputs[:] = [{'status': status}]
    assert retest.request_retest(p.root, 1, 't0 timing anomaly', ['t0'])['status'] == 'NEEDS_RETEST'
    assert retest.verify_final_best(p.root)['reason_code'] == 'DEVICE_REVIEW_REQUIRED'
    assert len(p.calls) == 1


def test_mcp_retest_exposes_only_round_reason_and_evidence(monkeypatch, tmp_path):
    from kernelgen.mcp_server import server
    schema = server.mcp._tool_manager._tools['request_retest'].parameters
    assert set(schema['properties']) == {'round_num', 'reason', 'evidence_workload_uuids'}
    monkeypatch.setattr(server, 'workspace_from_env', lambda: tmp_path)
    calls = []
    monkeypatch.setattr(server, 'request_retest_impl', lambda *args: calls.append(args) or {'status': 'PASSED'})
    assert server.request_retest(1, 't0 reference looks slow', ['t0'])['status'] == 'PASSED'
    assert calls == [(tmp_path, 1, 't0 reference looks slow', ['t0'])]


@pytest.mark.parametrize('reference, expected_status', [(2.1, 'PASSED'), (0.1, 'NEEDS_RETEST')])
def test_workflow_requires_confirmation_before_output_and_distillation(prepared, monkeypatch, reference, expected_status):
    from kernelgen.workflows.optimization.single_coder import workflow as module
    p = prepared
    p.outputs[:] = [measurement(reference)]
    monkeypatch.setattr(module, 'run_coder_loop', lambda *a, **kw: SimpleNamespace(summary='search completed'))
    distilled = []
    monkeypatch.setattr(module, 'run_distillation', lambda *a, **kw: distilled.append(kw['verified_geo_mean']))
    flow = module.SingleCoderOptimizationWorkflow(cwd=str(p.root), runtime_factory=lambda cwd: object())
    monkeypatch.setattr(flow, '_resolve_target_context', lambda inp: SimpleNamespace(device='NVIDIA H100'))
    monkeypatch.setattr(flow, '_build_tool_context', lambda *a, **kw: p.context)
    inp = module.SingleCoderOptimizationInput(
        definition={'name': 'op', 'op_type': 'elementwise', 'axes': {}, 'inputs': {}, 'outputs': {}, 'reference': 'def run(x): return x'},
        destination_passing_style=False, target_hardware='NVIDIA H100', catalog_name='fixture', profile_enabled=False,
    )
    output = flow.run(inp)
    assert output.status == expected_status
    assert output.search_best_geo_mean == 2.0
    assert output.best_geo_mean == (pytest.approx(reference) if expected_status == 'PASSED' else None)
    assert len(distilled) == (1 if expected_status == 'PASSED' else 0)
    saved = json.loads((p.root / 'optimize_definition_output.json').read_text())
    assert saved['final_verification']['status'] == expected_status
    assert Ledger(p.root).history.best_geo_mean == 2.0


@pytest.mark.parametrize('mutation', [None, 'missing', 'pending', 'wrong_sha', 'wrong_score'])
def test_epoch_selection_requires_confirmation_and_uses_verified_score(prepared, mutation):
    from kernelgen.workflows.optimization.kernelgen.epoch import collect_epoch_result
    p = prepared
    p.outputs[:] = [measurement(1.1)]
    verification = retest.verify_final_best(p.root)
    assert verification['status'] == 'PASSED'
    if mutation == 'pending': verification.update(status='NEEDS_RETEST', geo_mean=None)
    if mutation == 'wrong_sha': verification['solution_sha256'] = '0' * 64
    summary = p.root / '.kernelgen/final-verification.json'
    summary.write_text(json.dumps(verification))
    output = {'status': verification['status'], 'best_geo_mean': verification['geo_mean'],
              'best_code': Ledger(p.root).best['code'], 'final_verification': verification}
    if mutation == 'wrong_score': output['best_geo_mean'] = 99
    (p.root / 'optimize_definition_output.json').write_text(json.dumps(output))
    if mutation == 'missing': summary.unlink()
    result = collect_epoch_result('op', [(SimpleNamespace(status='PASSED'), 'agent0')],
                                  SimpleNamespace(path_of=lambda _: p.root))
    assert result.best_geo_mean == (pytest.approx(1.1) if mutation is None else None)
    assert result.status == ('PASSED' if mutation is None else 'FAILED')
    assert Ledger(p.root).history.best_geo_mean == 2.0

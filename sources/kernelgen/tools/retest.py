"""Independent remeasurement of immutable rounds, outside the search ledger.

The model can request execution, never supply timings, settings or a verdict.
KGS evaluates the complete bound suite in its ordinary fresh isolated worker.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from kernelgen.data.ledger import Ledger
from kernelgen.data.tool_context import load_tool_context
from kernelgen.framework.run_control import RunCancelled, WorkspaceRunControl
from kernelgen.tools import kernelgen_server_adapter as adapter
from kernelgen.tools.profile_round import _atomic_json, _load_snapshot, prepare_evaluation_bundle


MAX_AGENT_RETESTS = 2
DRIFT_FACTOR = 2.0
RETEST_DIR = Path('.kernelgen/retests')


def service_identity(status: dict) -> dict:
    """Ignore live queue counters, retain exposed runtime/configuration identity."""
    return {**adapter.service_signature(status), **{
        key: status.get(key) for key in ('server_version', 'software', 'target', 'devices')
    }}


def retest_contract(bundle, context, status: dict) -> dict:
    from kernelgen.tools.preflight import _bundle_identity
    return {
        'bundle': _bundle_identity(bundle, context),
        'settings': adapter._settings_from_context(context).model_dump(mode='json'),
        'service': service_identity(status),
    }


def _positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _measurements(result):
    rows = result.get('per_workload', [])
    keyed = {(r.get('phase'), r.get('uuid')): r for r in rows}
    if len(keyed) != len(rows) or not rows:
        raise ValueError('missing or duplicate workload identities')
    accuracy = [r for r in rows if r.get('phase') == 'correctness']
    timing = [r for r in rows if r.get('phase') == 'timing']
    if not any(r.get('status') == 'PASSED' for r in accuracy):
        raise ValueError('no executed passing correctness workload')
    if not timing or any(r.get('status') not in {'PASSED', 'SKIPPED'} for r in rows):
        raise ValueError('incomplete correctness or timing workloads')
    timed = [r for r in timing if r.get('status') == 'PASSED']
    if not timed or any(not _positive(r.get(field)) for r in timed
                        for field in ('reference_latency_ms', 'latency_ms')):
        raise ValueError('missing finite positive bilateral timing')
    return keyed, timed


def compare_measurements(original: dict, measured: dict) -> dict:
    """Report both sides; never use the old headline or select favorable points."""
    if measured.get('is_hack') or any(
        r.get('status') in {'INCORRECT', 'INCORRECT_NUMERICAL'} for r in measured.get('per_workload', [])
    ):
        return {'status': 'FAILED', 'reason_code': 'CORRECTNESS_OR_ADMISSION_FAILED', 'geo_mean': None}
    if measured.get('status') != 'PASSED':
        return {'status': 'NEEDS_RETEST', 'reason_code': 'EVALUATION_NOT_PASSED', 'geo_mean': None}
    try:
        for key in ('api_version', 'server_backend', 'reference_source', 'benchmark_fingerprint'):
            if original.get(key) != measured.get(key):
                raise ValueError('measurement ' + key + ' changed')
        before, _ = _measurements(original)
        after, timed = _measurements(measured)
        if before.keys() != after.keys() or any(
            before[k].get('axes', {}) != after[k].get('axes', {})
            or before[k].get('status') != after[k].get('status') for k in before
        ):
            raise ValueError('workload identities, parameters or skip outcomes changed')
    except (ValueError, TypeError) as exc:
        return {'status': 'NEEDS_RETEST', 'reason_code': 'INCOMPARABLE_MEASUREMENT', 'detail': str(exc), 'geo_mean': None}
    comparisons, drift = [], []
    for row in timed:
        old = before[(row['phase'], row['uuid'])]
        ratios = {field: row[field] / old[field] for field in ('reference_latency_ms', 'latency_ms')}
        comparisons.append({'uuid': row['uuid'], 'axes': row.get('axes', {}),
                            'before': {f: old[f] for f in ratios},
                            'after': {f: row[f] for f in ratios}, 'ratios': ratios})
        for field, ratio in ratios.items():
            if ratio > DRIFT_FACTOR or ratio < 1 / DRIFT_FACTOR:
                drift.append({'uuid': row['uuid'], 'side': field, 'ratio': ratio})
    geo = math.exp(sum(math.log(r['reference_latency_ms']) - math.log(r['latency_ms']) for r in timed) / len(timed))
    return {'status': 'NEEDS_RETEST' if drift else 'PASSED',
            'reason_code': 'TIMING_DRIFT' if drift else 'VERIFIED',
            'geo_mean': geo if not drift else None, 'measured_geo_mean': geo,
            'comparisons': comparisons, 'drift': drift, 'drift_factor': DRIFT_FACTOR}


def _prepared(workspace, round_num):
    from kernelgen.tools.preflight import _canonical_sha256
    context = load_tool_context(workspace)
    snapshot = _load_snapshot(workspace, round_num)
    record = snapshot['record']
    if record.evaluation.status != 'PASSED' or record.evaluation.is_hack:
        raise ValueError('retest requires an existing passing measured round')
    kernel = snapshot['path'] / 'main.py'
    if hashlib.sha256(kernel.read_bytes()).hexdigest() != record.solution.sha256:
        raise ValueError('immutable candidate hash mismatch')
    identity = snapshot['identity']
    if identity['evaluation_fingerprint'] != record.evaluation.fingerprint:
        raise ValueError('evaluation fingerprint mismatch')
    contract = identity.get('retest_contract')
    if not contract or not contract.get('service', {}).get('api_version'):
        raise ValueError('legacy snapshot lacks frozen remeasurement settings/environment')
    for key, value in [('solution', snapshot['solution']), ('definition', snapshot['definition'])]:
        if _canonical_sha256(value) != identity[key + '_sha256']:
            raise ValueError('snapshot ' + key + ' changed')
    if [_canonical_sha256(w) for w in snapshot['workloads']] != identity['workload_sha256']:
        raise ValueError('snapshot workloads changed')
    if _canonical_sha256(snapshot['result']) != identity.get('result_sha256'):
        raise ValueError('snapshot measurement changed')
    bundle = prepare_evaluation_bundle(kernel, context)
    status = adapter.get_service_status(context.eval_server_url)
    adapter.require_target_context(context, status)
    adapter.require_candidate_admission(status)
    if retest_contract(bundle, context, status) != contract:
        raise ValueError('candidate, suite, evaluation settings or environment changed')
    return snapshot, bundle, context, status


def _attempt_root(root, snapshot):
    from kernelgen.tools.preflight import _canonical_sha256
    # Re-submitting identical code as a new search round must not reset its
    # retest budget or erase an observed correctness failure.
    contract_sha = _canonical_sha256(snapshot['identity']['retest_contract'])
    return root / RETEST_DIR / snapshot['record'].solution.sha256 / contract_sha


def _invalidate_output(root, record, attempt_path):
    """Do not leave a prior successful export active while its best is retested."""
    ledger = Ledger(root)
    if ledger.history.best_round <= 0 or ledger.get_round(ledger.history.best_round).solution.sha256 != record.solution.sha256:
        return
    path = root / 'optimize_definition_output.json'
    if path.is_file():
        output = json.loads(path.read_text())
        pending = {'status': 'NEEDS_RETEST', 'reason_code': 'NEW_RETEST_PENDING',
                   'geo_mean': None, 'attempt_path': attempt_path}
        output.update(status='NEEDS_RETEST', best_geo_mean=None, final_verification=pending)
        _atomic_json(path, output)
        _atomic_json(root / '.kernelgen/final-verification.json', pending)


def _run(workspace, round_num, *, reason, evidence_workload_uuids, final, run_control):
    root = Path(workspace).resolve()
    control = run_control or WorkspaceRunControl(root, source='mcp:request_retest')
    control.checkpoint('BEFORE_RETEST')
    snapshot, bundle, context, status = _prepared(root, round_num)
    known = {r['uuid'] for r in snapshot['result'].get('per_workload', [])}
    if not final and (not reason.strip() or not evidence_workload_uuids or not set(evidence_workload_uuids) <= known):
        raise ValueError('agent retest requires a reason and existing evidence workload UUIDs')
    record = snapshot['record']
    base = _attempt_root(root, snapshot)
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'NEEDS_RETEST', 'reason_code': 'RETEST_IN_PROGRESS', 'geo_mean': None}
        attempts = [json.loads(p.read_text()) for p in sorted(base.glob('attempt-*.json'))]
        if any(a.get('result', {}).get('status') in {'TIMEOUT', 'SUSPECTED_DEVICE_ERROR'} for a in attempts):
            return {'status': 'NEEDS_RETEST', 'reason_code': 'DEVICE_REVIEW_REQUIRED', 'geo_mean': None}
        # A failed correctness check cannot be washed out by repeated runs of the same code.
        if any(a.get('verdict', {}).get('status') == 'FAILED' for a in attempts):
            return {'status': 'FAILED', 'reason_code': 'PRIOR_CORRECTNESS_FAILURE', 'geo_mean': None}
        if final and any(a['purpose'] == 'final' for a in attempts):
            latest = attempts[-1]
            return {**latest.get('verdict', {'status': 'NEEDS_RETEST', 'reason_code': 'INTERRUPTED_RETEST', 'geo_mean': None}),
                    'attempt_path': str((base / latest['filename']).relative_to(root)), 'replayed': True,
                    'round_num': round_num, 'solution_sha256': record.solution.sha256}
        if not final and sum(a['purpose'] == 'agent' for a in attempts) >= MAX_AGENT_RETESTS:
            return {'status': 'NEEDS_RETEST', 'reason_code': 'RETEST_BUDGET_EXHAUSTED', 'geo_mean': None}
        filename = f'attempt-{len(attempts) + 1:04d}.json'
        path = base / filename
        evidence = {
            'schema_version': 1, 'filename': filename, 'purpose': 'final' if final else 'agent',
            'round_num': round_num, 'solution_sha256': record.solution.sha256,
            'evaluation_fingerprint': record.evaluation.fingerprint,
            'contract': snapshot['identity']['retest_contract'],
            'reason': reason, 'evidence_workload_uuids': evidence_workload_uuids,
            'requested_at': datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(path, evidence)  # Reserve the attempt before remote execution; interruptions remain visible.
        _invalidate_output(root, record, str(path.relative_to(root)))
        control.record_event('RETEST_STARTED', stage='RETESTING', data={
            'round_num': round_num, 'attempt_path': str(path.relative_to(root)), 'purpose': evidence['purpose'],
        })
        try:
            measured, service = adapter.evaluate_bundle(bundle, context, run_control=control)
            evidence['result'] = measured
            evidence['service'] = service_identity(service)
            control.checkpoint('AFTER_RETEST')
            # Re-inspect the catalog/settings too, so a mutation during the request cannot silently pass.
            _prepared(root, round_num)
            if service_identity(service) != service_identity(status):
                raise ValueError('evaluation service changed during retest')
            previous = snapshot['result']
            compared_to = 'original_round'
            if attempts and attempts[-1].get('result', {}).get('status') == 'PASSED':
                previous = attempts[-1]['result']
                compared_to = attempts[-1]['filename']
            verdict = compare_measurements(previous, measured)
            verdict['compared_to'] = compared_to
            evidence['comparison_to_original'] = compare_measurements(snapshot['result'], measured)
            # Even after two stable retests, missing/changing cases must not replace the original suite.
            if evidence['comparison_to_original']['reason_code'] == 'INCOMPARABLE_MEASUREMENT':
                verdict = evidence['comparison_to_original']
            if measured.get('status') in {'TIMEOUT', 'SUSPECTED_DEVICE_ERROR'}:
                evidence['scheduler_after'] = adapter.get_service_status(context.eval_server_url).get('scheduler')
        except RunCancelled:
            evidence['verdict'] = {'status': 'NEEDS_RETEST', 'reason_code': 'RUN_CANCELLED', 'geo_mean': None}
            _atomic_json(path, evidence)
            raise
        except Exception as exc:
            verdict = {'status': 'NEEDS_RETEST', 'reason_code': 'RETEST_EXECUTION_ERROR',
                       'detail': f'{type(exc).__name__}: {exc}', 'geo_mean': None}
        evidence['verdict'] = verdict
        evidence['completed_at'] = datetime.now(timezone.utc).isoformat()
        _atomic_json(path, evidence)
        control.record_event('RETEST_COMPLETED', stage='RETESTING', data={
            'round_num': round_num, 'status': verdict['status'], 'attempt_path': str(path.relative_to(root)),
        })
        return {**verdict, 'attempt_path': str(path.relative_to(root)), 'round_num': round_num,
                'solution_sha256': record.solution.sha256}


def request_retest(workspace, round_num: int, reason: str, evidence_workload_uuids: list[str], *, run_control=None):
    """Agent-requested full-suite remeasurement; never adds a search round."""
    try:
        return _run(workspace, round_num, reason=reason, evidence_workload_uuids=evidence_workload_uuids,
                    final=False, run_control=run_control)
    except RunCancelled:
        raise
    except Exception as exc:
        return {'status': 'NEEDS_RETEST', 'reason_code': 'RETEST_PREPARATION_FAILED',
                'detail': str(exc), 'geo_mean': None}


def verify_final_best(workspace, *, run_control=None):
    """Mandatory final validation. A new winning round gets its own verification."""
    ledger = Ledger(workspace)
    best_round = ledger.history.best_round
    if best_round <= 0:
        return {'status': 'FAILED', 'reason_code': 'NO_PASSING_BEST', 'geo_mean': None}
    if ledger.history.rounds[-1].evaluation.status in {'TIMEOUT', 'SUSPECTED_DEVICE_ERROR'}:
        return {'status': 'NEEDS_RETEST', 'reason_code': 'DEVICE_REVIEW_REQUIRED', 'geo_mean': None}
    try:
        result = _run(workspace, best_round, reason='Mandatory final best verification',
                      evidence_workload_uuids=[], final=True, run_control=run_control)
        if Ledger(workspace).history.best_round != best_round:
            return {'status': 'NEEDS_RETEST', 'reason_code': 'BEST_CHANGED_DURING_RETEST', 'geo_mean': None}
        return result
    except RunCancelled:
        raise
    except Exception as exc:
        return {'status': 'NEEDS_RETEST', 'reason_code': 'RETEST_PREPARATION_FAILED',
                'detail': str(exc), 'geo_mean': None}

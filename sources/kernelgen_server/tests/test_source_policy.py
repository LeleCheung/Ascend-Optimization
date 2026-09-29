import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from kernelgen_server.protocol.schema import SourceCondition, Workload, WorkloadResult
from kernelgen_server.runtime.source_policy import policy_snapshot, source_flag, source_policy_scope, skip_reason
from kernelgen_server.evaluation.result import aggregate_workload_results
from kernelgen_server.runtime.source_policy import source_vendor


POLICY = 'flaggems/d64794e63b502cb836bc015a92a62c42de4be05a'


def test_source_vendor_uses_bound_logical_backend_and_requires_policy():
    for backend, vendor in [('npu', 'ascend'), ('mlu', 'cambricon'), ('metax', 'metax')]:
        with source_policy_scope(backend, POLICY):
            assert source_vendor() == vendor
    with pytest.raises(ValueError, match='undeclared'):
        source_vendor()
    with source_policy_scope('unknown', POLICY):
        with pytest.raises(ValueError, match='unknown'):
            source_vendor()


def test_known_unknown_and_explicit_policy_identity():
    for backend, expected in [('cuda', True), ('npu', False), ('musa', False), ('metax', True)]:
        with source_policy_scope(backend, POLICY):
            assert source_flag('support_fp64') is expected
    with source_policy_scope('cuda', 'other-revision'):
        with pytest.raises(ValueError, match='unsupported or undeclared source policy'):
            source_flag('support_fp64')
    assert policy_snapshot('unknown')['flags'] is None
    with pytest.raises(ValueError, match='undeclared source policy'):
        source_flag('support_fp64')


def test_nested_scopes_and_threads_do_not_leak():
    def value(backend):
        with source_policy_scope(backend, POLICY):
            return source_flag('support_fp64')
    with source_policy_scope('cuda', POLICY):
        assert value('npu') is False
        assert source_flag('support_fp64') is True
        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(value, ['npu', 'cuda'])) == [False, True]


def test_conditional_workload_round_trip_and_no_unknown_skip():
    workload = Workload(name='fp64', inputs={}, source_condition=SourceCondition(
        flags={'support_fp64': True}))
    assert Workload.model_validate_json(workload.model_dump_json()) == workload
    with source_policy_scope('npu', POLICY):
        assert 'support_fp64=False' in skip_reason(workload)
    with source_policy_scope('cuda', POLICY):
        assert skip_reason(workload) is None
    with source_policy_scope('unknown', POLICY):
        with pytest.raises(ValueError, match='unknown source policy'):
            skip_reason(workload)
    assert skip_reason(Workload(name='ordinary', inputs={})) is None
    with pytest.raises(ValueError):
        SourceCondition(flags={'made_up': True})


@pytest.mark.parametrize('correctness,expected', [
    (['PASSED', 'SKIP'], 'PASSED'), (['SKIP', 'SKIP'], 'ALL_SKIP'),
    (['RUNTIME_ERROR', 'SKIP'], 'PARTIAL_PASS'),
])
def test_skip_aggregation_preserves_timing_and_does_not_count_skip_as_pass(correctness, expected):
    results = [WorkloadResult(uuid=str(i), phase='correctness', status=status, log='source condition')
               for i, status in enumerate(correctness)]
    results.append(WorkloadResult(uuid='timing', phase='timing', status='PASSED',
                                  latency_ms=1, reference_latency_ms=2, speedup=2))
    response = aggregate_workload_results(results, device='cpu', backend='npu')
    assert response.status.value == expected
    assert response.num_passed == correctness.count('PASSED') + 1
    assert response.num_workloads == 3
    assert response.per_workload[-1].speedup == 2
    if expected == 'PASSED':
        assert response.geo_mean == 2
    else:
        assert response.geo_mean is None


def test_source_config_is_serializable_without_gems_import():
    snapshot = policy_snapshot('enflame')
    assert snapshot['flags']['support_int64'] is False
    assert json.loads(json.dumps(snapshot)) == snapshot
    snapshot['flags']['support_int64'] = True
    assert policy_snapshot('enflame')['flags']['support_int64'] is False


def test_all_timing_skipped_does_not_claim_measured_pass():
    response = aggregate_workload_results([
        WorkloadResult(uuid='correctness', phase='correctness', status='PASSED'),
        WorkloadResult(uuid='timing', phase='timing', status='SKIP', log='source condition'),
    ], device='cpu', backend='npu')
    assert response.status.value == 'ALL_SKIP'
    assert response.geo_mean is None

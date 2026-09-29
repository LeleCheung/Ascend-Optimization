import pytest
from types import SimpleNamespace

pytest.importorskip('torch')

from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.protocol.schema import Definition, EvaluateRequest, Implementation, SourceFile, Workload
from kernelgen_server.runtime.source_policy import source_flag


POLICY = 'flaggems/d64794e63b502cb836bc015a92a62c42de4be05a'


class CpuDevice:
    def set_device(self, _): pass
    def synchronize(self, _): pass
    def empty_cache(self): pass
    def time(self, fn, args, *rest):
        fn(*args)
        return 1.0


def request():
    oracle = f'''
from kernelgen_server.runtime.source_policy import source_flag
def run(x):
    assert source_flag("support_fp64") is False
    return x
'''
    definition = Definition(api_version='v6.2', name='identity',
                            parameters=[{'name': 'x', 'required': True}], outputs=['out'], reference=oracle,
                            source_policy_id=POLICY)
    implementation = Implementation(name='candidate', definition='identity', language='python',
                                    entrypoint='main.py::run', sources=[SourceFile(
                                        path='main.py', content='def run(x): return x')])
    ordinary = Workload(name='ordinary', inputs={'x': {'type': 'scalar', 'value': 1}})
    conditional = Workload(name='fp64', inputs={'x': {'type': 'random', 'shape': [1], 'dtype': 'invalid_dtype'}},
                          source_condition={'flags': {'support_fp64': True}})
    return EvaluateRequest(api_version='v6.2', definition=definition, implementation=implementation,
                           correctness_workloads=[ordinary, conditional],
                           timing_workloads=[ordinary.model_copy(update={'name': 'timing'})],
                           settings={'num_trials': 1, 'warmup_ms': 0, 'benchmark_ms': 1})


def test_skips_before_allocation_and_runs_baseline_with_bound_policy():
    result = EvaluationEngine(CpuDevice(), 'cpu', 'npu').evaluate(request())
    assert result.status.value == 'PASSED', result.model_dump()
    assert [w.status.value for w in result.per_workload] == ['PASSED', 'SKIP', 'PASSED']
    assert result.geo_mean == 1
    assert 'support_fp64' in result.per_workload[1].log
    with pytest.raises(ValueError, match='undeclared source policy'):
        source_flag('support_fp64')


def test_all_correctness_skip_still_runs_benchmark():
    req = request()
    req.correctness_workloads = req.correctness_workloads[1:]
    result = EvaluationEngine(CpuDevice(), 'cpu', 'npu').evaluate(req)
    assert result.status.value == 'ALL_SKIP', result.model_dump()
    assert result.per_workload[-1].speedup == 1


def test_valid_receives_tuple_elements_even_with_one_declared_output():
    req = request()
    req.definition.reference = '''
def run(x): return (x, x + 1)
def valid(ref_outputs, sol_outputs, inputs, ctx):
    assert len(ref_outputs) == len(sol_outputs) == 2
    assert ref_outputs == sol_outputs == [inputs['x'], inputs['x'] + 1]
    return True
'''
    req.implementation.sources[0].content = 'def run(x): return (x, x + 1)'
    assert req.definition.outputs == ['out']
    result = EvaluationEngine(CpuDevice(), 'cpu', 'npu').evaluate(req)
    assert result.status.value == 'PASSED', result.model_dump()


def test_unknown_backend_is_not_a_skip_or_success():
    result = EvaluationEngine(CpuDevice(), 'cpu', 'unknown').evaluate(request())
    assert result.status.value == 'RUNTIME_ERROR'
    assert 'unknown source policy' in result.log


def test_false_condition_does_not_hide_an_applicable_runtime_error():
    req = request()
    req.correctness_workloads[1].source_condition.flags['support_fp64'] = False
    result = EvaluationEngine(CpuDevice(), 'cpu', 'npu').evaluate(req)
    assert result.status.value != 'PASSED'
    assert not any(w.status.value == 'SKIP' for w in result.per_workload)


def test_preflight_skips_inapplicable_timing_before_allocation(monkeypatch):
    import kernelgen_server.evaluation.engine as engine
    monkeypatch.setattr(engine, 'check_candidate_admission', lambda *a, **k:
                        SimpleNamespace(is_hack=False, hack_reason='', log=''))
    monkeypatch.setattr(engine, 'log_graylist_hits', lambda *a, **k: None)
    req = request()
    conditional = req.correctness_workloads.pop()
    req.timing_workloads.append(conditional)
    result = EvaluationEngine(CpuDevice(), 'cpu', 'npu').preflight(req)
    assert result['status'] == 'PASSED'
    assert result['per_workload']['fp64']['status'] == 'SKIP'
    assert 'support_fp64' in result['per_workload']['fp64']['reason']


@pytest.mark.parametrize('conditional', [False, True])
def test_profile_main_binds_policy_and_rejects_inapplicable_case(tmp_path, monkeypatch, conditional):
    import kernelgen_server.profiling.runner as runner
    req = request()
    target = SimpleNamespace(expected_backend='npu', definition=req.definition,
                             workload=req.correctness_workloads[1 if conditional else 0])
    (tmp_path / 'target.json').write_text('{}')
    monkeypatch.setattr(runner.ProfileTarget, 'model_validate_json', lambda *args: target)
    monkeypatch.setattr(runner, 'configure_device', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'get_device', lambda _: CpuDevice())
    monkeypatch.setattr('sys.argv', ['runner', '--data-dir', str(tmp_path), '--device', 'cpu',
                                   '--implementation-source-root', str(tmp_path)])
    observed = []
    monkeypatch.setattr(runner, '_run_profile', lambda *args: observed.append(source_flag('support_fp64')))
    if conditional:
        with pytest.raises(ValueError, match='inapplicable'):
            runner.main()
        assert observed == []
    else:
        runner.main()
        assert observed == [False]
    with pytest.raises(ValueError, match='undeclared'):
        source_flag('support_fp64')

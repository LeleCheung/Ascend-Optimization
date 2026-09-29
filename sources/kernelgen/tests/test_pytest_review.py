"""Behavioral checks for native pytest review; CPU fixtures do not model GPU timing."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / '.kernelgen/skills/pytest-review/scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


@pytest.mark.parametrize('source', [
    'import torch as t\nt.add = lambda x: x',
    'from torch import ops as o\no.aten.foo = lambda x: x',
    'import flag_gems as g\nsetattr(g, "linear_backward", lambda x: x)',
    'import triton\ndel triton.runtime.jit.JITFunction.run',
    'import pytest\npytest.skip = lambda: None',
])
def test_source_flags_protected_writes(tmp_path, source):
    p = tmp_path / 'candidate.py'
    p.write_text(source)
    result = load_script('review_source').scan(p, candidate=True)
    assert any(f['code'] == 'PROTECTED_STATE_WRITE' for f in result['findings'])


def test_source_allows_ordinary_tensor_calls_and_labels_gems_baseline(tmp_path):
    p = tmp_path / 'candidate.py'
    p.write_text('import torch\ndef run(x):\n    return torch.empty_like(x)\n')
    assert load_script('review_source').scan(p, candidate=True)['findings'] == []
    p.write_text('import flag_gems\nbench(torch_op=flag_gems.linear_backward)\n')
    assert load_script('review_source').scan(p)['findings'][0]['code'] == 'GEMS_BASELINE_REQUIRES_ISOLATION'


BASE = '''
from types import SimpleNamespace
class Benchmark:
    def __init__(self, op, candidate, count=2):
        self.torch_op, self.gems_op, self.count = op, candidate, count
    def get_latency(self, op, *args, **kwargs):
        op(*args, **kwargs)
        return 0.1
    def run(self):
        for i in range(self.count):
            metric = SimpleNamespace()
            metric.latency_base = self.get_latency(self.torch_op, i)
            metric.latency = self.get_latency(self.gems_op, i)
'''


def run_probe(tmp_path, source, *, phase='candidate', benchmark=False, base=BASE, expected=None, extra=(), reference_text=None, accuracy_source=None):
    gems = tmp_path / 'gems'
    gems.mkdir(exist_ok=True)
    write(gems / 'conftest.py', '''
        def pytest_addoption(parser):
            parser.addoption('--level', default='core')
    ''')
    write(gems / 'candidate.py', '''
        def run(*args, **kwargs):
            return args[0] if args else None
    ''')
    write(gems / 'test_op.py', source)
    write(gems / 'base.py', base)
    output = tmp_path / 'report.json'
    argv = [sys.executable, str(SCRIPTS / 'probe_pytest.py'), '--gems-root', str(gems),
            '--candidate', str(gems / 'candidate.py'), '--phase', phase, '--output', str(output)]
    if benchmark:
        argv += ['--benchmark-file', str(gems / 'base.py')]
    if phase == 'reference-as-solution':
        write(gems / 'flag_gems/__init__.py', '# Fixture for target framework identity.\n')
        (gems / 'test_op.py').write_text('import flag_gems\n' + (gems / 'test_op.py').read_text())
        reference = tmp_path / 'reference.py'
        reference.write_bytes((gems / 'candidate.py').read_bytes() if reference_text is None else reference_text.encode())
        argv += ['--reference-source', str(reference)]
    if expected is not None:
        expectation = tmp_path / 'expected.json'
        expectation.write_text(json.dumps(expected))
        argv += ['--expect', str(expectation)]
    selections = ['test_op.py']
    if accuracy_source is not None:
        write(gems / 'tests/test_accuracy.py', accuracy_source)
        write(gems / 'benchmark/test_bench.py', (gems / 'test_op.py').read_text())
        selections = ['tests/test_accuracy.py', 'benchmark/test_bench.py']
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONPATH=str(gems))
    completed = subprocess.run(argv + ['--', *selections, '-q', *extra], env=env, text=True, capture_output=True, timeout=40)
    assert output.exists(), completed.stdout + completed.stderr
    return completed, json.loads(output.read_text())


def codes(report):
    return {f['code'] for f in report['findings']}


def test_reference_solution_ready_with_advisory_coverage_and_legal_skip(tmp_path):
    proc, report = run_probe(tmp_path, '''
        import pytest
        from candidate import run
        def test_op(): assert run(1) == 1
        def test_unsupported(): pytest.skip('unsupported dtype')
    ''', phase='reference-as-solution')
    assert proc.returncode == 0
    assert report['status'] == 'READY'
    assert report['problems'] == []
    assert report['suggestions']
    assert report['reference_sha256'] == report['identity']['candidate_sha256']


@pytest.mark.parametrize('source,signal', [
    ('def test_op(): pass', 'CANDIDATE_NOT_INJECTED'),
    ('def test_op(): flag_gems.missing_api()', 'API_UNAVAILABLE'),
    ('def test_op(): raise NotImplementedError("missing backend API")', 'API_UNAVAILABLE'),
    ('class CompilationError(Exception): pass\ndef test_op(): raise CompilationError("invalid IR")', 'COMPILATION_FAILED'),
])
def test_reference_solution_stops_with_evidence_for_preparation_failure(tmp_path, source, signal):
    proc, report = run_probe(tmp_path, source, phase='reference-as-solution')
    assert proc.returncode == 1 and report['status'] == 'NEEDS_FIX'
    assert any(p['signal'] == signal and p['evidence'] for p in report['problems'])


def test_reference_solution_assertion_is_not_automatically_reference_bug(tmp_path):
    _, report = run_probe(tmp_path, 'from candidate import run\ndef test_op(): assert run(1) == 2', phase='reference-as-solution')
    assert report['status'] == 'NEEDS_FIX'
    assert not any(p['signal'] == 'REFERENCE_SEMANTICS_INVALID' for p in report['problems'])


def test_reference_solution_all_skip_is_not_ready(tmp_path):
    _, report = run_probe(tmp_path, 'import pytest\ndef test_op(): pytest.skip("unsupported")', phase='reference-as-solution')
    assert report['status'] == 'NEEDS_FIX'
    assert 'NO_EXECUTED_TESTS' in codes(report)


def test_reference_solution_self_baseline_never_becomes_ready(tmp_path):
    _, report = run_probe(tmp_path, 'from candidate import run\nfrom base import Benchmark\ndef test_op(): Benchmark(run, run).run()',
                          phase='reference-as-solution', benchmark=True)
    assert report['status'] == 'NEEDS_FIX'
    assert 'BASELINE_CALLED_CANDIDATE' in codes(report)


def test_reference_solution_valid_benchmark_needs_no_separate_expectation(tmp_path):
    proc, report = run_probe(tmp_path, 'from candidate import run\nfrom base import Benchmark\ndef test_op(): Benchmark(lambda x: x, run).run()',
                            phase='reference-as-solution', benchmark=True)
    assert report['status'] == 'READY' and proc.returncode == 0
    assert report['argv'][-2:] == ['--level', 'core']


@pytest.mark.parametrize('execute_accuracy', [False, True])
def test_mixed_native_run_requires_accuracy_execution_not_all_workloads(tmp_path, execute_accuracy):
    accuracy = 'import pytest\nfrom candidate import run\ndef test_skipped(): pytest.skip("unsupported vendor")\n'
    if execute_accuracy:
        accuracy += 'def test_supported(): assert run(1) == 1\n'
    proc, report = run_probe(
        tmp_path,
        'from candidate import run\nfrom base import Benchmark\ndef test_bench(): Benchmark(lambda x: x, run).run()',
        phase='reference-as-solution', benchmark=True, accuracy_source=accuracy,
    )
    assert report['exit_code'] == 0
    assert report['status'] == ('READY' if execute_accuracy else 'NEEDS_FIX')
    assert proc.returncode == (0 if execute_accuracy else 1)
    missing = [p for p in report['problems'] if p['evidence']['code'] == 'NO_EXECUTED_TESTS']
    if execute_accuracy:
        assert not missing
        assert any(s['code'].startswith('UNREVIEWED_SKIP:') for s in report['suggestions'])
    else:
        assert missing[0]['evidence']['detail']['phase'] == 'accuracy'


def test_reference_source_mismatch_stops_before_pytest_import(tmp_path):
    proc, report = run_probe(tmp_path, 'raise RuntimeError("must not import")',
                            phase='reference-as-solution', reference_text='def run(x): return 999\n')
    assert proc.returncode == 1 and report['status'] == 'NEEDS_FIX'
    assert not report['execution_started'] and report['exit_code'] is None
    assert report['schema'] == 'pytest-review-readiness/v1'
    assert report['problems'][0]['evidence']['code'] == 'REFERENCE_SOLUTION_MISMATCH'


def test_missing_reference_dependency_during_collection_has_api_signal(tmp_path):
    _, report = run_probe(tmp_path, 'import nonexistent_kg_reference_dependency\ndef test_op(): pass', phase='reference-as-solution')
    assert report['status'] == 'NEEDS_FIX'
    assert any(p['signal'] == 'API_UNAVAILABLE' and p['evidence']['stage'] == 'collection' for p in report['problems'])


def test_each_timing_workload_is_observed_and_core_is_fixed(tmp_path):
    proc, report = run_probe(tmp_path, '''
        from base import Benchmark
        from candidate import run
        def test_op():
            Benchmark(lambda x: x, run).run()
    ''', benchmark=True)
    assert proc.returncode == 2  # unproven input independence, never a semantic pass
    assert not report['findings']
    assert len(report['measurements']) == 4
    assert len({m['case'] for m in report['measurements']}) == 2
    assert [m['candidate_calls'] for m in report['measurements']] == [0, 1, 0, 1]
    assert report['argv'][-2:] == ['--level', 'core']


def test_self_comparison_detected_even_when_native_pytest_passes(tmp_path):
    proc, report = run_probe(tmp_path, '''
        from base import Benchmark
        from candidate import run
        def test_op():
            Benchmark(run, run).run()
    ''', benchmark=True)
    assert report['exit_code'] == 0
    assert proc.returncode == 1
    assert 'BASELINE_CALLED_CANDIDATE' in codes(report)


def test_partial_workload_bypass_not_hidden_by_one_successful_candidate_call(tmp_path):
    _, report = run_probe(tmp_path, '''
        from base import Benchmark
        from candidate import run
        def test_op():
            Benchmark(lambda x: x, run, count=1).run()
            Benchmark(lambda x: x, lambda x: x, count=1).run()
    ''', benchmark=True)
    assert 'TIMING_CANDIDATE_NOT_CALLED' in codes(report)
    assert 'CANDIDATE_NOT_CALLED' not in codes(report)


@pytest.mark.parametrize('source,code', [
    ('def test_op():\n    assert True\n', 'CANDIDATE_NOT_CALLED'),
    ('import pytest\ndef test_op():\n    pytest.skip("not supported")\n', 'NO_EXECUTED_TESTS'),
    ('import pytest\n@pytest.mark.xfail\ndef test_op():\n    assert False\n', 'NO_EXECUTED_TESTS'),
    ('import pytest\n@pytest.fixture\ndef broken():\n    raise RuntimeError("setup")\ndef test_op(broken):\n    pass\n', 'PYTEST_FAILED'),
    ('import pytest\n@pytest.fixture\ndef broken():\n    yield\n    raise RuntimeError("teardown")\ndef test_op(broken):\n    pass\n', 'PYTEST_FAILED'),
])
def test_pytest_exit_zero_or_one_alone_is_not_acceptance(tmp_path, source, code):
    _, report = run_probe(tmp_path, source)
    assert code in codes(report)


def test_reference_import_side_effect_candidate_call_is_detected(tmp_path):
    _, report = run_probe(tmp_path, '''
        from candidate import run
        run(1)
        def test_op():
            pass
    ''', phase='reference')
    assert 'REFERENCE_CALLED_CANDIDATE' in codes(report)


def test_allowed_skip_does_not_hide_all_skip(tmp_path):
    expected = {'collected': ['test_op.py::test_op'], 'allowed_skips': ['test_op.py::test_op']}
    _, report = run_probe(tmp_path, '''
        import pytest
        def test_op():
            pytest.skip('unsupported')
    ''', expected=expected)
    assert 'NO_EXECUTED_TESTS' in codes(report)
    assert not any(x.startswith('UNREVIEWED_SKIP') for x in report['unverified'])


def test_legal_skip_with_executed_case_and_collection_identity(tmp_path):
    source = '''
        from candidate import run
        import pytest
        def test_op():
            run(1)
        def test_invalid_dim():
            pytest.skip('invalid dim')
    '''
    _, initial = run_probe(tmp_path, source)
    expected = {k: initial[k] for k in ('collected', 'identity', 'packages')}
    expected['allowed_skips'] = ['test_op.py::test_invalid_dim']
    _, report = run_probe(tmp_path, source, expected=expected)
    assert not report['findings']
    assert set(report['unverified']) == {'ACCURACY_REFERENCE_BOUNDARY_REQUIRES_REVIEW', 'FRAMEWORK_IMPORT_NOT_OBSERVED'}
    expected['collected'] = ['test_op.py::wrong_overload']
    _, changed = run_probe(tmp_path, source, expected=expected)
    assert 'COLLECTION_MISMATCH' in codes(changed)


def test_changed_source_or_environment_invalidates_review(tmp_path):
    source = 'from candidate import run\ndef test_op():\n    run(1)\n'
    _, initial = run_probe(tmp_path, source)
    expected = json.loads(json.dumps({k: initial[k] for k in ('collected', 'identity', 'packages')}))
    expected['packages']['torch'] = 'different-version'
    _, report = run_probe(tmp_path, source, expected=expected)
    assert 'SOURCE_OR_ENVIRONMENT_NOT_REVIEWED' in report['unverified']
    expected['packages'] = initial['packages']
    _, report = run_probe(tmp_path, source + '# changed\n', expected=expected)
    assert 'SOURCE_OR_ENVIRONMENT_NOT_REVIEWED' in report['unverified']


def test_source_change_during_candidate_execution_is_reported(tmp_path):
    _, report = run_probe(tmp_path, '''
        from pathlib import Path
        from candidate import run
        def test_op():
            run(1)
            Path('base.py').write_text('# overwritten')
    ''')
    assert 'SOURCE_OR_OBSERVER_CHANGED' in codes(report)


def test_unknown_native_timing_structure_is_not_guessed(tmp_path):
    base = BASE.replace('metric.latency_base =', 'metric.ref =').replace('metric.latency =', 'metric.cand =')
    _, report = run_probe(tmp_path, '''
        from base import Benchmark
        from candidate import run
        def test_op():
            Benchmark(lambda x: x, run).run()
    ''', benchmark=True, base=base)
    assert 'UNKNOWN_TIMING_ROLE' in report['unverified']
    assert 'INCOMPLETE_TIMING_PAIR' in codes(report)


def test_tensor_version_mutation_with_shared_storage_is_observable(tmp_path):
    # Duck-typed CPU fixture verifies observation logic, not vendor Tensor behavior.
    base = BASE.replace('self.get_latency(self.torch_op, i)', 'self.get_latency(self.torch_op, self.tensor)').replace('self.get_latency(self.gems_op, i)', 'self.get_latency(self.gems_op, self.tensor)')
    _, report = run_probe(tmp_path, '''
        import sys
        from types import SimpleNamespace
        from base import Benchmark
        from candidate import run
        class Tensor:
            _version = 0
            shape = (3,)
            dtype = 'float32'
            device = 'cpu'
            def stride(self): return (1,)
            def untyped_storage(self): return self
            def data_ptr(self): return 42
            def storage_offset(self): return 0
        def test_op():
            sys.modules['torch'] = SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor))
            def mutate(x): x._version += 1
            bench = Benchmark(mutate, run, count=1)
            bench.tensor = Tensor()
            bench.run()
    ''', benchmark=True, base=base)
    assert 'MUTATED_REFERENCE_INPUT_REUSED' in codes(report)


def test_skill_scripts_materialize_for_both_runtimes(tmp_path):
    from kernelgen.framework.agent_skills import canonical_agent_skills_dir, materialize_agent_skills
    materialize_agent_skills(canonical_agent_skills_dir(), tmp_path)
    for provider in ('.claude', '.agents'):
        for filename in ('review_source.py', 'probe_pytest.py'):
            assert (tmp_path / provider / 'skills/pytest-review/scripts' / filename).read_bytes() == (SCRIPTS / filename).read_bytes()


def test_fixture_candidate_call_cannot_hide_test_body_bypass(tmp_path):
    _, report = run_probe(tmp_path, '''
        import pytest
        from candidate import run
        @pytest.fixture
        def initialized():
            run(1)
        def test_op(initialized):
            pass
    ''')
    assert 'CANDIDATE_NOT_CALLED' in codes(report)


def test_source_tracks_import_alias_inside_baseline_helper(tmp_path):
    p = tmp_path / 'test_op.py'
    p.write_text('from flag_gems import _resize_output_ as resize\ndef dummy(x):\n    return resize(x)\nbench(torch_op=dummy)\n')
    assert load_script('review_source').scan(p)['findings'][0]['code'] == 'GEMS_BASELINE_REQUIRES_ISOLATION'


def test_nonfinite_timing_is_rejected(tmp_path):
    _, report = run_probe(tmp_path, '''
        from base import Benchmark
        from candidate import run
        def test_op():
            Benchmark(lambda x: x, run).run()
    ''', benchmark=True, base=BASE.replace('return 0.1', "return float('nan')"))
    assert 'INVALID_TIMING' in codes(report)


def test_importing_another_gems_checkout_is_reported(tmp_path):
    _, report = run_probe(tmp_path, '''
        import sys
        from types import SimpleNamespace
        from candidate import run
        sys.modules['flag_gems'] = SimpleNamespace(__file__='/other/checkout/flag_gems/__init__.py')
        def test_op():
            run(1)
    ''')
    assert 'FRAMEWORK_IMPORT_MISMATCH' in codes(report)


def test_profiler_json_creation_does_not_change_source_identity(tmp_path):
    proc, report = run_probe(tmp_path, """
        from pathlib import Path
        from candidate import run
        def test_op():
            output = Path('.flaggems_ascend_profile_other_job/ASCEND_PROFILER_OUTPUT/trace_view.json')
            output.parent.mkdir(parents=True)
            output.write_text('{}')
            assert run(1) == 1
    """, phase='reference-as-solution')
    assert proc.returncode == 0
    assert report['status'] == 'READY'
    assert report['identity'] == report['identity_after']
    assert report['observer_intact'] is True


@pytest.mark.parametrize('path', [
    '.flaggems_ascend_profile_other_job/helper.py',
    '.flaggems_ascend_profile_other_job/config.yaml',
    'src/config.json',
])
def test_profiler_exclusion_keeps_helpers_and_source_config(tmp_path, path):
    _, report = run_probe(tmp_path, f"""
        from pathlib import Path
        from candidate import run
        def test_op():
            output = Path({path!r})
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{{}}')
            assert run(1) == 1
    """, phase='reference-as-solution')
    assert report['status'] == 'NEEDS_FIX'
    assert 'SOURCE_OR_OBSERVER_CHANGED' in codes(report)
    assert report['observer_intact'] is True


def test_tracked_profiler_json_is_still_source(tmp_path):
    _, report = run_probe(tmp_path, """
        from pathlib import Path
        import subprocess
        from candidate import run
        def test_op():
            subprocess.run(['git', 'init', '-q'], check=True)
            output = Path('.flaggems_ascend_profile_config/settings.json')
            output.parent.mkdir()
            output.write_text('{}')
            subprocess.run(['git', 'add', '--', str(output)], check=True)
            assert run(1) == 1
    """, phase='reference-as-solution')
    assert report['status'] == 'NEEDS_FIX'
    assert '.flaggems_ascend_profile_config/settings.json' in report['identity_after']['files']
    assert 'SOURCE_OR_OBSERVER_CHANGED' in codes(report)


def test_observer_replacement_still_blocks(tmp_path):
    _, report = run_probe(tmp_path, """
        import sys
        from candidate import run
        def test_op():
            assert run(1) == 1
            sys.setprofile(None)
    """, phase='reference-as-solution')
    assert report['status'] == 'NEEDS_FIX'
    assert report['observer_intact'] is False
    assert 'SOURCE_OR_OBSERVER_CHANGED' in codes(report)


@pytest.mark.parametrize('directory, tracked, allowed', [
    ('.flaggems_ascend_profile_other', False, True),
    ('.flaggems_ascend_profile_other', True, False),
    ('src', False, False),
])
def test_disappearing_directory_during_identity_scan(tmp_path, monkeypatch, directory, tracked, allowed):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    probe = load_script('probe_pytest')
    candidate = tmp_path / 'candidate.py'
    candidate.write_text('def run(x): return x\n')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    if tracked:
        config = tmp_path / directory / 'config.json'
        config.parent.mkdir()
        config.write_text('{}')
        subprocess.run(['git', '-C', str(tmp_path), 'add', '--', str(config)], check=True)
    def racing_walk(root, onerror):
        onerror(FileNotFoundError(2, 'directory removed', str(root / directory)))
        yield str(root), [], ['candidate.py']
    monkeypatch.setattr(probe.os, 'walk', racing_walk)
    if allowed:
        assert 'candidate.py' in probe.source_identity(tmp_path, candidate)['files']
    else:
        with pytest.raises(FileNotFoundError):
            probe.source_identity(tmp_path, candidate)

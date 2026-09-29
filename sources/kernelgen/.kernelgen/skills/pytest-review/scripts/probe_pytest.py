#!/usr/bin/env python3
"""Observe native pytest execution; does not inject candidates or produce headline timing.

Run in a fresh target-side process, under the existing KGS Debug device slot.
Only pytest and Python stdlib are required by this helper.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from review_source import sha256


def is_profiler_artifact(relative_path, tracked_files):
    """Exclude generated Ascend JSON data, never tracked files or Python helpers."""
    path = Path(relative_path)
    return (len(path.parts) > 1
            and path.parts[0].startswith('.flaggems_ascend_profile_')
            and path.suffix == '.json'
            and path.as_posix() not in tracked_files)


def source_identity(root, candidate):
    """Bind source/config contents; concurrent profiler output is not source."""
    extensions = {'.py', '.ini', '.cfg', '.toml', '.yaml', '.yml', '.json'}
    try:
        tracked = subprocess.check_output(
            ['git', '-C', str(root), 'ls-files', '-z', '--cached'],
            stderr=subprocess.DEVNULL).decode().split('\0')
    except (OSError, subprocess.CalledProcessError):
        tracked = []
    tracked = set(tracked)
    def on_walk_error(error):
        # Another scheduled profiler may remove its temporary directory while
        # we traverse it. Real source and tracked paths must still fail closed.
        relative = Path(error.filename).relative_to(root)
        prefix = relative.as_posix() + '/'
        if (isinstance(error, FileNotFoundError)
                and relative.parts[0].startswith('.flaggems_ascend_profile_')
                and not any(name.startswith(prefix) for name in tracked)):
            return
        raise error

    files = {}
    paths = (Path(directory) / name
             for directory, _, names in os.walk(root, onerror=on_walk_error)
             for name in names)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if (path.suffix not in extensions
                or any(p in {'.git', '__pycache__', '.pytest_cache'} for p in relative.parts)
                or is_profiler_artifact(relative, tracked)):
            continue
        if path.is_file():
            files[relative.as_posix()] = sha256(path)
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'commit': commit, 'source_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'files': files, 'candidate_sha256': sha256(candidate)}


def phase_execution_findings(collected, reports):
    """Do not let a passing native phase hide an entirely skipped other phase."""
    executed = {r['nodeid'] for r in reports
                if r['when'] == 'call' and r['outcome'] in {'passed', 'failed'} and not r['xfail']}
    findings = []
    for phase, prefix in (('accuracy', 'tests/'), ('benchmark', 'benchmark/')):
        selected = {nodeid for nodeid in collected if nodeid.replace('\\', '/').startswith(prefix)}
        if selected and not selected.intersection(executed):
            findings.append({'code': 'NO_EXECUTED_TESTS',
                             'detail': {'phase': phase, 'collected': len(selected), 'executed': 0}})
    return findings


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def tensors(value):
    """Metadata/version observation only: no device copies, no input mutation."""
    torch = sys.modules.get('torch')
    result = []

    def visit(obj, label):
        if torch is not None and torch.is_tensor(obj):
            try:
                result.append({'path': label, 'object': id(obj), 'storage': obj.untyped_storage().data_ptr(),
                               'version': obj._version, 'shape': list(obj.shape), 'stride': list(obj.stride()),
                               'offset': obj.storage_offset(), 'dtype': str(obj.dtype), 'device': str(obj.device)})
            except (AttributeError, RuntimeError, NotImplementedError):
                result.append({'path': label, 'unavailable': True})
        elif isinstance(obj, (tuple, list)):
            for i, item in enumerate(obj):
                visit(item, label + f'[{i}]')
        elif isinstance(obj, dict):
            for key, item in obj.items():
                visit(item, label + f'[{key!r}]')
    visit(value, 'inputs')
    return result


def role_lines(path):
    """Recognize explicit metric assignments, not timing-call order or op identity."""
    result = {}
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != 'get_latency':
            continue
        roles = {t.attr for t in node.targets if isinstance(t, ast.Attribute)}
        role = 'reference' if roles == {'latency_base'} else 'candidate' if roles == {'latency'} else 'unknown'
        for line in range(call.lineno, call.end_lineno + 1):
            result[line] = role
    return result


class Observer:
    def __init__(self, root, candidate, entrypoint, phase, benchmark_file):
        self.root, self.candidate, self.entrypoint, self.phase = root, candidate, entrypoint, phase
        self.benchmark_file = benchmark_file
        self.roles = role_lines(benchmark_file) if benchmark_file else {}
        self.collected, self.reports, self.measurements, self.errors = [], [], [], []
        self.calls, self.stack = {}, {}
        self.call_start = {}
        self.active = None
        self.sequence = {}
        self.metrics = []  # Keep identities live so Python cannot reuse a completed case id.
        self.code_files = {}

    def profile(self, frame, event, arg):
        if event == 'call':
            filename = self.code_files.get(frame.f_code.co_filename)
            if filename is None:
                filename = Path(frame.f_code.co_filename).resolve()
                self.code_files[frame.f_code.co_filename] = filename
            if filename == self.candidate and frame.f_code.co_name == self.entrypoint:
                self.calls[self.active] = self.calls.get(self.active, 0) + 1
                for record in self.stack.values():
                    record['candidate_calls'] += 1
            if self.benchmark_file and frame.f_code.co_name == 'get_latency' and filename == self.benchmark_file:
                caller = frame.f_back
                role = self.roles.get(caller.f_lineno, 'unknown') if Path(caller.f_code.co_filename).resolve() == self.benchmark_file else 'unknown'
                # A metric object is fresh per native workload; keep a local ordinal too.
                key = (self.active, id(caller.f_locals.get('metric')))
                if key not in self.sequence:
                    self.sequence[key] = len(self.sequence)
                    self.metrics.append(caller.f_locals.get('metric'))
                op = frame.f_locals.get('op')
                self.stack[id(frame)] = {'nodeid': self.active, 'case': self.sequence[key], 'role': role,
                    'callsite': caller.f_lineno, 'candidate_calls': 0,
                    'op': {'module': getattr(op, '__module__', None), 'name': getattr(op, '__qualname__', type(op).__name__)},
                    'before': tensors((frame.f_locals.get('args', ()), frame.f_locals.get('kwargs', {})))}
        elif event == 'return' and id(frame) in self.stack:
            record = self.stack.pop(id(frame))
            record['after'] = tensors((frame.f_locals.get('args', ()), frame.f_locals.get('kwargs', {})))
            record['latency'] = arg if finite_positive(arg) else None
            self.measurements.append(record)

    def pytest_configure(self, config):
        # Includes options inherited through PYTEST_ADDOPTS and pytest.ini.
        if getattr(config.option, 'numprocesses', None) or getattr(config.option, 'forked', False):
            import pytest
            raise pytest.UsageError('pytest review probe requires serial execution without xdist/forked')
        if self.benchmark_file and config.getoption('level', default=None) != 'core':
            import pytest
            raise pytest.UsageError('pytest review benchmark requires --level core')

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            crash = str(getattr(getattr(report.longrepr, 'reprcrash', None), 'message', ''))
            # Pytest wraps import collection failures in a string rather than ReprExceptionInfo.
            if not crash:
                crash = str(report.longrepr).splitlines()[-1] if str(report.longrepr).splitlines() else ''
            missing_api = crash.startswith(('ImportError:', 'ModuleNotFoundError:', 'E   ImportError:', 'E   ModuleNotFoundError:'))
            self.errors.append({'code': 'API_UNAVAILABLE' if missing_api else 'COLLECTION_ERROR',
                                'nodeid': report.nodeid, 'stage': 'collection', 'detail': str(report.longrepr)})

    def pytest_runtest_makereport(self, item, call):
        if call.excinfo is not None:
            error = call.excinfo.value
            kind = type(error).__name__
            signal = None
            if isinstance(error, (ImportError, NotImplementedError)) or (
                isinstance(error, AttributeError) and isinstance(getattr(error, 'obj', None), ModuleType)
            ):
                signal = 'API_UNAVAILABLE'
            elif kind in {'CompilationError', 'CompileTimeAssertionFailure', 'OutOfResources', 'PTXASError'}:
                signal = 'COMPILATION_FAILED'
            if signal:
                self.errors.append({'code': signal, 'nodeid': item.nodeid, 'stage': call.when,
                                    'detail': f'{type(error).__module__}.{kind}: {error}'})

    def pytest_runtest_setup(self, item):
        self.active = item.nodeid

    def pytest_runtest_call(self, item):
        self.call_start[item.nodeid] = self.calls.get(item.nodeid, 0)

    def pytest_runtest_logreport(self, report):
        self.reports.append({'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome,
                             'xfail': bool(getattr(report, 'wasxfail', False)),
                             'reason': str(report.longrepr) if report.failed or report.skipped else '',
                             'candidate_calls': self.calls.get(report.nodeid, 0) - self.call_start.get(report.nodeid, 0) if report.when == 'call' else self.calls.get(report.nodeid, 0)})
        if report.when == 'teardown':
            self.active = None

    def summarize(self, exit_code, expected=None, allowed_skips=()):
        findings = list(self.errors)
        findings.extend(phase_execution_findings(self.collected, self.reports))
        unresolved = []
        passed = [r for r in self.reports if r['when'] == 'call' and r['outcome'] == 'passed' and not r['xfail']]
        skipped = [r for r in self.reports if r['outcome'] == 'skipped' or r['xfail']]
        def add(code, detail):
            findings.append({'code': code, 'detail': detail})
        if exit_code != 0 or any(r['outcome'] == 'failed' for r in self.reports):
            add('PYTEST_FAILED', f'exit_code={exit_code}; inspect setup/call/teardown reports')
        if not passed:
            add('NO_EXECUTED_TESTS', 'No successful non-xfail test calls; skips are not passes.')
        if expected is None:
            unresolved.append('EXPECTED_CASES_NOT_REVIEWED')
        elif sorted(self.collected) != sorted(expected):
            add('COLLECTION_MISMATCH', {'expected': expected, 'actual': self.collected})
        for record in skipped:
            if record['nodeid'] not in allowed_skips:
                unresolved.append('UNREVIEWED_SKIP:' + record['nodeid'])
        if self.phase == 'reference' and sum(self.calls.values()):
            add('REFERENCE_CALLED_CANDIDATE', dict(self.calls))
        if self.phase == 'candidate':
            for record in passed:
                if record['candidate_calls'] == 0:
                    add('CANDIDATE_NOT_CALLED', record['nodeid'])
        if self.benchmark_file:
            if not self.measurements:
                add('NO_TIMING_CASES', 'No recognized native get_latency execution.')
            groups = {}
            for record in self.measurements:
                key = (record['nodeid'], record['case'])
                groups.setdefault(key, []).append(record)
                if record['role'] == 'unknown':
                    unresolved.append('UNKNOWN_TIMING_ROLE')
                if record['role'] == 'reference' and record['candidate_calls']:
                    add('BASELINE_CALLED_CANDIDATE', {'nodeid': record['nodeid'], 'case': record['case']})
                if self.phase == 'candidate' and record['role'] == 'candidate' and not record['candidate_calls']:
                    add('TIMING_CANDIDATE_NOT_CALLED', {'nodeid': record['nodeid'], 'case': record['case']})
                if record['latency'] is None:
                    add('INVALID_TIMING', {'nodeid': record['nodeid'], 'case': record['case']})
            for key, records in groups.items():
                refs = [r for r in records if r['role'] == 'reference']
                candidates = [r for r in records if r['role'] == 'candidate']
                if len(refs) != 1 or len(candidates) != 1:
                    add('INCOMPLETE_TIMING_PAIR', {'nodeid': key[0], 'case': key[1]})
                    continue
                before = {v['object']: v for v in refs[0]['before'] if 'object' in v}
                changed = [v for v in refs[0]['after'] if 'object' in v and v != before.get(v['object'])]
                used = {v['storage'] for v in candidates[0]['before'] if 'storage' in v and v['storage'] != 0}
                if any(v['storage'] in used for v in changed):
                    add('MUTATED_REFERENCE_INPUT_REUSED', {'nodeid': key[0], 'case': key[1]})
            # _version does not see raw Triton writes: never certify input independence.
            unresolved.append('INPUT_VALUE_INDEPENDENCE_REQUIRES_REVIEW')
        else:
            unresolved.append('ACCURACY_REFERENCE_BOUNDARY_REQUIRES_REVIEW')
        return {'status': 'ISSUES_FOUND' if findings else 'REVIEW_REQUIRED' if unresolved else 'CHECKS_PASSED',
                'findings': findings, 'unverified': sorted(set(unresolved)), 'passed_calls': len(passed),
                'scope': 'Observed routes only; not numerical-reference approval or trustworthy headline timing.'}


def readiness_result(result):
    """Minimal preparation verdict; coverage suggestions do not block generation.

    No numerical-oracle claim is inferred from reference self-consistency.
    The legacy audit evidence remains available in the same report.
    """
    advisory = {'COLLECTION_MISMATCH', 'INVALID_TIMING'}
    injection = {'CANDIDATE_NOT_CALLED', 'TIMING_CANDIDATE_NOT_CALLED',
                 'BASELINE_CALLED_CANDIDATE', 'REFERENCE_CALLED_CANDIDATE', 'FRAMEWORK_IMPORT_MISMATCH'}
    missing_proof = {'FRAMEWORK_IMPORT_NOT_OBSERVED', 'UNKNOWN_TIMING_ROLE'}
    problems, suggestions = [], []
    for finding in result['findings']:
        code = finding['code']
        if code in advisory:
            suggestions.append({'priority': 'P1', **finding})
            continue
        signal = 'CANDIDATE_NOT_INJECTED' if code in injection else code if code in {
            'API_UNAVAILABLE', 'COMPILATION_FAILED', 'REFERENCE_SEMANTICS_INVALID',
        } else None
        problems.append({'signal': signal, 'priority': 'P0' if signal else None,
                         'evidence': finding, 'responsibility': 'unresolved',
                         'next_action': 'Fix the preparation failure or diagnose the recorded stage; do not patch the candidate to compensate.'})
    for code in result['unverified']:
        if code in missing_proof:
            problems.append({'signal': None, 'priority': None, 'evidence': {'code': code},
                             'responsibility': 'test preparation', 'next_action': 'Restore execution identity evidence.'})
        else:
            suggestions.append({'priority': 'P1', 'code': code})
    return {'status': 'NEEDS_FIX' if problems else 'READY', 'problems': problems, 'suggestions': suggestions,
            'scope': 'Reference-as-solution preparation only; no numerical-reference certification, performance acceptance or runtime BLOCK.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gems-root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--entrypoint', default='run')
    parser.add_argument('--phase', choices=['reference', 'candidate', 'reference-as-solution'], required=True)
    parser.add_argument('--reference-source', type=Path, help='Prepared reference solution source; required for reference-as-solution')
    parser.add_argument('--benchmark-file', type=Path)
    parser.add_argument('--expect', type=Path, help='Reviewed JSON: collected nodeids and allowed_skips')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root, candidate, output = args.gems_root.resolve(), args.candidate.resolve(), args.output.resolve()
    benchmark = args.benchmark_file.resolve() if args.benchmark_file else None
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ['--'] else args.pytest_args
    if output.is_relative_to(root):
        parser.error('--output must be outside --gems-root to avoid changing reviewed sources')
    if not root.is_dir() or not candidate.is_file() or (benchmark and (not benchmark.is_file() or not benchmark.is_relative_to(root))):
        parser.error('gems root/candidate must exist; benchmark file must be inside gems root')
    reference_source = args.reference_source.resolve() if args.reference_source else None
    if args.phase == 'reference-as-solution' and (reference_source is None or not reference_source.is_file()):
        parser.error('reference-as-solution requires an existing --reference-source')
    if any(x in pytest_args for x in ('-n', '--numprocesses', '--forked')) or any(x.startswith(('-n=', '--numprocesses=')) for x in pytest_args):
        parser.error('probe requires serial pytest in one process')
    if benchmark:
        # Do not silently accept --level comprehensive supplied by a caller.
        for i, value in enumerate(pytest_args):
            if value.startswith('--level=') and value != '--level=core':
                parser.error('native benchmark review requires --level core')
            if value == '--level' and (i + 1 == len(pytest_args) or pytest_args[i + 1] != 'core'):
                parser.error('native benchmark review requires --level core')
        if '--level' not in pytest_args and '--level=core' not in pytest_args:
            pytest_args += ['--level', 'core']
    expected = json.loads(args.expect.read_text()) if args.expect else {}
    before = source_identity(root, candidate)
    observer = Observer(root, candidate, args.entrypoint, 'candidate' if args.phase == 'reference-as-solution' else args.phase, benchmark)
    reference_sha = sha256(reference_source) if reference_source else None
    if args.phase == 'reference-as-solution' and reference_sha != before['candidate_sha256']:
        result = {'status': 'NEEDS_FIX', 'schema': 'pytest-review-readiness/v1',
                  'phase': args.phase, 'reference_source': str(reference_source), 'problems': [{'signal': None,
                  'evidence': {'code': 'REFERENCE_SOLUTION_MISMATCH', 'reference_sha256': reference_sha,
                               'candidate_sha256': before['candidate_sha256']},
                  'responsibility': 'test preparation', 'next_action': 'Inject the prepared reference source unchanged.'}],
                  'suggestions': [], 'exit_code': None, 'execution_started': False}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
        return 1
    import pytest
    previous = sys.getprofile()
    if previous is not None:
        parser.error('an existing profiler is active; use a fresh process')
    # Observe imports and fixture calls as well as test bodies. No candidate import/injection here.
    os.chdir(root)
    sys.setprofile(observer.profile)
    try:
        exit_code = int(pytest.main(pytest_args, plugins=[observer]))
    finally:
        profile_intact = sys.getprofile() == observer.profile
        sys.setprofile(previous)
    after = source_identity(root, candidate)
    result = observer.summarize(exit_code, expected.get('collected'), expected.get('allowed_skips', []))
    if before != after or not profile_intact:
        result['findings'].append({'code': 'SOURCE_OR_OBSERVER_CHANGED', 'detail': {'source_changed': before != after, 'observer_intact': profile_intact}})
        result['status'] = 'ISSUES_FOUND'
    if reference_source and sha256(reference_source) != reference_sha:
        result['findings'].append({'code': 'REFERENCE_SOURCE_CHANGED', 'detail': str(reference_source)})
        result['status'] = 'ISSUES_FOUND'
    versions = {}
    for package in ('pytest', 'torch', 'triton', 'flagtree', 'torch-npu', 'torch-musa'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if expected.get('identity') != before or expected.get('packages') != versions:
        result['unverified'].append('SOURCE_OR_ENVIRONMENT_NOT_REVIEWED')
        if result['status'] != 'ISSUES_FOUND':
            result['status'] = 'REVIEW_REQUIRED'
    framework_file = getattr(sys.modules.get('flag_gems'), '__file__', None)
    if framework_file and not Path(framework_file).resolve().is_relative_to(root):
        result['findings'].append({'code': 'FRAMEWORK_IMPORT_MISMATCH', 'detail': str(framework_file)})
        result['status'] = 'ISSUES_FOUND'
    elif not framework_file:
        result['unverified'].append('FRAMEWORK_IMPORT_NOT_OBSERVED')
        if result['status'] != 'ISSUES_FOUND':
            result['status'] = 'REVIEW_REQUIRED'
    result.update(framework_file=framework_file, observer_sha256=sha256(Path(__file__)), entrypoint=args.entrypoint,
                  schema='pytest-review-probe/v1', phase=args.phase, argv=pytest_args, identity=before,
                  identity_after=after, observer_intact=profile_intact, python=sys.version, packages=versions, collected=observer.collected,
                  reports=observer.reports, measurements=observer.measurements, exit_code=exit_code)
    if args.phase == 'reference-as-solution':
        result['audit_status'] = result['status']
        result.update(readiness_result(result), reference_source=str(reference_source), reference_sha256=reference_sha,
                      schema='pytest-review-readiness/v1')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return 1 if result['status'] in {'ISSUES_FOUND', 'NEEDS_FIX'} else 2 if result['status'] == 'REVIEW_REQUIRED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
